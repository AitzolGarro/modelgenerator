"""
Job API endpoints.
Supports: generate, animate, refine, scene, skin, generate_2d, animate_2d job types.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.models.job import Job, JobType, JobStatus
from app.schemas.job import JobCreate, JobResponse, JobListResponse, validate_params_for_gpu

router = APIRouter(prefix="/jobs", tags=["jobs"])
settings = get_settings()

VALID_JOB_TYPES = {t.value for t in JobType}


_VALID_2D_STYLES = {"anime", "pixel_art", "cartoon", "realistic", "chibi", "comic"}


def _job_to_response(job: Job) -> JobResponse:
    response = JobResponse.model_validate(job)
    if job.image_path:
        response.image_url = f"/api/v1/files/{job.image_path}"
    if job.model_path:
        response.model_url = f"/api/v1/files/{job.model_path}"
    if job.export_path:
        response.export_url = f"/api/v1/files/{job.export_path}"
    if job.input_file_path:
        response.input_file_url = f"/api/v1/files/{job.input_file_path}"
    # 2D-specific
    if job.sprite_sheet_path:
        response.sprite_sheet_url = f"/api/v1/files/{job.sprite_sheet_path}"
    if job.model_json_path:
        response.model_json_url = f"/api/v1/files/{job.model_json_path}"
    return response


@router.post("", response_model=JobResponse, status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    """Create a new job (generate, scene, generate_2d, animate_2d, …)."""
    if payload.job_type not in VALID_JOB_TYPES:
        raise HTTPException(400, f"Invalid job_type: {payload.job_type}. Valid: {VALID_JOB_TYPES}")

    input_file_path = None
    style = None

    # For animate/refine/skin: resolve source GLB from an existing job
    if payload.job_type in ("animate", "refine", "skin"):
        if not payload.source_job_id:
            raise HTTPException(400, f"source_job_id required for {payload.job_type} jobs")
        source_job = db.query(Job).filter(Job.id == payload.source_job_id).first()
        if not source_job:
            raise HTTPException(404, f"Source job {payload.source_job_id} not found")
        if not source_job.export_path:
            raise HTTPException(400, f"Source job {payload.source_job_id} has no exported GLB")
        input_file_path = source_job.export_path

    # --- 2D: generate_2d ---
    elif payload.job_type == "generate_2d":
        raw_style = (payload.style or settings.STYLE_2D_DEFAULT).strip().lower()
        if raw_style not in _VALID_2D_STYLES:
            raise HTTPException(
                400,
                f"Invalid style {raw_style!r}. Valid styles: {sorted(_VALID_2D_STYLES)}",
            )
        style = raw_style

    # --- 2D: animate_2d ---
    elif payload.job_type == "animate_2d":
        if not payload.source_job_id:
            raise HTTPException(400, "source_job_id required for animate_2d jobs (must point to a generate_2d job)")
        source_job = db.query(Job).filter(Job.id == payload.source_job_id).first()
        if not source_job:
            raise HTTPException(404, f"Source job {payload.source_job_id} not found")
        if source_job.job_type != JobType.GENERATE_2D.value:
            raise HTTPException(
                400,
                f"Source job {payload.source_job_id} must be a generate_2d job "
                f"(got {source_job.job_type})"
            )
        if source_job.status != JobStatus.COMPLETED.value and source_job.status != "completed":
            raise HTTPException(
                400,
                f"Source job {payload.source_job_id} is not completed yet (status: {source_job.status})"
            )
        if not source_job.model_json_path:
            raise HTTPException(400, f"Source job {payload.source_job_id} has no 2D model JSON")

        # ★ Server-side VRAM validation: reject params that would OOM
        _num_frames = payload.num_frames or 33
        _resolution = payload.anim_resolution or "480p"
        from app.api.health import _get_gpu_memory_mb
        gpu_mem = _get_gpu_memory_mb()
        vram_check = validate_params_for_gpu(_num_frames, _resolution, gpu_mem)
        if not vram_check["ok"]:
            raise HTTPException(
                422,
                f"Parámetros de animación rechazados: {vram_check['message']}"
            )

    job = Job(
        job_type=payload.job_type,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        num_steps=payload.num_steps,
        guidance_scale=payload.guidance_scale,
        seed=payload.seed,
        input_file_path=input_file_path,
        style=style,
        status=JobStatus.PENDING,
        # Animation parameters (animate_2d / Wan2.1) — None when not provided
        num_frames=payload.num_frames,
        anim_inference_steps=payload.anim_inference_steps,
        anim_guidance_scale=payload.anim_guidance_scale,
        anim_resolution=payload.anim_resolution,
        enhance_animation=(
            int(payload.enhance_animation)
            if payload.enhance_animation is not None
            else None
        ),
        enhance_personality=payload.enhance_personality,
        enhance_intensity=payload.enhance_intensity,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # For animate_2d: link source_job_id in job.input_file_path as a marker
    if payload.job_type == "animate_2d" and payload.source_job_id:
        job.input_file_path = f"__2d_source_job:{payload.source_job_id}"
        db.commit()
        db.refresh(job)

    return _job_to_response(job)


@router.post("/upload", response_model=JobResponse, status_code=201)
async def create_job_with_upload(
    file: UploadFile = File(..., description="GLB file to animate, refine, or texture"),
    job_type: str = Form(..., description="animate, refine, or skin"),
    prompt: str = Form(..., description="Animation description, refinement instructions, or texture description"),
    negative_prompt: str = Form(None),
    db: Session = Depends(get_db),
):
    """Create an animate/refine/skin job with a file upload instead of source_job_id."""
    if job_type not in ("animate", "refine", "skin"):
        raise HTTPException(400, "Upload endpoint only supports animate/refine/skin job types")

    if not file.filename or not file.filename.lower().endswith(".glb"):
        raise HTTPException(400, "Only .glb files accepted")

    # Save uploaded file
    uploads_dir = settings.STORAGE_ROOT / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Create job first to get ID
    job = Job(
        job_type=job_type,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_steps=30,
        guidance_scale=7.5,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Save file with job ID
    upload_dir = uploads_dir / str(job.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / "input.glb"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Store relative path
    job.input_file_path = str(file_path.relative_to(settings.STORAGE_ROOT))
    db.commit()
    db.refresh(job)

    return _job_to_response(job)


@router.get("", response_model=JobListResponse)
def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    job_type: str | None = Query(None, description="Filter by job type"),
    db: Session = Depends(get_db),
):
    """List jobs with pagination and filters."""
    query = db.query(Job)

    if status:
        try:
            query = query.filter(Job.status == JobStatus(status))
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    if job_type:
        if job_type not in VALID_JOB_TYPES:
            raise HTTPException(400, f"Invalid job_type: {job_type}")
        query = query.filter(Job.job_type == job_type)

    total = query.count()
    jobs = (
        query.order_by(Job.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return _job_to_response(job)


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(400, "Can only delete completed or failed jobs")
    db.delete(job)
    db.commit()


@router.post("/{job_id}/retry", response_model=JobResponse)
def retry_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.FAILED:
        raise HTTPException(400, "Only failed jobs can be retried")
    if job.retry_count >= settings.WORKER_MAX_RETRIES:
        raise HTTPException(400, f"Max retries ({settings.WORKER_MAX_RETRIES}) exceeded")

    job.status = JobStatus.PENDING
    job.error_message = None
    db.commit()
    db.refresh(job)
    return _job_to_response(job)
