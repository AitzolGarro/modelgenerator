"""
Job API endpoints.
Supports: generate, animate, refine, scene job types.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.models.job import Job, JobType, JobStatus
from app.schemas.job import JobCreate, JobResponse, JobListResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])
settings = get_settings()

VALID_JOB_TYPES = {t.value for t in JobType}


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
    return response


@router.post("", response_model=JobResponse, status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    """Create a new job (generate, scene)."""
    if payload.job_type not in VALID_JOB_TYPES:
        raise HTTPException(400, f"Invalid job_type: {payload.job_type}. Valid: {VALID_JOB_TYPES}")

    input_file_path = None

    # For animate/refine/skin: resolve source from an existing job
    if payload.job_type in ("animate", "refine", "skin"):
        if not payload.source_job_id:
            raise HTTPException(400, f"source_job_id required for {payload.job_type} jobs")
        source_job = db.query(Job).filter(Job.id == payload.source_job_id).first()
        if not source_job:
            raise HTTPException(404, f"Source job {payload.source_job_id} not found")
        if not source_job.export_path:
            raise HTTPException(400, f"Source job {payload.source_job_id} has no exported GLB")
        input_file_path = source_job.export_path

    job = Job(
        job_type=payload.job_type,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        num_steps=payload.num_steps,
        guidance_scale=payload.guidance_scale,
        seed=payload.seed,
        input_file_path=input_file_path,
        status=JobStatus.PENDING,
    )
    db.add(job)
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
