"""
Job API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.models.job import Job, JobStatus
from app.schemas.job import JobCreate, JobResponse, JobListResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])
settings = get_settings()


def _job_to_response(job: Job) -> JobResponse:
    """Convert a Job model to a JobResponse with computed URLs."""
    response = JobResponse.model_validate(job)

    # Build download/preview URLs
    if job.image_path:
        response.image_url = f"/api/v1/files/{job.image_path}"
    if job.model_path:
        response.model_url = f"/api/v1/files/{job.model_path}"
    if job.export_path:
        response.export_url = f"/api/v1/files/{job.export_path}"

    return response


@router.post("", response_model=JobResponse, status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    """Create a new generation job."""
    job = Job(
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        num_steps=payload.num_steps,
        guidance_scale=payload.guidance_scale,
        seed=payload.seed,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return _job_to_response(job)


@router.get("", response_model=JobListResponse)
def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    """List jobs with pagination."""
    query = db.query(Job)

    if status:
        try:
            status_enum = JobStatus(status)
            query = query.filter(Job.status == status_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

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
    """Get a single job by ID."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return _job_to_response(job)


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a job. Only allowed for completed or failed jobs."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(
            400, f"Cannot delete job in status {job.status.value}. Wait for completion."
        )

    db.delete(job)
    db.commit()


@router.post("/{job_id}/retry", response_model=JobResponse)
def retry_job(job_id: int, db: Session = Depends(get_db)):
    """Retry a failed job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job.status != JobStatus.FAILED:
        raise HTTPException(400, "Only failed jobs can be retried")

    max_retries = settings.WORKER_MAX_RETRIES
    if job.retry_count >= max_retries:
        raise HTTPException(400, f"Max retries ({max_retries}) exceeded")

    job.status = JobStatus.PENDING
    job.error_message = None
    db.commit()
    db.refresh(job)

    return _job_to_response(job)
