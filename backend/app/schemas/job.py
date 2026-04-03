"""
Pydantic schemas for job API.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    """Request to create a new generation job."""
    job_type: str = Field("generate", description="Job type: generate, animate, refine, scene")
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt")
    negative_prompt: str | None = Field(None, max_length=2000)
    num_steps: int = Field(30, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    seed: int | None = Field(None, ge=0)
    # For animate/refine: reference to an existing job's GLB or uploaded file
    source_job_id: int | None = Field(None, description="Source job ID to use its GLB as input")


class JobResponse(BaseModel):
    """Full job representation."""
    id: int
    job_type: str = "generate"
    prompt: str
    negative_prompt: str | None = None
    status: str
    input_file_path: str | None = None
    image_path: str | None = None
    model_path: str | None = None
    textured_model_path: str | None = None
    export_path: str | None = None
    error_message: str | None = None
    num_steps: int
    guidance_scale: float
    seed: int | None = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    # Computed URLs
    input_file_url: str | None = None
    image_url: str | None = None
    model_url: str | None = None
    export_url: str | None = None

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    """Paginated list of jobs."""
    jobs: list[JobResponse]
    total: int
    page: int
    page_size: int


class JobStatusUpdate(BaseModel):
    id: int
    status: str
    progress_message: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    gpu_available: bool = False
    gpu_name: str | None = None
