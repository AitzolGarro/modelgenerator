"""
Pydantic schemas for job API.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    """Request to create a new generation job."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for 3D model generation")
    negative_prompt: str | None = Field(None, max_length=2000, description="Negative prompt to guide generation")
    num_steps: int = Field(30, ge=1, le=150, description="Number of diffusion steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description="Classifier-free guidance scale")
    seed: int | None = Field(None, ge=0, description="Random seed for reproducibility")


class JobResponse(BaseModel):
    """Full job representation."""

    id: int
    prompt: str
    negative_prompt: str | None = None
    status: str
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

    # Computed URLs for frontend
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
    """Status update for SSE or polling."""

    id: int
    status: str
    progress_message: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    gpu_available: bool = False
    gpu_name: str | None = None
