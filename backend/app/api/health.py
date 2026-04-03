"""
Health check endpoint.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.job import HealthResponse
from app.services.factory import get_gpu_info

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check with GPU status."""
    gpu = get_gpu_info()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        gpu_available=gpu["available"],
        gpu_name=gpu.get("name"),
    )
