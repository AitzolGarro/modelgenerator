"""
Health check and GPU info endpoints.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.job import (
    HealthResponse,
    ANIMATION_PRESETS,
    validate_params_for_gpu,
)
from app.services.factory import get_gpu_info

router = APIRouter(tags=["health"])
settings = get_settings()

# Cache GPU info at module level (doesn't change during runtime)
_gpu_info: dict | None = None


def _get_gpu_memory_mb() -> int | None:
    """Get GPU total VRAM in MB (cached)."""
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = get_gpu_info()
    if _gpu_info.get("available") and _gpu_info.get("memory_total") is not None:
        return int(_gpu_info["memory_total"]) // (1024 * 1024)
    return None


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check with GPU status."""
    gpu = _gpu_info or get_gpu_info()
    memory_total_mb = _get_gpu_memory_mb()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        gpu_available=gpu["available"],
        gpu_name=gpu.get("name"),
        gpu_memory_total_mb=memory_total_mb,
    )


@router.get("/animation-presets")
def get_animation_presets():
    """Return animation presets with availability based on detected GPU.

    Each preset includes whether it's available on this GPU and a
    warning message if VRAM is tight.
    """
    gpu_memory_mb = _get_gpu_memory_mb()
    gpu = _gpu_info or get_gpu_info()

    result = []
    for key, preset in ANIMATION_PRESETS.items():
        check = validate_params_for_gpu(
            num_frames=preset["num_frames"],
            resolution=preset["anim_resolution"],
            gpu_memory_total_mb=gpu_memory_mb,
        )
        result.append({
            "key": key,
            "label": preset["label"],
            "description": preset["description"],
            "num_frames": preset["num_frames"],
            "anim_inference_steps": preset["anim_inference_steps"],
            "anim_guidance_scale": preset["anim_guidance_scale"],
            "anim_resolution": preset["anim_resolution"],
            "available": check["ok"],
            "warning": check.get("message", ""),
            "estimated_vram_mb": check["estimated_mb"],
        })

    return {
        "presets": result,
        "gpu_name": gpu.get("name"),
        "gpu_memory_total_mb": gpu_memory_mb,
    }
