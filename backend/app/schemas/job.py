"""
Pydantic schemas for job API.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class JobCreate(BaseModel):
    """Request to create a new generation job."""
    job_type: str = Field(
        "generate",
        description="Job type: generate, animate, refine, scene, skin, generate_2d, animate_2d",
    )
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt")
    negative_prompt: str | None = Field(None, max_length=2000)
    num_steps: int = Field(30, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    seed: int | None = Field(None, ge=0)
    # For animate/refine/skin: reference to an existing job's GLB or uploaded file
    source_job_id: int | None = Field(None, description="Source job ID to use its output as input")
    # 2D-specific
    style: str | None = Field(
        None,
        description="2D style preset: anime, pixel_art, cartoon, realistic, chibi, comic",
    )
    # Animation parameters (animate_2d / Wan2.1)
    num_frames: int | None = Field(
        None,
        description="Wan2.1 frame count (must satisfy 4k+1 rule, range 17–201)",
    )
    anim_inference_steps: int | None = Field(None, ge=10, le=100)
    anim_guidance_scale: float | None = Field(None, ge=1.0, le=20.0)
    anim_resolution: Literal["480p", "720p"] | None = Field(None)
    enhance_animation: bool | None = Field(None)
    enhance_personality: Literal["calm", "aggressive", "heavy", "light"] | None = Field(None)
    enhance_intensity: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("num_frames")
    @classmethod
    def validate_4k_plus_1(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if (v - 1) % 4 != 0:
            raise ValueError(
                f"num_frames must satisfy the 4k+1 rule (got {v}; "
                f"valid examples: 17, 21, 25, 29, 33, …)"
            )
        if not (17 <= v <= 201):
            raise ValueError(f"num_frames must be between 17 and 201 (got {v})")
        return v


# ── Animation presets ────────────────────────────────────────────────────────
# Each preset auto-configures params based on GPU VRAM tier.
# The frontend shows these as simple choices for non-technical users.

ANIMATION_PRESETS: dict[str, dict] = {
    "rapido": {
        "label": "Rápido",
        "description": "Generación rápida, calidad aceptable",
        "num_frames": 17,
        "anim_inference_steps": 20,
        "anim_guidance_scale": 7.0,
        "anim_resolution": "480p",
        "min_vram_mb": 16000,
        "estimated_time_factor": 1.0,
    },
    "calidad": {
        "label": "Calidad",
        "description": "Buen balance entre calidad y tiempo",
        "num_frames": 33,
        "anim_inference_steps": 30,
        "anim_guidance_scale": 9.0,
        "anim_resolution": "480p",
        "min_vram_mb": 20000,
        "estimated_time_factor": 2.8,
    },
    "alta": {
        "label": "Alta calidad",
        "description": "Máxima calidad, más tiempo de generación",
        "num_frames": 49,
        "anim_inference_steps": 40,
        "anim_guidance_scale": 9.0,
        "anim_resolution": "480p",
        "min_vram_mb": 24000,
        "estimated_time_factor": 5.5,
    },
    "cinematico": {
        "label": "Cinemático",
        "description": "Máxima calidad y resolución (requiere GPU potente)",
        "num_frames": 49,
        "anim_inference_steps": 40,
        "anim_guidance_scale": 9.0,
        "anim_resolution": "720p",
        "min_vram_mb": 30000,
        "estimated_time_factor": 10.0,
    },
}

# ── VRAM estimation constants ────────────────────────────────────────────────
VRAM_BASE_MB: int = 15360  # Wan2.1 14B int4 base footprint
VRAM_FRAME_MB: dict[str, float] = {
    "480p": 6.4,
    "720p": 14.7,
}


def estimate_vram_mb(
    num_frames: int = 33,
    resolution: str = "480p",
) -> float:
    """Estimate total VRAM needed for given animation params."""
    frame_overhead = VRAM_FRAME_MB.get(resolution, 6.4)
    return VRAM_BASE_MB + num_frames * frame_overhead


def validate_params_for_gpu(
    num_frames: int,
    resolution: str,
    gpu_memory_total_mb: int | None,
) -> dict:
    """Validate that animation params won't OOM on the detected GPU.

    Returns {"ok": True/False, "estimated_mb": ..., "available_mb": ..., "message": ...}
    """
    estimated = estimate_vram_mb(num_frames, resolution)
    result = {
        "estimated_mb": round(estimated),
        "available_mb": gpu_memory_total_mb,
    }
    if gpu_memory_total_mb is None:
        result["ok"] = True
        result["message"] = "No se pudo detectar la GPU — no se puede validar"
        return result

    if estimated > gpu_memory_total_mb:
        result["ok"] = False
        result["message"] = (
            f"Estos parámetros necesitan ~{round(estimated / 1024, 1)} GB VRAM "
            f"pero tu GPU solo tiene {round(gpu_memory_total_mb / 1024, 1)} GB. "
            f"Reduce los fotogramas o usa resolución 480p."
        )
    elif estimated > gpu_memory_total_mb * 0.90:
        result["ok"] = True
        result["message"] = (
            f"⚠️ Uso estimado: ~{round(estimated / 1024, 1)} GB de "
            f"{round(gpu_memory_total_mb / 1024, 1)} GB — puede ser ajustado"
        )
    else:
        result["ok"] = True
        result["message"] = ""
    return result


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
    # 2D fields
    style: str | None = None
    sprite_sheet_path: str | None = None
    model_json_path: str | None = None
    # Animation parameters (animate_2d / Wan2.1)
    num_frames: int | None = None
    anim_inference_steps: int | None = None
    anim_guidance_scale: float | None = None
    anim_resolution: str | None = None
    enhance_animation: int | None = None  # SQLite stores bool as int
    enhance_personality: str | None = None
    enhance_intensity: float | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    # Computed URLs
    input_file_url: str | None = None
    image_url: str | None = None
    model_url: str | None = None
    export_url: str | None = None
    # 2D-specific computed URLs (set from sprite_sheet_path / model_json_path)
    sprite_sheet_url: str | None = None
    model_json_url: str | None = None

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
    gpu_memory_total_mb: int | None = None
