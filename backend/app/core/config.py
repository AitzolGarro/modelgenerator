"""
Application configuration.
All settings are loaded from environment variables with sensible defaults.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # --- App ---
    APP_NAME: str = "ModelGenerator"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # --- Server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./storage/modelgenerator.db"

    # --- Storage ---
    STORAGE_ROOT: Path = Path("./storage")
    IMAGES_DIR: str = "images"
    MODELS_DIR: str = "models"
    EXPORTS_DIR: str = "exports"

    # --- ML: Text-to-Image ---
    TEXT_TO_IMAGE_MODEL: str = "stabilityai/stable-diffusion-xl-base-1.0"
    TEXT_TO_IMAGE_DEVICE: str = "cuda"
    TEXT_TO_IMAGE_DTYPE: str = "float16"
    IMAGE_WIDTH: int = 1024
    IMAGE_HEIGHT: int = 1024
    IMAGE_NUM_STEPS: int = 30
    IMAGE_GUIDANCE_SCALE: float = 7.5

    # --- ML: Image-to-3D Backend ---
    IMAGE_TO_3D_BACKEND: str = "auto"  # "auto", "instantmesh", "triposr"

    # --- ML: Image-to-3D (InstantMesh) ---
    INSTANTMESH_MODEL: str = "TencentARC/InstantMesh"
    INSTANTMESH_CONFIG: str = "instant-mesh-large"
    INSTANTMESH_DEVICE: str = "cuda"
    INSTANTMESH_USE_TEXTURE_MAP: bool = True  # requires nvdiffrast
    INSTANTMESH_MC_RESOLUTION: int = 256

    # --- ML: Image-to-3D (TripoSR) ---
    TRIPOSR_MODEL: str = "stabilityai/TripoSR"
    TRIPOSR_DEVICE: str = "cuda"
    TRIPOSR_CHUNK_SIZE: int = 8192
    TRIPOSR_MC_RESOLUTION: int = 256

    # --- ML: Texturing ---
    TEXTURING_ENABLED: bool = True
    TEXTURE_RESOLUTION: int = 1024

    # --- ML: UV Texturing Pipeline ---
    UV_TEXTURE_VIEWS: int = 6
    UV_TEXTURE_RESOLUTION: int = 1024
    CONTROLNET_DEPTH_MODEL: str = "diffusers/controlnet-depth-sdxl-1.0"
    CONTROLNET_CONDITIONING_SCALE: float = 0.8
    UV_SEAM_BLEND_RADIUS: int = 3
    UV_METHOD: str = "auto"  # "xatlas" | "lscm" | "auto" (xatlas first, lscm fallback)

    # --- 2D Generation ---
    CHARACTER_2D_WIDTH: int = 1024
    CHARACTER_2D_HEIGHT: int = 1536          # taller for full-body characters
    SPRITE_SHEET_FRAME_SIZE: int = 512
    SPRITE_SHEET_FPS: int = 12
    SAM_MODEL_TYPE: str = "vit_b"
    SAM_CHECKPOINT: str = ""                 # auto-downloaded if empty
    STYLE_2D_DEFAULT: str = "anime"          # default style preset

    # --- Export ---
    EXPORT_FORMAT: str = "glb"  # glb | obj | both

    # --- Worker ---
    WORKER_POLL_INTERVAL: float = 2.0
    WORKER_MAX_RETRIES: int = 2

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    @property
    def images_path(self) -> Path:
        return self.STORAGE_ROOT / self.IMAGES_DIR

    @property
    def models_path(self) -> Path:
        return self.STORAGE_ROOT / self.MODELS_DIR

    @property
    def exports_path(self) -> Path:
        return self.STORAGE_ROOT / self.EXPORTS_DIR

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        for d in [self.images_path, self.models_path, self.exports_path]:
            d.mkdir(parents=True, exist_ok=True)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
