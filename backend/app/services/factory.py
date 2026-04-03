"""
Service factory: creates the right implementation based on config and availability.
"""

import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import (
    TextToImageService,
    ImageTo3DService,
    TexturingService,
    ExportService,
    AssetStorageService,
)

logger = get_logger(__name__)
settings = get_settings()


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_triposr() -> bool:
    try:
        from app.services.tsr_local.system import TSR  # noqa: F401
        return True
    except Exception:
        return False


def _has_diffusers() -> bool:
    try:
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def create_text_to_image_service() -> TextToImageService:
    """Create text-to-image service based on available dependencies."""
    if _has_cuda() and _has_diffusers():
        from app.services.text_to_image import SDXLTextToImageService
        logger.info("Using SDXL text-to-image service (GPU)")
        return SDXLTextToImageService()
    else:
        from app.services.text_to_image import MockTextToImageService
        logger.warning(
            "CUDA or diffusers not available. Using mock text-to-image service. "
            "Install: pip install torch diffusers transformers accelerate"
        )
        return MockTextToImageService()


def create_image_to_3d_service() -> ImageTo3DService:
    """Create image-to-3D service based on available dependencies."""
    if _has_cuda() and _has_triposr():
        from app.services.image_to_3d import TripoSRImageTo3DService
        logger.info("Using TripoSR image-to-3D service (GPU)")
        return TripoSRImageTo3DService()
    else:
        from app.services.image_to_3d import MockImageTo3DService
        logger.warning(
            "CUDA or TripoSR not available. Using mock image-to-3D service. "
            "Install TripoSR: https://github.com/VAST-AI-Research/TripoSR"
        )
        return MockImageTo3DService()


def create_texturing_service() -> TexturingService:
    """Create texturing service."""
    if settings.TEXTURING_ENABLED:
        try:
            import trimesh  # noqa: F401
            from app.services.texturing import BasicTexturingService
            logger.info("Using basic texturing service")
            return BasicTexturingService()
        except ImportError:
            logger.warning("trimesh not available, using passthrough texturing")
            from app.services.texturing import PassthroughTexturingService
            return PassthroughTexturingService()
    else:
        from app.services.texturing import PassthroughTexturingService
        logger.info("Texturing disabled, using passthrough")
        return PassthroughTexturingService()


def create_export_service() -> ExportService:
    """Create export service."""
    from app.services.export import TrimeshExportService
    return TrimeshExportService()


def create_storage_service() -> AssetStorageService:
    """Create asset storage service."""
    from app.services.storage import LocalAssetStorageService
    return LocalAssetStorageService()


# --- GPU info ---

def get_gpu_info() -> dict:
    """Get GPU availability and name."""
    if _has_cuda():
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "device_count": torch.cuda.device_count(),
        }
    return {"available": False, "name": None}
