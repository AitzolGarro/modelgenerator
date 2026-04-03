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
    AnimationService,
    MeshRefinementService,
    SceneGenerationService,
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
    if _has_cuda() and _has_diffusers():
        from app.services.text_to_image import SDXLTextToImageService
        logger.info("Using SDXL text-to-image service (GPU)")
        return SDXLTextToImageService()
    else:
        from app.services.text_to_image import MockTextToImageService
        logger.warning("Using mock text-to-image service")
        return MockTextToImageService()


def create_image_to_3d_service() -> ImageTo3DService:
    if _has_cuda() and _has_triposr():
        from app.services.image_to_3d import TripoSRImageTo3DService
        logger.info("Using TripoSR image-to-3D service (GPU)")
        return TripoSRImageTo3DService()
    else:
        from app.services.image_to_3d import MockImageTo3DService
        logger.warning("Using mock image-to-3D service")
        return MockImageTo3DService()


def create_texturing_service() -> TexturingService:
    if settings.TEXTURING_ENABLED:
        try:
            import trimesh  # noqa: F401
            from app.services.texturing import BasicTexturingService
            logger.info("Using basic texturing service")
            return BasicTexturingService()
        except ImportError:
            from app.services.texturing import PassthroughTexturingService
            return PassthroughTexturingService()
    else:
        from app.services.texturing import PassthroughTexturingService
        return PassthroughTexturingService()


def create_export_service() -> ExportService:
    from app.services.export import TrimeshExportService
    return TrimeshExportService()


def create_storage_service() -> AssetStorageService:
    from app.services.storage import LocalAssetStorageService
    return LocalAssetStorageService()


def create_animation_service() -> AnimationService:
    from app.services.animation import ProceduralAnimationService
    logger.info("Using procedural animation service")
    return ProceduralAnimationService()


def create_refinement_service() -> MeshRefinementService:
    from app.services.refinement import TrimeshRefinementService
    logger.info("Using trimesh refinement service")
    return TrimeshRefinementService()


def create_scene_service(
    text_to_image: TextToImageService,
    image_to_3d: ImageTo3DService,
) -> SceneGenerationService:
    from app.services.scene import CompositeSceneService
    logger.info("Using composite scene service")
    return CompositeSceneService(text_to_image, image_to_3d)


# --- GPU info ---

def get_gpu_info() -> dict:
    if _has_cuda():
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "device_count": torch.cuda.device_count(),
        }
    return {"available": False, "name": None}
