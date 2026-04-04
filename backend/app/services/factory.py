"""
Service factory: creates the right implementation based on config and availability.
"""

from pathlib import Path

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
    SkinGenerationService,
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


def _has_instantmesh() -> bool:
    """Return True if the InstantMesh repo is present (cloned in Docker)."""
    try:
        from app.services.image_to_3d_instantmesh import _instantmesh_available
        return _instantmesh_available()
    except Exception:
        return False


def _has_diffusers() -> bool:
    try:
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pyrender() -> bool:
    try:
        import pyrender  # noqa: F401
        return True
    except Exception:
        return False


def _has_xatlas() -> bool:
    try:
        import xatlas  # noqa: F401
        return True
    except Exception:
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
    backend = settings.IMAGE_TO_3D_BACKEND  # "auto", "instantmesh", "triposr"

    # --- InstantMesh path ---
    if backend in ("instantmesh", "auto"):
        if _has_instantmesh():
            try:
                from app.services.image_to_3d_instantmesh import InstantMeshImageTo3DService
                svc = InstantMeshImageTo3DService()
                svc.load_model()  # validates repo presence; raises if missing
                logger.info("Using InstantMesh image-to-3D service (subprocess)")
                return svc
            except Exception as exc:
                if backend == "instantmesh":
                    # User explicitly requested InstantMesh — raise rather than silently fall back
                    raise RuntimeError(
                        f"IMAGE_TO_3D_BACKEND=instantmesh but InstantMesh failed to load: {exc}"
                    ) from exc
                logger.warning(
                    f"InstantMesh unavailable ({exc}), falling back to TripoSR"
                )
        else:
            if backend == "instantmesh":
                raise RuntimeError(
                    "IMAGE_TO_3D_BACKEND=instantmesh but InstantMesh repo not found. "
                    "Make sure the Docker image was built with InstantMesh cloned."
                )
            logger.info("InstantMesh repo not found — using TripoSR")

    # --- TripoSR path (also fallback from auto/instantmesh) ---
    if _has_cuda() and _has_triposr():
        from app.services.image_to_3d import TripoSRImageTo3DService
        logger.info("Using TripoSR image-to-3D service (GPU)")
        return TripoSRImageTo3DService()

    # --- Final fallback: mock ---
    from app.services.image_to_3d import MockImageTo3DService
    logger.warning("Using mock image-to-3D service (no GPU/TripoSR/InstantMesh available)")
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
    mocap_dir = Path(__file__).parent / "mocap_data"
    if mocap_dir.is_dir() and any(mocap_dir.glob("*.bvh")):
        try:
            from app.services.animation_mocap import MocapAnimationService
            logger.info("Using mocap animation service (BVH data found)")
            return MocapAnimationService()
        except Exception as exc:
            logger.warning(f"Failed to load MocapAnimationService ({exc}) — falling back to procedural")
    else:
        logger.warning("mocap_data/ not found or empty — using procedural animation service")
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


def create_skin_service(
    text_to_image: TextToImageService | None = None,
) -> SkinGenerationService:
    # Best path: GPU + pyrender EGL → full UV texturing pipeline
    if _has_cuda() and _has_pyrender() and _has_diffusers() and text_to_image is not None:
        from app.services.uv_texturing import UVTexturingService
        logger.info("Using UV texturing service (GPU + pyrender + ControlNet pipeline)")
        return UVTexturingService(text_to_image)

    # Good path: GPU + diffusers but no pyrender → SDXL img2img with box UV
    if _has_cuda() and _has_diffusers() and text_to_image is not None:
        from app.services.skin_generator import SDXLSkinGenerationService
        logger.info("Using SDXL skin generation service (GPU, no pyrender)")
        return SDXLSkinGenerationService(text_to_image)

    # Fallback: no GPU or no diffusers → mock
    from app.services.skin_generator import MockSkinGenerationService
    logger.warning("Using mock skin generation service (no GPU/pyrender)")
    return MockSkinGenerationService()


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
