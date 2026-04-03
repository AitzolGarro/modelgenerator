"""
Image-to-3D service using TripoSR.

Uses a local copy of TripoSR source (app/services/tsr_local/) with
scikit-image marching cubes replacing torchmcubes (which doesn't build on Python 3.14).

Model weights: stabilityai/TripoSR on HuggingFace (~1GB, downloaded on first run).
"""

from pathlib import Path

import torch
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import ImageTo3DService

logger = get_logger(__name__)
settings = get_settings()


class TripoSRImageTo3DService(ImageTo3DService):
    """
    TripoSR-based image to 3D model conversion.
    Uses the local tsr_local package (patched for Python 3.14 compat).
    """

    def __init__(self) -> None:
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            logger.info("TripoSR model already loaded")
            return

        from app.services.tsr_local.system import TSR

        logger.info(f"Loading TripoSR model: {settings.TRIPOSR_MODEL}")

        self._model = TSR.from_pretrained(
            settings.TRIPOSR_MODEL,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model.renderer.set_chunk_size(settings.TRIPOSR_CHUNK_SIZE)
        self._model.to(settings.TRIPOSR_DEVICE)

        logger.info("TripoSR model loaded successfully")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for TripoSR: remove background, resize foreground."""
        # Remove background
        try:
            import rembg
            logger.info("Removing background with rembg")
            image = rembg.remove(image)
        except ImportError:
            logger.warning("rembg not installed, using image as-is")

        # Ensure RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Resize foreground to fill 85% of the frame (TripoSR convention)
        image = self._resize_foreground(image, ratio=0.85)

        return image

    def _resize_foreground(self, image: Image.Image, ratio: float = 0.85) -> Image.Image:
        """Resize foreground and center on gray background."""
        image_arr = np.array(image)

        # Find bounding box of non-transparent pixels
        if image_arr.shape[2] == 4:
            alpha = image_arr[:, :, 3]
            coords = np.argwhere(alpha > 0)
            if len(coords) == 0:
                return image

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1

            # Crop to foreground
            fg = image_arr[y_min:y_max, x_min:x_max]

            # Calculate new size
            h, w = fg.shape[:2]
            max_dim = max(h, w)
            new_size = int(max_dim / ratio)

            # Create centered image with gray background
            result = np.zeros((new_size, new_size, 4), dtype=np.uint8)
            # Gray background
            result[:, :, :3] = 127
            result[:, :, 3] = 255

            # Paste foreground centered
            y_offset = (new_size - h) // 2
            x_offset = (new_size - w) // 2

            # Blend with alpha
            fg_float = fg.astype(np.float32) / 255.0
            alpha_fg = fg_float[:, :, 3:4]

            result_region = result[y_offset:y_offset+h, x_offset:x_offset+w].astype(np.float32) / 255.0
            blended = fg_float[:, :, :3] * alpha_fg + result_region[:, :, :3] * (1 - alpha_fg)

            result[y_offset:y_offset+h, x_offset:x_offset+w, :3] = (blended * 255).astype(np.uint8)
            result[y_offset:y_offset+h, x_offset:x_offset+w, 3] = 255

            return Image.fromarray(result)

        return image

    def generate(self, image: Image.Image, output_dir: Path) -> Path:
        if self._model is None:
            self.load_model()

        output_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess
        processed = self._preprocess_image(image)

        # Convert to format expected by TripoSR
        # TripoSR expects PIL Image as RGB with object on gray/white background
        if processed.mode == "RGBA":
            # Composite onto gray background
            arr = np.array(processed).astype(np.float32) / 255.0
            rgb = arr[:, :, :3] * arr[:, :, 3:4] + 0.5 * (1 - arr[:, :, 3:4])
            processed = Image.fromarray((rgb * 255).astype(np.uint8))

        logger.info("Running TripoSR inference...")

        with torch.no_grad():
            scene_codes = self._model([processed], device=settings.TRIPOSR_DEVICE)

        logger.info("Extracting mesh...")

        meshes = self._model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=settings.TRIPOSR_MC_RESOLUTION,
        )

        mesh = meshes[0]
        output_path = output_dir / "mesh.obj"
        mesh.export(str(output_path))

        logger.info(f"Mesh exported to {output_path}")
        return output_path

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("TripoSR model unloaded")


class MockImageTo3DService(ImageTo3DService):
    """Mock service that generates a simple cube OBJ for testing."""

    def load_model(self) -> None:
        logger.info("Mock image-to-3D service loaded")

    def generate(self, image: Image.Image, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "mesh.obj"

        logger.info(f"Mock generating 3D model at {output_path}")

        obj_content = """# Mock 3D model - cube
mtllib mesh.mtl
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
vn 0.0 0.0 1.0
vn 0.0 0.0 -1.0
vn 1.0 0.0 0.0
vn -1.0 0.0 0.0
vn 0.0 1.0 0.0
vn 0.0 -1.0 0.0
f 1//1 2//1 3//1 4//1
f 5//2 8//2 7//2 6//2
f 2//3 6//3 7//3 3//3
f 1//4 4//4 8//4 5//4
f 4//5 3//5 7//5 8//5
f 1//6 5//6 6//6 2//6
"""
        output_path.write_text(obj_content)
        return output_path

    def unload_model(self) -> None:
        logger.info("Mock image-to-3D service unloaded")
