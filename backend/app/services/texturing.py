"""
Basic texturing service.

This applies the reference image as a simple projection texture
onto the 3D model. For MVP, this is a basic UV-mapped texture.

For production, consider:
- TEXTure (https://github.com/TEXTurePaper/TEXTurePaper)
- Text2Tex
- Paint3D
"""

from pathlib import Path

import trimesh
import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.services.base import TexturingService

logger = get_logger(__name__)


class BasicTexturingService(TexturingService):
    """
    Applies the reference image as a basic texture to the mesh.
    Uses trimesh for UV mapping and texture application.
    """

    def apply_texture(
        self,
        mesh_path: Path,
        reference_image: Image.Image,
        output_path: Path,
    ) -> Path:
        logger.info(f"Applying basic texture to {mesh_path}")

        try:
            mesh = trimesh.load(str(mesh_path))

            if isinstance(mesh, trimesh.Scene):
                # Get the first geometry from the scene
                geometries = list(mesh.geometry.values())
                if not geometries:
                    logger.warning("No geometry found in scene, skipping texturing")
                    return mesh_path
                mesh = geometries[0]

            if not isinstance(mesh, trimesh.Trimesh):
                logger.warning(f"Unexpected mesh type: {type(mesh)}, skipping texturing")
                return mesh_path

            # Ensure UV coordinates exist
            if mesh.visual.uv is None or len(mesh.visual.uv) == 0:
                logger.info("No UV coords found, generating box projection UVs")
                mesh = self._generate_box_uvs(mesh)

            # Resize reference image to texture resolution
            from app.core.config import get_settings
            settings = get_settings()
            tex_size = settings.TEXTURE_RESOLUTION
            texture = reference_image.convert("RGB").resize((tex_size, tex_size))

            # Apply texture
            material = trimesh.visual.texture.SimpleMaterial(image=texture)
            color_visuals = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv,
                material=material,
                image=texture,
            )
            mesh.visual = color_visuals

            # Export
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(output_path))

            logger.info(f"Textured model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Texturing failed: {e}, returning original mesh")
            return mesh_path

    def _generate_box_uvs(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Generate simple box-projection UV coordinates."""
        vertices = mesh.vertices
        normals = mesh.vertex_normals

        uvs = np.zeros((len(vertices), 2))

        for i, (v, n) in enumerate(zip(vertices, normals)):
            abs_n = np.abs(n)
            if abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
                uvs[i] = [v[1], v[2]]
            elif abs_n[1] >= abs_n[0] and abs_n[1] >= abs_n[2]:
                uvs[i] = [v[0], v[2]]
            else:
                uvs[i] = [v[0], v[1]]

        # Normalize to [0, 1]
        if len(uvs) > 0:
            uv_min = uvs.min(axis=0)
            uv_max = uvs.max(axis=0)
            uv_range = uv_max - uv_min
            uv_range[uv_range == 0] = 1.0
            uvs = (uvs - uv_min) / uv_range

        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        return mesh


class PassthroughTexturingService(TexturingService):
    """No-op texturing that just copies the mesh as-is."""

    def apply_texture(
        self,
        mesh_path: Path,
        reference_image: Image.Image,
        output_path: Path,
    ) -> Path:
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(mesh_path), str(output_path))
        return output_path
