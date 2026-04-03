"""
Scene generation service: creates 3D environments/scenarios.

Pipeline:
1. Generate a panoramic environment image (SDXL with landscape prompt)
2. Create a ground plane/terrain mesh
3. Generate environment objects as separate meshes
4. Compose everything into a single GLB scene

For production, consider:
- Blockade Labs Skybox AI for panoramic environments
- LumaAI Scene generation
- ComfyUI + panoramic depth estimation
"""

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import SceneGenerationService, TextToImageService, ImageTo3DService

logger = get_logger(__name__)
settings = get_settings()


class CompositeSceneService(SceneGenerationService):
    """
    Scene generation by composing:
    1. A textured ground plane from a generated top-down image
    2. A skybox/backdrop from a panoramic image
    3. Optional: foreground objects from the text prompt
    
    Uses existing text-to-image and image-to-3D services.
    """

    def __init__(
        self,
        text_to_image: TextToImageService,
        image_to_3d: ImageTo3DService,
    ) -> None:
        self.text_to_image = text_to_image
        self.image_to_3d = image_to_3d

    def load_model(self) -> None:
        # Models are loaded via the injected services
        logger.info("Composite scene service ready")

    def generate(
        self,
        prompt: str,
        output_dir: Path,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating scene: '{prompt[:60]}'")

        scene = trimesh.Scene()

        # Step 1: Generate ground/terrain texture
        logger.info("Generating ground texture...")
        ground_prompt = (
            f"top-down aerial view of {prompt}, terrain texture, "
            "flat lighting, seamless, game environment texture"
        )
        ground_image = self.text_to_image.generate(
            prompt=ground_prompt,
            negative_prompt=negative_prompt or "text, watermark, UI, person, character",
            width=1024,
            height=1024,
            num_steps=20,
            guidance_scale=7.0,
            seed=seed,
        )
        ground_image.save(str(output_dir / "ground_texture.png"))

        # Create ground plane with texture
        ground_mesh = self._create_ground_plane(ground_image, size=10.0)
        scene.add_geometry(ground_mesh, node_name="ground")

        # Step 2: Generate backdrop/sky image
        logger.info("Generating backdrop...")
        backdrop_prompt = (
            f"panoramic landscape background of {prompt}, "
            "wide angle, environment art, no characters, cinematic"
        )
        backdrop_image = self.text_to_image.generate(
            prompt=backdrop_prompt,
            negative_prompt=negative_prompt or "text, watermark, UI, person",
            width=1024,
            height=512,
            num_steps=20,
            guidance_scale=7.0,
            seed=(seed + 1) if seed else None,
        )
        backdrop_image.save(str(output_dir / "backdrop.png"))

        # Create curved backdrop
        backdrop_mesh = self._create_backdrop(backdrop_image, radius=12.0, height=8.0)
        scene.add_geometry(backdrop_mesh, node_name="backdrop")

        # Step 3: Generate a central element/focal point
        logger.info("Generating focal element...")
        element_prompt = (
            f"single object for {prompt}, centered, white background, "
            "product photography, studio lighting"
        )
        element_image = self.text_to_image.generate(
            prompt=element_prompt,
            negative_prompt="multiple objects, text, blurry",
            width=1024,
            height=1024,
            num_steps=20,
            guidance_scale=7.5,
            seed=(seed + 2) if seed else None,
        )
        element_image.save(str(output_dir / "element.png"))

        # Generate 3D element
        try:
            element_dir = output_dir / "element_3d"
            element_path = self.image_to_3d.generate(element_image, element_dir)

            element_mesh = trimesh.load(str(element_path))
            if isinstance(element_mesh, trimesh.Scene):
                for name, geom in element_mesh.geometry.items():
                    # Place element on the ground
                    if isinstance(geom, trimesh.Trimesh):
                        bounds = geom.bounds
                        geom.apply_translation([0, -bounds[0][1], 0])
                    scene.add_geometry(geom, node_name=f"element_{name}")
            elif isinstance(element_mesh, trimesh.Trimesh):
                bounds = element_mesh.bounds
                element_mesh.apply_translation([0, -bounds[0][1], 0])
                scene.add_geometry(element_mesh, node_name="element")
        except Exception as e:
            logger.warning(f"Element 3D generation failed: {e}, scene will have ground + backdrop only")

        # Export scene as GLB
        output_path = output_dir / "scene.glb"
        glb_data = scene.export(file_type="glb")
        output_path.write_bytes(glb_data)

        logger.info(f"Scene exported: {output_path} ({len(glb_data)} bytes)")
        return output_path

    def _create_ground_plane(self, texture: Image.Image, size: float = 10.0) -> trimesh.Trimesh:
        """Create a textured ground plane."""
        # Create a subdivided plane for better visual quality
        segments = 20
        vertices = []
        faces = []
        uvs = []

        for i in range(segments + 1):
            for j in range(segments + 1):
                x = (i / segments - 0.5) * size
                z = (j / segments - 0.5) * size
                vertices.append([x, 0, z])
                uvs.append([i / segments, j / segments])

        for i in range(segments):
            for j in range(segments):
                v0 = i * (segments + 1) + j
                v1 = v0 + 1
                v2 = v0 + (segments + 1)
                v3 = v2 + 1
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])

        mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
        )

        # Apply texture
        material = trimesh.visual.texture.SimpleMaterial(image=texture)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.array(uvs),
            material=material,
            image=texture,
        )

        return mesh

    def _create_backdrop(
        self, texture: Image.Image, radius: float = 12.0, height: float = 8.0
    ) -> trimesh.Trimesh:
        """Create a curved backdrop cylinder segment with the panoramic image."""
        segments_h = 10
        segments_arc = 30
        arc_angle = np.pi * 1.2  # 216 degrees of coverage

        vertices = []
        faces = []
        uvs = []

        for i in range(segments_h + 1):
            for j in range(segments_arc + 1):
                angle = -arc_angle / 2 + (j / segments_arc) * arc_angle
                x = radius * np.sin(angle)
                y = (i / segments_h) * height
                z = -radius * np.cos(angle)
                vertices.append([x, y, z])
                uvs.append([j / segments_arc, i / segments_h])

        for i in range(segments_h):
            for j in range(segments_arc):
                v0 = i * (segments_arc + 1) + j
                v1 = v0 + 1
                v2 = v0 + (segments_arc + 1)
                v3 = v2 + 1
                # Reversed winding for inside-facing
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

        mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
        )

        material = trimesh.visual.texture.SimpleMaterial(image=texture)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.array(uvs),
            material=material,
            image=texture,
        )

        return mesh

    def unload_model(self) -> None:
        pass
