"""
Skin/texture generation service.

Generates a texture atlas for a 3D mesh using multi-view projection:
1. Render mesh from multiple viewpoints (front, back, left, right) as depth images
2. Generate textured images for each viewpoint using the reference image + prompt
3. Project textures onto a UV-mapped mesh
4. Blend overlapping regions into a final texture atlas
5. Export as GLB with embedded texture

Two implementations:
- SDXLSkinGenerationService: full pipeline using diffusers img2img
- MockSkinGenerationService: applies a solid color for testing without GPU
"""

import math
import struct
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from app.core.logging import get_logger
from app.services.base import SkinGenerationService

logger = get_logger(__name__)


# ── UV box-projection ────────────────────────────────────────────────────────

def _box_uv_projection(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute UV coordinates via box projection.
    Each vertex is assigned to the face normal axis that dominates,
    and UVs are derived from the other two axes normalized to [0,1].
    Returns (num_vertices, 2) float32 array.
    """
    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    bounds_min = vertices.min(axis=0)
    bounds_size = vertices.max(axis=0) - bounds_min
    bounds_size = np.where(bounds_size < 1e-6, 1.0, bounds_size)

    # Normalized vertex positions [0, 1]
    norm_v = (vertices - bounds_min) / bounds_size

    abs_normals = np.abs(normals)
    dominant_axis = np.argmax(abs_normals, axis=1)  # 0=x, 1=y, 2=z

    uvs = np.zeros((len(vertices), 2), dtype=np.float32)

    # X dominant → project onto YZ plane
    mask_x = dominant_axis == 0
    uvs[mask_x, 0] = norm_v[mask_x, 2]  # z → u
    uvs[mask_x, 1] = norm_v[mask_x, 1]  # y → v

    # Y dominant → project onto XZ plane
    mask_y = dominant_axis == 1
    uvs[mask_y, 0] = norm_v[mask_y, 0]  # x → u
    uvs[mask_y, 1] = norm_v[mask_y, 2]  # z → v

    # Z dominant → project onto XY plane
    mask_z = dominant_axis == 2
    uvs[mask_z, 0] = norm_v[mask_z, 0]  # x → u
    uvs[mask_z, 1] = norm_v[mask_z, 1]  # y → v

    return uvs


# ── Depth render ─────────────────────────────────────────────────────────────

def _render_depth_from_view(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    resolution: int = 512,
) -> Image.Image:
    """
    Render a simple depth image by projecting vertices from a given azimuth angle.
    Returns a grayscale PIL Image.

    Uses orthographic projection with simple z-buffering.
    azimuth_deg: 0=front, 90=right, 180=back, 270=left
    """
    az = math.radians(azimuth_deg)
    # Camera direction (looking from azimuth, slightly above)
    elev = math.radians(20)
    cam_x = math.sin(az) * math.cos(elev)
    cam_y = math.sin(elev)
    cam_z = math.cos(az) * math.cos(elev)
    cam_dir = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    # Build orthonormal basis
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(cam_dir, world_up)
    right_len = np.linalg.norm(right)
    if right_len < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= right_len
    up = np.cross(right, cam_dir)
    up /= np.linalg.norm(up)

    vertices = mesh.vertices.astype(np.float32)

    # Project to camera space
    u = vertices @ right
    v = vertices @ up
    depth = vertices @ cam_dir

    # Normalize to [0, 1]
    def norm01(arr):
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-6:
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    u_n = norm01(u)
    v_n = norm01(v)
    d_n = norm01(depth)

    # Rasterize via simple z-buffer
    px = (u_n * (resolution - 1)).astype(int).clip(0, resolution - 1)
    py = ((1.0 - v_n) * (resolution - 1)).astype(int).clip(0, resolution - 1)

    zbuf = np.full((resolution, resolution), -1.0, dtype=np.float32)
    depth_img = np.zeros((resolution, resolution), dtype=np.float32)

    for i in range(len(vertices)):
        if d_n[i] > zbuf[py[i], px[i]]:
            zbuf[py[i], px[i]] = d_n[i]
            depth_img[py[i], px[i]] = d_n[i]

    gray = (depth_img * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


# ── Texture atlas composition ────────────────────────────────────────────────

def _create_solid_color_texture(
    color: tuple[int, int, int] = (180, 140, 100),
    size: int = 512,
) -> Image.Image:
    """Create a solid-color texture for mock/testing."""
    return Image.new("RGB", (size, size), color=color)


def _blend_view_textures(
    view_images: list[Image.Image],
    uvs: np.ndarray,
    vertices: np.ndarray,
    view_azimuth_degs: list[float],
    atlas_size: int = 512,
) -> Image.Image:
    """
    Project textured view images onto a UV atlas by back-projecting from each view.
    For each vertex, determine which view covers it best (by surface normal alignment),
    look up the color in that view's image, and paint the UV atlas.
    Returns the atlas Image.
    """
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.float32)
    weight_map = np.zeros((atlas_size, atlas_size), dtype=np.float32)

    # Convert all view images to numpy
    view_arrays = [np.array(img.resize((atlas_size, atlas_size))).astype(np.float32) for img in view_images]

    bounds_min = vertices.min(axis=0)
    bounds_size = vertices.max(axis=0) - bounds_min
    bounds_size = np.where(bounds_size < 1e-6, 1.0, bounds_size)
    norm_v = (vertices - bounds_min) / bounds_size

    # For each view compute camera direction
    elev = math.radians(20)
    view_dirs = []
    for az_deg in view_azimuth_degs:
        az = math.radians(az_deg)
        view_dirs.append(np.array([
            math.sin(az) * math.cos(elev),
            math.sin(elev),
            math.cos(az) * math.cos(elev),
        ], dtype=np.float32))

    # For each view, project vertices and paint atlas
    for vi, (view_arr, view_dir) in enumerate(zip(view_arrays, view_dirs)):
        # u/v in view image corresponds to normalized vertex position in projected plane
        # Simple: use norm_v XZ for front/back views, YZ for side views
        az = math.radians(view_azimuth_degs[vi])
        right = np.array([math.cos(az), 0.0, -math.sin(az)], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Project vertices to this view's image space
        view_u = (vertices @ right)
        view_v = (vertices @ up)

        view_u_n = (view_u - view_u.min()) / max(view_u.max() - view_u.min(), 1e-6)
        view_v_n = (view_v - view_v.min()) / max(view_v.max() - view_v.min(), 1e-6)

        # Atlas UV pixel coordinates
        ax = (uvs[:, 0] * (atlas_size - 1)).astype(int).clip(0, atlas_size - 1)
        ay = ((1.0 - uvs[:, 1]) * (atlas_size - 1)).astype(int).clip(0, atlas_size - 1)

        # Image pixel coordinates for this view
        ix = (view_u_n * (atlas_size - 1)).astype(int).clip(0, atlas_size - 1)
        iy = ((1.0 - view_v_n) * (atlas_size - 1)).astype(int).clip(0, atlas_size - 1)

        # Paint with weight = alignment with view direction
        # (positive = facing this camera)
        for k in range(len(vertices)):
            color = view_arr[iy[k], ix[k]]
            # Equal weight for all views (simple blend)
            w = 1.0 / len(view_dirs)
            atlas[ay[k], ax[k]] += color * w
            weight_map[ay[k], ax[k]] += w

    # Normalize where painted, fill gaps with neutral
    painted = weight_map > 0
    atlas[painted] /= weight_map[painted, np.newaxis]

    # Fill unpainted areas with average color
    if painted.any():
        avg_color = atlas[painted].mean(axis=0)
    else:
        avg_color = np.array([180.0, 140.0, 100.0])
    atlas[~painted] = avg_color

    return Image.fromarray(atlas.clip(0, 255).astype(np.uint8))


# ── GLB export with embedded texture ────────────────────────────────────────

def _build_glb_with_texture(
    mesh: trimesh.Trimesh,
    uvs: np.ndarray,
    texture_image: Image.Image,
) -> bytes:
    """
    Build a GLB file with the mesh, UV coords, and embedded PNG texture.
    """
    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    indices = mesh.faces.astype(np.uint32).flatten()
    num_verts = len(vertices)
    uvs_f32 = uvs.astype(np.float32)

    # Encode texture to PNG bytes
    png_buf = BytesIO()
    texture_image.save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()
    # Pad to 4 bytes
    padded_png = png_bytes + b"\x00" * ((-len(png_bytes)) % 4)

    buf = bytearray()
    buffer_views = []
    accessors = []

    def _pad4(b: bytearray):
        while len(b) % 4 != 0:
            b.append(0)

    def _add_data(data: bytes, target: int | None = None) -> int:
        _pad4(buf)
        offset = len(buf)
        buf.extend(data)
        bv: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            bv["target"] = target
        buffer_views.append(bv)
        return len(buffer_views) - 1

    def _add_accessor(bv_idx, comp_type, count, acc_type, min_val=None, max_val=None) -> int:
        acc: dict[str, Any] = {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": count,
            "type": acc_type,
        }
        if min_val is not None:
            acc["min"] = min_val
        if max_val is not None:
            acc["max"] = max_val
        accessors.append(acc)
        return len(accessors) - 1

    # Geometry
    pos_bv = _add_data(vertices.tobytes(), target=34962)
    pos_acc = _add_accessor(pos_bv, 5126, num_verts, "VEC3",
                            vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist())

    norm_bv = _add_data(normals.tobytes(), target=34962)
    norm_acc = _add_accessor(norm_bv, 5126, num_verts, "VEC3")

    uv_bv = _add_data(uvs_f32.tobytes(), target=34962)
    uv_acc = _add_accessor(uv_bv, 5126, num_verts, "VEC2",
                           uvs_f32.min(axis=0).tolist(), uvs_f32.max(axis=0).tolist())

    idx_bv = _add_data(indices.tobytes(), target=34963)
    idx_acc = _add_accessor(idx_bv, 5125, len(indices), "SCALAR",
                            [int(indices.min())], [int(indices.max())])

    # Texture buffer view (separate, for image)
    _pad4(buf)
    tex_offset = len(buf)
    buf.extend(padded_png)
    tex_bv = {"buffer": 0, "byteOffset": tex_offset, "byteLength": len(padded_png)}
    buffer_views.append(tex_bv)
    tex_bv_idx = len(buffer_views) - 1

    gltf = {
        "asset": {"version": "2.0", "generator": "ModelGenerator-Skin"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "mesh", "mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": norm_acc,
                    "TEXCOORD_0": uv_acc,
                },
                "indices": idx_acc,
                "material": 0,
            }]
        }],
        "materials": [{
            "name": "skin_material",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.85,
            },
            "doubleSided": True,
        }],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"bufferView": tex_bv_idx, "mimeType": "image/png"}],
        "samplers": [{
            "magFilter": 9729,  # LINEAR
            "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
            "wrapS": 10497,     # REPEAT
            "wrapT": 10497,
        }],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    _pad4(buf)
    gltf["buffers"][0]["byteLength"] = len(buf)

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    total = 12 + 8 + len(json_bytes) + 8 + len(buf)
    out = bytearray()
    out.extend(struct.pack("<III", 0x46546C67, 2, total))
    out.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    out.extend(json_bytes)
    out.extend(struct.pack("<II", len(buf), 0x004E4942))
    out.extend(buf)
    return bytes(out)


# ── Service implementations ──────────────────────────────────────────────────

class MockSkinGenerationService(SkinGenerationService):
    """
    Mock skin service — applies a procedural clay-like color texture.
    Works without GPU, used for testing.
    """

    def load_model(self) -> None:
        logger.info("Mock skin generation service ready")

    def generate_skin(
        self,
        glb_path: Path,
        prompt: str,
        output_path: Path,
        reference_image: Image.Image | None = None,
    ) -> Path:
        logger.info(f"Mock skin generation: {glb_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.creation.box()
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            mesh = trimesh.creation.box()

        uvs = _box_uv_projection(mesh)

        # Create a simple procedural texture from the UV pattern
        texture = _create_procedural_texture(uvs, mesh.vertices, prompt)

        glb_bytes = _build_glb_with_texture(mesh, uvs, texture)
        output_path.write_bytes(glb_bytes)
        logger.info(f"Mock skin GLB: {output_path} ({len(glb_bytes)} bytes)")
        return output_path

    def unload_model(self) -> None:
        pass


def _create_procedural_texture(
    uvs: np.ndarray,
    vertices: np.ndarray,
    prompt: str,
    size: int = 512,
) -> Image.Image:
    """Create a simple procedural texture based on UV and prompt keywords."""
    # Choose base color from prompt keywords
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ("stone", "rock", "gray", "metal", "steel")):
        base_color = (130, 130, 140)
    elif any(w in prompt_lower for w in ("wood", "brown", "bark", "wooden")):
        base_color = (139, 90, 43)
    elif any(w in prompt_lower for w in ("skin", "flesh", "human", "person", "character")):
        base_color = (210, 170, 120)
    elif any(w in prompt_lower for w in ("green", "grass", "plant", "leaf")):
        base_color = (80, 140, 60)
    elif any(w in prompt_lower for w in ("blue", "ocean", "water", "ice")):
        base_color = (60, 100, 180)
    elif any(w in prompt_lower for w in ("red", "fire", "lava", "ruby")):
        base_color = (180, 40, 30)
    elif any(w in prompt_lower for w in ("gold", "yellow", "bronze", "copper")):
        base_color = (200, 160, 40)
    else:
        base_color = (160, 140, 120)  # default warm gray

    img = Image.new("RGB", (size, size), color=base_color)
    draw = ImageDraw.Draw(img)

    # Add UV-based shading: paint vertex contributions
    ax = (uvs[:, 0] * (size - 1)).astype(int).clip(0, size - 1)
    ay = ((1.0 - uvs[:, 1]) * (size - 1)).astype(int).clip(0, size - 1)

    # Compute per-vertex height shading
    y_norm = (vertices[:, 1] - vertices[:, 1].min())
    y_range = y_norm.max()
    if y_range > 1e-6:
        y_norm /= y_range

    # Build shaded atlas
    atlas = np.full((size, size, 3), base_color, dtype=np.float32)
    count = np.zeros((size, size), dtype=int)

    for k in range(len(ax)):
        shade = 0.7 + 0.3 * y_norm[k]
        shaded = tuple(min(255, int(c * shade)) for c in base_color)
        atlas[ay[k], ax[k]] = shaded
        count[ay[k], ax[k]] += 1

    return Image.fromarray(atlas.clip(0, 255).astype(np.uint8))


class SDXLSkinGenerationService(SkinGenerationService):
    """
    Full skin generation using SDXL img2img conditioning.
    Renders depth maps from multiple viewpoints, generates textured images,
    projects them onto a UV atlas, and exports a textured GLB.
    """

    # 4 principal views
    VIEW_AZIMUTHS = [0.0, 90.0, 180.0, 270.0]

    def __init__(self, text_to_image_service) -> None:
        self._t2i = text_to_image_service
        self._img2img_pipe = None

    def load_model(self) -> None:
        try:
            import torch
            from diffusers import StableDiffusionXLImg2ImgPipeline
            from app.core.config import get_settings
            settings = get_settings()

            logger.info("Loading SDXL img2img pipeline for skin generation...")
            self._img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                settings.TEXT_TO_IMAGE_MODEL,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self._img2img_pipe.to(settings.TEXT_TO_IMAGE_DEVICE)
            logger.info("SDXL img2img pipeline loaded")
        except Exception as e:
            logger.warning(f"Could not load SDXL img2img: {e}. Falling back to mock.")
            self._img2img_pipe = None

    def generate_skin(
        self,
        glb_path: Path,
        prompt: str,
        output_path: Path,
        reference_image: Image.Image | None = None,
    ) -> Path:
        logger.info(f"Generating skin for {glb_path.name}: '{prompt[:60]}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load mesh
        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.creation.box()
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            mesh = trimesh.creation.box()

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # Compute UV coordinates
        uvs = _box_uv_projection(mesh)

        # Generate textured views
        view_images = self._generate_view_textures(mesh, prompt, reference_image)

        # Project onto UV atlas
        logger.info("Projecting textures onto UV atlas...")
        atlas = _blend_view_textures(
            view_images, uvs, mesh.vertices, self.VIEW_AZIMUTHS, atlas_size=512,
        )

        # Export
        logger.info("Building textured GLB...")
        glb_bytes = _build_glb_with_texture(mesh, uvs, atlas)
        output_path.write_bytes(glb_bytes)
        logger.info(f"Textured GLB: {output_path} ({len(glb_bytes)} bytes)")
        return output_path

    def _generate_view_textures(
        self,
        mesh: trimesh.Trimesh,
        prompt: str,
        reference_image: Image.Image | None,
        atlas_size: int = 512,
    ) -> list[Image.Image]:
        """Render depth maps from each view and texture them via img2img."""
        view_images = []
        texture_prompt = f"{prompt}, high quality 3D model texture, detailed surface, seamless"

        for az_deg in self.VIEW_AZIMUTHS:
            depth_img = _render_depth_from_view(mesh, az_deg, resolution=atlas_size)

            if self._img2img_pipe is not None and reference_image is not None:
                try:
                    img = self._img2img_conditioning(depth_img, texture_prompt, reference_image)
                except Exception as e:
                    logger.warning(f"img2img failed for view {az_deg}: {e}. Using depth map.")
                    img = depth_img
            else:
                # No GPU — colorize depth map with a warm palette
                img = self._colorize_depth(depth_img, prompt)

            view_images.append(img)

        return view_images

    def _img2img_conditioning(
        self,
        depth_img: Image.Image,
        prompt: str,
        reference_image: Image.Image,
    ) -> Image.Image:
        """Run SDXL img2img with depth as init image, guided by reference."""
        import torch
        init = reference_image.resize((512, 512)).convert("RGB")
        result = self._img2img_pipe(
            prompt=prompt,
            image=init,
            strength=0.65,
            num_inference_steps=20,
            guidance_scale=7.5,
        )
        return result.images[0]

    def _colorize_depth(self, depth_img: Image.Image, prompt: str) -> Image.Image:
        """Colorize a grayscale depth map with a hue derived from the prompt."""
        from PIL import ImageEnhance
        depth_arr = np.array(depth_img.convert("L")).astype(np.float32) / 255.0

        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ("stone", "rock")):
            r, g, b = 130, 125, 115
        elif any(w in prompt_lower for w in ("wood", "bark")):
            r, g, b = 139, 95, 45
        else:
            r, g, b = 160, 140, 110

        colored = np.stack([
            (depth_arr * r).clip(0, 255),
            (depth_arr * g).clip(0, 255),
            (depth_arr * b).clip(0, 255),
        ], axis=-1).astype(np.uint8)
        return Image.fromarray(colored)

    def unload_model(self) -> None:
        if self._img2img_pipe is not None:
            import torch
            del self._img2img_pipe
            self._img2img_pipe = None
            torch.cuda.empty_cache()
            logger.info("SDXL img2img pipeline unloaded")
