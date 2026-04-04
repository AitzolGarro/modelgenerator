"""
UV Texturing Pipeline Service.

Full pipeline for generating photorealistic textures for 3D meshes:
1. UV unwrapping: xatlas (when available) or scipy LSCM fallback
2. Multi-view depth rendering: pyrender EGL (when available) or numpy z-buffer fallback
3. Depth-conditioned texture generation: ControlNet SDXL or regular img2img fallback
4. Atlas projection: face-normal-weighted, barycentric texel sampling
5. Seam blending: Gaussian blur on chart border texels
6. GLB export with PBR baseColorTexture

On Bazzite (no xatlas compile): runs scipy LSCM + numpy depth + img2img fallback.
In Docker (full deps): runs xatlas + pyrender EGL + ControlNet SDXL.
"""

import math
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import SkinGenerationService
from app.services.skin_generator import _build_glb_with_texture, _box_uv_projection

logger = get_logger(__name__)
settings = get_settings()


# ── Camera / view helpers ────────────────────────────────────────────────────

# 6 camera views: (azimuth_deg, elevation_deg)
_VIEW_CONFIGS = [
    (0.0, 0.0),     # front
    (90.0, 0.0),    # right
    (180.0, 0.0),   # back
    (270.0, 0.0),   # left
    (0.0, 45.0),    # top-front
    (180.0, 45.0),  # top-back
]


def _make_camera_pose(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """
    Build a 4×4 camera-to-world pose matrix for the given spherical coordinates.
    Camera looks toward the origin from position at (az, elev, dist).
    """
    az = math.radians(azimuth_deg)
    elev = math.radians(elevation_deg)

    # Camera position in world space
    cam_x = distance * math.sin(az) * math.cos(elev)
    cam_y = distance * math.sin(elev)
    cam_z = distance * math.cos(az) * math.cos(elev)
    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    # Build rotation: z-axis = -forward (OpenGL convention)
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right_len = np.linalg.norm(right)
    if right_len < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_len
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward  # OpenGL: cam looks down -Z
    pose[:3, 3] = cam_pos
    return pose


def _view_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from camera toward origin (world space)."""
    az = math.radians(azimuth_deg)
    elev = math.radians(elevation_deg)
    dx = math.sin(az) * math.cos(elev)
    dy = math.sin(elev)
    dz = math.cos(az) * math.cos(elev)
    v = np.array([-dx, -dy, -dz], dtype=np.float32)  # toward origin
    return v / (np.linalg.norm(v) + 1e-12)


# ── LSCM UV unwrapping (pure Python fallback) ────────────────────────────────

def _lscm_uv_projection(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Least Squares Conformal Maps (LSCM) UV unwrapping — pure scipy implementation.
    Finds mesh boundary, fixes 2 boundary vertices as UV anchors,
    solves a sparse linear system to compute conformal UVs.
    Returns (num_vertices, 2) float32 array.
    """
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
    except ImportError:
        logger.warning("scipy not available, falling back to box UV projection")
        return _box_uv_projection(mesh)

    vertices = mesh.vertices.astype(np.float64)
    faces = mesh.faces.astype(np.int32)
    n_verts = len(vertices)
    n_faces = len(faces)

    # ── Find boundary edges ──────────────────────────────────────────────────
    edge_count: dict[tuple[int, int], int] = {}
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [(a, b) for (a, b), cnt in edge_count.items() if cnt == 1]

    if len(boundary_edges) < 2:
        # Closed mesh (no boundary) — use box projection as fallback
        logger.info("Closed mesh detected — using box UV instead of LSCM")
        return _box_uv_projection(mesh)

    # Build boundary vertex order (walk the boundary loop)
    adj: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Walk from first boundary vertex
    start = boundary_edges[0][0]
    boundary_loop = [start]
    visited = {start}
    current = start
    while True:
        neighbours = [v for v in adj.get(current, []) if v not in visited]
        if not neighbours:
            break
        nxt = neighbours[0]
        boundary_loop.append(nxt)
        visited.add(nxt)
        current = nxt

    if len(boundary_loop) < 2:
        logger.warning("Could not walk boundary — falling back to box UV")
        return _box_uv_projection(mesh)

    # Fix 2 opposite boundary vertices as anchors
    pinned_idx0 = boundary_loop[0]
    pinned_idx1 = boundary_loop[len(boundary_loop) // 2]
    pinned_uv0 = np.array([0.1, 0.5])
    pinned_uv1 = np.array([0.9, 0.5])

    # ── Build LSCM system ────────────────────────────────────────────────────
    # For each face, the conformal energy is minimised by solving:
    #   sum_f area_f * |d/du - J d/dv|^2 where J rotates by 90 degrees
    # We build a complex-valued (real+imag) sparse least-squares system.
    # u and v unknowns are interleaved: X[2i] = u_i, X[2i+1] = v_i

    free_verts = [i for i in range(n_verts) if i not in (pinned_idx0, pinned_idx1)]
    vert_to_free = {}
    for fi, vi in enumerate(free_verts):
        vert_to_free[vi] = fi
    n_free = len(free_verts)

    # 2 unknowns per free vertex (u, v)
    n_unknowns = 2 * n_free
    n_equations = 2 * n_faces

    A = lil_matrix((n_equations, n_unknowns), dtype=np.float64)
    b = np.zeros(n_equations, dtype=np.float64)

    for fi, face in enumerate(faces):
        v0, v1, v2 = face
        p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]

        # Local 2D coordinates using edge vectors
        e1 = p1 - p0
        e2 = p2 - p0
        e1_len = np.linalg.norm(e1)
        if e1_len < 1e-12:
            continue
        e1_norm = e1 / e1_len

        # Gram-Schmidt for e2 in local frame
        e2_u = np.dot(e2, e1_norm)
        e2_perp = e2 - e2_u * e1_norm
        e2_perp_len = np.linalg.norm(e2_perp)
        if e2_perp_len < 1e-12:
            continue

        # Local 2D positions
        lp0 = np.array([0.0, 0.0])
        lp1 = np.array([e1_len, 0.0])
        lp2 = np.array([e2_u, e2_perp_len])

        # LSCM weight coefficients (W matrix entries)
        # Derived from conformal condition df/dz conjugate = 0
        area2 = abs((lp1[0] - lp0[0]) * (lp2[1] - lp0[1]) -
                    (lp2[0] - lp0[0]) * (lp1[1] - lp0[1]))
        if area2 < 1e-12:
            continue

        inv_area = 1.0 / area2

        # For LSCM: the coefficients for vertices (v0, v1, v2)
        # Wu = inv_area * [(lp2-lp1), (lp0-lp2), (lp1-lp0)] in u
        # Wv = inv_area * [(lp2-lp1), (lp0-lp2), (lp1-lp0)] rotated by 90

        def _coeff(a, b_pt):
            """Complex LSCM coefficient for an edge a→b in local 2D."""
            return np.array([b_pt[0] - a[0], b_pt[1] - a[1]])

        c0 = _coeff(lp1, lp2) * inv_area  # coeff for vertex v0
        c1 = _coeff(lp2, lp0) * inv_area  # coeff for vertex v1
        c2 = _coeff(lp0, lp1) * inv_area  # coeff for vertex v2

        row_u = 2 * fi
        row_v = 2 * fi + 1

        for vert_idx, coeff in [(v0, c0), (v1, c1), (v2, c2)]:
            cu, cv = coeff[0], coeff[1]
            if vert_idx == pinned_idx0:
                b[row_u] -= cu * pinned_uv0[0] - cv * pinned_uv0[1]
                b[row_v] -= cv * pinned_uv0[0] + cu * pinned_uv0[1]
            elif vert_idx == pinned_idx1:
                b[row_u] -= cu * pinned_uv1[0] - cv * pinned_uv1[1]
                b[row_v] -= cv * pinned_uv1[0] + cu * pinned_uv1[1]
            else:
                col_u = 2 * vert_to_free[vert_idx]
                col_v = col_u + 1
                A[row_u, col_u] += cu
                A[row_u, col_v] -= cv
                A[row_v, col_u] += cv
                A[row_v, col_v] += cu

    try:
        A_csc = A.tocsc()
        # Solve least-squares via normal equations A^T A x = A^T b
        AtA = A_csc.T @ A_csc
        Atb = A_csc.T @ b
        x = spsolve(AtA, Atb)
    except Exception as exc:
        logger.warning(f"LSCM solver failed: {exc}. Falling back to box UV.")
        return _box_uv_projection(mesh)

    # ── Assemble UV array ────────────────────────────────────────────────────
    uvs = np.zeros((n_verts, 2), dtype=np.float32)
    for fi, vi in enumerate(free_verts):
        uvs[vi, 0] = float(x[2 * fi])
        uvs[vi, 1] = float(x[2 * fi + 1])
    uvs[pinned_idx0] = pinned_uv0.astype(np.float32)
    uvs[pinned_idx1] = pinned_uv1.astype(np.float32)

    # Normalise to [0, 1]
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range = np.where(uv_range < 1e-6, 1.0, uv_range)
    uvs = (uvs - uv_min) / uv_range

    return uvs


# ── xatlas UV unwrapping ─────────────────────────────────────────────────────

def _xatlas_uv_projection(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray, "np.ndarray | None"]:
    """
    Compute UV coordinates using xatlas chart-based parameterisation.
    Returns (remapped_mesh, uvs, chart_ids).
    xatlas remaps vertices (seam splits), so the returned mesh has a different
    vertex/face count from the input.
    """
    import xatlas  # type: ignore

    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)

    vmapping, new_indices, new_uvs = xatlas.parametrize(vertices, faces)

    # Remap vertices to match the new (split-seam) index buffer
    new_verts = vertices[vmapping]
    new_faces = new_indices.astype(np.int32)

    remapped_mesh = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        process=False,
    )

    uvs = new_uvs.astype(np.float32)

    # xatlas doesn't expose chart_ids directly in all bindings — try, else None
    chart_ids = None
    try:
        _, _, _, chart_ids = xatlas.parametrize(vertices, faces, return_chart_ids=True)  # type: ignore
    except Exception:
        pass

    return remapped_mesh, uvs, chart_ids


# ── Depth rendering (pyrender EGL) ───────────────────────────────────────────

def _render_depth_views_pyrender(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Render 6 depth views using pyrender with EGL headless backend.
    Returns list of (color_uint8_HxWx3, depth_float32_HxW, camera_pose_4x4).
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender  # type: ignore

    # Compute bounding sphere for camera distance
    center = mesh.bounding_sphere.primitive.center
    radius = mesh.bounding_sphere.primitive.radius
    distance = max(radius * 2.5, 0.01)

    # Build pyrender mesh (convert trimesh → pyrender)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    fov_y = math.radians(45.0)
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=1.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    results = []

    try:
        for az_deg, elev_deg in _VIEW_CONFIGS:
            scene = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
            scene.add(pr_mesh)

            pose = _make_camera_pose(az_deg, elev_deg, distance + np.linalg.norm(center))
            scene.add(camera, pose=pose)
            scene.add(light, pose=pose)

            color, depth = renderer.render(scene)
            # depth is float32 in metres — normalise to [0,1] for display / conditioning
            depth_norm = depth.copy()
            valid = depth_norm > 0
            if valid.any():
                d_min = depth_norm[valid].min()
                d_max = depth_norm[valid].max()
                if d_max > d_min:
                    depth_norm[valid] = (depth_norm[valid] - d_min) / (d_max - d_min)
                depth_norm[~valid] = 0.0

            results.append((color, depth_norm, pose))
    finally:
        renderer.delete()

    return results


def _render_depth_views_numpy(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fallback depth renderer using numpy z-buffering (no GPU, no occlusion culling).
    Returns list of (color_uint8_HxWx3, depth_float32_HxW, camera_pose_4x4).
    """
    vertices = mesh.vertices.astype(np.float32)
    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center, axis=1).max()
    distance = max(radius * 2.5, 0.01)

    results = []

    for az_deg, elev_deg in _VIEW_CONFIGS:
        pose = _make_camera_pose(az_deg, elev_deg, distance)

        # Extract camera basis from pose
        right = pose[:3, 0].astype(np.float32)
        up = pose[:3, 1].astype(np.float32)
        forward = -pose[:3, 2].astype(np.float32)  # camera looks down -Z

        # Project vertices to screen
        u = vertices @ right
        v = vertices @ up
        depth = vertices @ forward

        def norm01(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)

        u_n = norm01(u)
        v_n = norm01(v)
        d_n = norm01(depth)

        px = (u_n * (resolution - 1)).astype(int).clip(0, resolution - 1)
        py = ((1.0 - v_n) * (resolution - 1)).astype(int).clip(0, resolution - 1)

        zbuf = np.full((resolution, resolution), -1.0, dtype=np.float32)
        depth_img = np.zeros((resolution, resolution), dtype=np.float32)

        for i in range(len(vertices)):
            if d_n[i] > zbuf[py[i], px[i]]:
                zbuf[py[i], px[i]] = d_n[i]
                depth_img[py[i], px[i]] = d_n[i]

        # Create a simple grayscale color image from depth
        gray = (depth_img * 255).astype(np.uint8)
        color = np.stack([gray, gray, gray], axis=-1)

        results.append((color, depth_img, pose))

    return results


# ── Atlas projection ─────────────────────────────────────────────────────────

def _barycentric_coords(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[float, float, float]:
    """Compute barycentric coordinates of point p w.r.t. triangle (a, b, c)."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)


def _project_view(
    world_pos: np.ndarray,
    pose: np.ndarray,
    fov_y: float,
    image_size: int,
) -> tuple[int, int] | None:
    """
    Project a 3D world position into a view's 2D image coordinates.
    Returns (px, py) pixel coordinates, or None if behind the camera.
    """
    # Transform to camera space
    R = pose[:3, :3]
    t = pose[:3, 3]
    cam_pos_world = t
    cam_forward = -pose[:3, 2]  # camera looks down -Z in world

    rel = world_pos - cam_pos_world
    # Check if point is in front of camera
    if np.dot(rel, cam_forward) < 0:
        return None

    # Project using perspective (simplified pinhole)
    # Convert world→camera
    cam_right = pose[:3, 0]
    cam_up = pose[:3, 1]

    x_cam = np.dot(rel, cam_right)
    y_cam = np.dot(rel, cam_up)
    z_cam = np.dot(rel, cam_forward)

    if z_cam < 1e-6:
        return None

    half_tan = math.tan(fov_y / 2.0)
    # NDC [-1, 1]
    ndc_x = x_cam / (z_cam * half_tan)
    ndc_y = y_cam / (z_cam * half_tan)

    # Pixel coords
    px = int((ndc_x * 0.5 + 0.5) * (image_size - 1))
    py = int((1.0 - (ndc_y * 0.5 + 0.5)) * (image_size - 1))

    if 0 <= px < image_size and 0 <= py < image_size:
        return (px, py)
    return None


def _bilinear_sample(image_arr: np.ndarray, px: float, py: float) -> np.ndarray:
    """Sample image (H, W, C) using bilinear interpolation."""
    h, w = image_arr.shape[:2]
    x = max(0.0, min(float(px), w - 1.001))
    y = max(0.0, min(float(py), h - 1.001))
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    c = (
        image_arr[y0, x0] * (1 - fx) * (1 - fy)
        + image_arr[y0, x1] * fx * (1 - fy)
        + image_arr[y1, x0] * (1 - fx) * fy
        + image_arr[y1, x1] * fx * fy
    )
    return c


def _rasterize_triangle_uv(
    uv0: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
    atlas_size: int,
) -> list[tuple[int, int, float, float, float]]:
    """
    Scanline rasterize a UV triangle at atlas_size resolution.
    Yields (atlas_px, atlas_py, bary_u, bary_v, bary_w) for each texel inside.
    """
    # Convert UV [0,1] → pixel coords
    def uv_to_px(uv):
        return np.array([uv[0] * (atlas_size - 1), (1.0 - uv[1]) * (atlas_size - 1)])

    p0 = uv_to_px(uv0)
    p1 = uv_to_px(uv1)
    p2 = uv_to_px(uv2)

    # Bounding box
    min_x = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(atlas_size - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(atlas_size - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))))

    results = []
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            pt = np.array([px + 0.5, py + 0.5])
            u, v, w = _barycentric_coords(pt, p0, p1, p2)
            # Inside triangle check (with small epsilon for edge pixels)
            if u >= -0.01 and v >= -0.01 and w >= -0.01:
                results.append((px, py, u, v, w))
    return results


def _project_atlas(
    mesh: trimesh.Trimesh,
    uvs: np.ndarray,
    view_images: list[Image.Image],
    view_configs: list[tuple[float, float]],
    view_poses: list[np.ndarray],
    atlas_size: int = 1024,
    render_size: int = 512,
) -> np.ndarray:
    """
    Project view images onto the UV atlas using face-normal-weighted view selection.
    For each UV texel: find face → pick best view → sample colour via bilinear interp.
    Fills holes via nearest-neighbour after projection.
    Returns (atlas_size, atlas_size, 4) RGBA uint8 array.
    """
    # Pre-convert images to numpy
    view_arrays = []
    for img in view_images:
        arr = np.array(img.resize((render_size, render_size)).convert("RGB"), dtype=np.float32)
        view_arrays.append(arr)

    # Pre-compute view directions (toward origin)
    view_dirs = np.array(
        [_view_direction(az, elev) for az, elev in view_configs],
        dtype=np.float32,
    )

    face_normals = mesh.face_normals.astype(np.float32)
    faces = mesh.faces

    atlas = np.zeros((atlas_size, atlas_size, 4), dtype=np.float32)
    coverage = np.zeros((atlas_size, atlas_size), dtype=bool)

    fov_y = math.radians(45.0)
    n_views = len(view_dirs)

    logger.info(f"Projecting {len(faces)} faces onto {atlas_size}×{atlas_size} atlas...")

    for face_idx, face in enumerate(faces):
        v0_idx, v1_idx, v2_idx = face
        normal = face_normals[face_idx]

        # Select best view for this face
        dots = np.dot(view_dirs, normal)
        best_view = int(np.argmax(dots))

        uv0 = uvs[v0_idx]
        uv1 = uvs[v1_idx]
        uv2 = uvs[v2_idx]

        p0 = mesh.vertices[v0_idx].astype(np.float32)
        p1 = mesh.vertices[v1_idx].astype(np.float32)
        p2 = mesh.vertices[v2_idx].astype(np.float32)

        pose = view_poses[best_view]
        view_arr = view_arrays[best_view]

        for atlas_px, atlas_py, bu, bv, bw in _rasterize_triangle_uv(uv0, uv1, uv2, atlas_size):
            # Barycentric interpolation → world position
            world_pos = bu * p0 + bv * p1 + bw * p2

            # Project to view image
            screen = _project_view(world_pos, pose, fov_y, render_size)
            if screen is not None:
                px, py = screen
                color = _bilinear_sample(view_arr, px, py)
                atlas[atlas_py, atlas_px, :3] = color
                atlas[atlas_py, atlas_px, 3] = 255.0
                coverage[atlas_py, atlas_px] = True

    # ── Hole filling: nearest-neighbour for uncovered texels ────────────────
    uncovered = ~coverage
    if uncovered.any() and coverage.any():
        try:
            from scipy.ndimage import distance_transform_edt
            _, indices = distance_transform_edt(uncovered, return_indices=True)
            atlas[uncovered] = atlas[indices[0][uncovered], indices[1][uncovered]]
        except ImportError:
            # No scipy — fill with average color
            avg = atlas[coverage].mean(axis=0)
            atlas[uncovered] = avg

    return atlas.clip(0, 255).astype(np.uint8)


# ── Seam blending ────────────────────────────────────────────────────────────

def _blend_seams(
    atlas: np.ndarray,
    chart_ids: "np.ndarray | None",
    blend_radius: int = 3,
) -> np.ndarray:
    """
    Apply Gaussian blending along chart border texels to reduce visible seams.
    If chart_ids is None (box UV or LSCM fallback), skip blending.
    Returns modified atlas (H, W, 4) uint8.
    """
    if chart_ids is None:
        return atlas

    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("cv2 not available — skipping seam blending")
        return atlas

    atlas_size = atlas.shape[0]

    # Rasterize chart_ids into UV space to find border texels
    # For a simple border detection, use the atlas alpha channel as proxy:
    # border = pixels adjacent to a different chart or to an uncovered texel
    alpha = atlas[:, :, 3].astype(np.float32)
    covered = alpha > 0

    # Detect border: covered pixels adjacent to uncovered or to different value
    from scipy.ndimage import binary_dilation
    border_mask = covered & binary_dilation(~covered)

    # Dilate border mask
    dilated = binary_dilation(border_mask, iterations=blend_radius)

    if not dilated.any():
        return atlas

    # Blur entire atlas, then composite only the dilated region
    atlas_rgb = atlas[:, :, :3].astype(np.uint8)
    blurred = cv2.GaussianBlur(atlas_rgb, (2 * blend_radius + 1, 2 * blend_radius + 1), 0)

    result = atlas.copy()
    result[dilated, :3] = blurred[dilated]
    return result


# ── ControlNet / texture generation ─────────────────────────────────────────

def _generate_textures_controlnet(
    pipe,
    depth_views: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prompt: str,
    reference_image: "Image.Image | None",
    view_size: int = 512,
    out_size: int = 1024,
    conditioning_scale: float = 0.8,
) -> list[Image.Image]:
    """
    Generate one SDXL image per view conditioned on depth using ControlNet.
    """
    texture_prompt = (
        f"{prompt}, high quality 3D model texture, detailed surface, "
        "photorealistic PBR material, seamless"
    )
    white_init = Image.new("RGB", (out_size, out_size), color=(255, 255, 255))
    if reference_image is not None:
        white_init = reference_image.resize((out_size, out_size)).convert("RGB")

    images = []
    for i, (color_arr, depth_arr, pose) in enumerate(depth_views):
        logger.info(f"  ControlNet view {i + 1}/{len(depth_views)}...")
        # Convert depth to RGB for ControlNet
        depth_uint8 = (depth_arr * 255).clip(0, 255).astype(np.uint8)
        depth_rgb = Image.fromarray(np.stack([depth_uint8] * 3, axis=-1)).resize((out_size, out_size))

        try:
            import torch
            result = pipe(
                prompt=texture_prompt,
                image=white_init,
                control_image=depth_rgb,
                controlnet_conditioning_scale=conditioning_scale,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42 + i),
            )
            images.append(result.images[0])
        except Exception as exc:
            logger.warning(f"ControlNet failed for view {i}: {exc}. Using depth fallback.")
            images.append(depth_rgb)

    return images


def _generate_textures_img2img(
    pipe,
    depth_views: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prompt: str,
    reference_image: "Image.Image | None",
    out_size: int = 1024,
) -> list[Image.Image]:
    """
    Fallback: regular SDXL img2img conditioned on reference image + depth.
    """
    texture_prompt = (
        f"{prompt}, high quality 3D model texture, detailed surface, "
        "photorealistic PBR material, seamless"
    )
    images = []
    for i, (color_arr, depth_arr, pose) in enumerate(depth_views):
        logger.info(f"  img2img view {i + 1}/{len(depth_views)}...")
        init_image = (
            reference_image.resize((out_size, out_size)).convert("RGB")
            if reference_image is not None
            else Image.new("RGB", (out_size, out_size), (200, 180, 160))
        )
        try:
            import torch
            result = pipe(
                prompt=texture_prompt,
                image=init_image,
                strength=0.65,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42 + i),
            )
            images.append(result.images[0])
        except Exception as exc:
            logger.warning(f"img2img failed for view {i}: {exc}. Using depth colorization.")
            depth_uint8 = (depth_arr * 255).clip(0, 255).astype(np.uint8)
            images.append(
                Image.fromarray(np.stack([depth_uint8] * 3, axis=-1)).resize((out_size, out_size))
            )
    return images


def _generate_textures_mock(
    depth_views: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prompt: str,
    out_size: int = 1024,
) -> list[Image.Image]:
    """
    No-GPU mock: return colorized depth maps as texture views.
    """
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ("stone", "rock", "gray", "metal")):
        r, g, b = 130, 130, 140
    elif any(w in prompt_lower for w in ("wood", "brown", "bark")):
        r, g, b = 139, 90, 43
    elif any(w in prompt_lower for w in ("skin", "flesh", "human", "character")):
        r, g, b = 210, 170, 120
    else:
        r, g, b = 160, 140, 120

    images = []
    for _, depth_arr, _ in depth_views:
        depth_uint8 = (depth_arr * 255).clip(0, 255).astype(np.uint8)
        colored = np.stack([
            (depth_arr * r).clip(0, 255),
            (depth_arr * g).clip(0, 255),
            (depth_arr * b).clip(0, 255),
        ], axis=-1).astype(np.uint8)
        images.append(Image.fromarray(colored).resize((out_size, out_size)))
    return images


# ── Main service ─────────────────────────────────────────────────────────────

class UVTexturingService(SkinGenerationService):
    """
    Full UV texturing pipeline:
    - xatlas UV unwrap (or scipy LSCM fallback)
    - pyrender EGL 6-view depth rendering (or numpy z-buffer fallback)
    - ControlNet SDXL depth-conditioned generation (or img2img / mock fallback)
    - Face-normal-weighted atlas projection with barycentric sampling
    - Gaussian seam blending
    - GLB export with PBR baseColorTexture

    Pass the TextToImageService so we can reuse the same SDXL base model
    without loading it twice if already loaded.
    """

    def __init__(self, text_to_image_service=None) -> None:
        self._t2i = text_to_image_service
        self._controlnet_pipe = None   # ControlNet pipeline (best path)
        self._img2img_pipe = None      # Fallback img2img pipeline
        self._use_controlnet = False
        self._use_img2img = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load ControlNet SDXL pipeline, or fall back to img2img."""
        import torch
        from app.core.config import get_settings
        s = get_settings()

        logger.info("Loading UV texturing pipeline...")

        # Try ControlNet first
        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
            logger.info(f"Loading ControlNet model: {s.CONTROLNET_DEPTH_MODEL}")
            controlnet = ControlNetModel.from_pretrained(
                s.CONTROLNET_DEPTH_MODEL,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            self._controlnet_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                s.TEXT_TO_IMAGE_MODEL,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self._controlnet_pipe.to(s.TEXT_TO_IMAGE_DEVICE)
            self._use_controlnet = True
            logger.info("ControlNet SDXL pipeline loaded")
            return
        except Exception as exc:
            logger.warning(f"ControlNet unavailable: {exc}. Trying img2img fallback.")

        # Try plain SDXL img2img
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            self._img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                s.TEXT_TO_IMAGE_MODEL,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self._img2img_pipe.to(s.TEXT_TO_IMAGE_DEVICE)
            self._use_img2img = True
            logger.info("SDXL img2img pipeline loaded (ControlNet fallback)")
        except Exception as exc:
            logger.warning(f"img2img unavailable: {exc}. Will use mock texture generation.")

    def unload_model(self) -> None:
        """Free VRAM."""
        import torch
        if self._controlnet_pipe is not None:
            del self._controlnet_pipe
            self._controlnet_pipe = None
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
        self._use_controlnet = False
        self._use_img2img = False
        torch.cuda.empty_cache()
        logger.info("UV texturing pipeline unloaded")

    # ── Main entry point ─────────────────────────────────────────────────────

    def generate_skin(
        self,
        glb_path: Path,
        prompt: str,
        output_path: Path,
        reference_image: "Image.Image | None" = None,
    ) -> Path:
        if not glb_path.exists():
            raise FileNotFoundError(f"Mesh not found: {glb_path}")

        s = get_settings()
        atlas_size = s.UV_TEXTURE_RESOLUTION
        render_size = 512
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"UV texturing: {glb_path.name!r} | prompt: {prompt[:60]!r}")

        # ── Load mesh ────────────────────────────────────────────────────────
        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.creation.box()
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            mesh = trimesh.creation.box()

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # ── UV unwrapping ────────────────────────────────────────────────────
        mesh, uvs, chart_ids = self._unwrap_uvs(mesh)

        # ── Depth rendering ──────────────────────────────────────────────────
        depth_views = self._render_depth_views(mesh, render_size)

        # ── Texture generation ───────────────────────────────────────────────
        view_images = self._generate_textures(depth_views, prompt, reference_image, atlas_size)

        # ── Atlas projection ─────────────────────────────────────────────────
        view_poses = [pose for _, _, pose in depth_views]
        atlas = _project_atlas(
            mesh, uvs, view_images,
            view_configs=_VIEW_CONFIGS,
            view_poses=view_poses,
            atlas_size=atlas_size,
            render_size=atlas_size,
        )

        # ── Seam blending ────────────────────────────────────────────────────
        atlas = _blend_seams(atlas, chart_ids, blend_radius=s.UV_SEAM_BLEND_RADIUS)

        # ── GLB export ───────────────────────────────────────────────────────
        atlas_image = Image.fromarray(atlas[:, :, :3])
        logger.info("Building textured GLB...")
        glb_bytes = _build_glb_with_texture(mesh, uvs, atlas_image)
        output_path.write_bytes(glb_bytes)
        logger.info(f"Textured GLB: {output_path} ({len(glb_bytes):,} bytes)")
        return output_path

    # ── Internal methods ─────────────────────────────────────────────────────

    def _unwrap_uvs(
        self, mesh: trimesh.Trimesh
    ) -> tuple[trimesh.Trimesh, np.ndarray, "np.ndarray | None"]:
        """
        UV unwrap: tries xatlas first, then scipy LSCM, then box UV.
        Returns (mesh, uvs, chart_ids).  xatlas may remap vertices.
        """
        s = get_settings()
        method = s.UV_METHOD

        if method in ("xatlas", "auto"):
            try:
                import xatlas  # noqa: F401
                logger.info("UV unwrapping with xatlas...")
                new_mesh, uvs, chart_ids = _xatlas_uv_projection(mesh)
                logger.info(f"xatlas: {len(new_mesh.vertices)} verts after seam split")
                return new_mesh, uvs, chart_ids
            except ImportError:
                if method == "xatlas":
                    raise RuntimeError("xatlas requested but not available")
                logger.info("xatlas not available — trying LSCM fallback")

        if method in ("lscm", "auto"):
            logger.info("UV unwrapping with scipy LSCM...")
            uvs = _lscm_uv_projection(mesh)
            return mesh, uvs, None

        # Box UV last resort
        logger.info("UV unwrapping with box projection (fallback)")
        uvs = _box_uv_projection(mesh)
        return mesh, uvs, None

    def _render_depth_views(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 512,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Render 6 depth views: pyrender EGL first, numpy z-buffer fallback.
        """
        try:
            logger.info("Rendering depth views with pyrender EGL...")
            views = _render_depth_views_pyrender(mesh, resolution)
            logger.info(f"pyrender: {len(views)} depth views rendered")
            return views
        except Exception as exc:
            logger.warning(f"pyrender unavailable ({exc}). Using numpy z-buffer fallback.")
            views = _render_depth_views_numpy(mesh, resolution)
            logger.info(f"numpy: {len(views)} depth views rendered")
            return views

    def _generate_textures(
        self,
        depth_views: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        prompt: str,
        reference_image: "Image.Image | None",
        out_size: int = 1024,
    ) -> list[Image.Image]:
        """
        Generate per-view textures: ControlNet → img2img → mock.
        """
        s = get_settings()

        if self._use_controlnet and self._controlnet_pipe is not None:
            logger.info("Generating textures with ControlNet SDXL...")
            return _generate_textures_controlnet(
                self._controlnet_pipe,
                depth_views,
                prompt,
                reference_image,
                view_size=512,
                out_size=out_size,
                conditioning_scale=s.CONTROLNET_CONDITIONING_SCALE,
            )

        if self._use_img2img and self._img2img_pipe is not None:
            logger.info("Generating textures with SDXL img2img (ControlNet fallback)...")
            return _generate_textures_img2img(
                self._img2img_pipe,
                depth_views,
                prompt,
                reference_image,
                out_size=out_size,
            )

        logger.info("Generating textures with mock colorizer (no GPU)...")
        return _generate_textures_mock(depth_views, prompt, out_size=out_size)
