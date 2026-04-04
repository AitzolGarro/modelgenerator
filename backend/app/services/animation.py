"""
Animation service: auto-rigs a GLB mesh and adds skeletal animation.

Adaptive approach:
1. Analyze mesh geometry to find actual body proportions
2. Use vertex density slicing to find torso center, limb roots, extremities
3. Place bones at geometry-derived positions, not fixed ratios
4. Use geodesic-aware segment weights based on actual mesh shape
5. Scale animation amplitudes to mesh proportions
"""

import math
import json
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import KDTree

from app.core.logging import get_logger
from app.services.base import AnimationService

logger = get_logger(__name__)


# ── Skeleton structure (hierarchy only — positions computed at runtime) ───

@dataclass
class BoneDef:
    name: str
    parent_idx: int | None = None


BONE_HIERARCHY = [
    BoneDef("root", None),        # 0
    BoneDef("hip", 0),            # 1
    BoneDef("spine", 1),          # 2
    BoneDef("chest", 2),          # 3
    BoneDef("neck", 3),           # 4
    BoneDef("head", 4),           # 5
    BoneDef("upper_arm_l", 3),    # 6
    BoneDef("lower_arm_l", 6),    # 7
    BoneDef("upper_arm_r", 3),    # 8
    BoneDef("lower_arm_r", 8),    # 9
    BoneDef("upper_leg_l", 1),    # 10
    BoneDef("lower_leg_l", 10),   # 11
    BoneDef("upper_leg_r", 1),    # 12
    BoneDef("lower_leg_r", 12),   # 13
]

NUM_BONES = len(BONE_HIERARCHY)

# ── Animation presets ────────────────────────────────────────

ANIMATION_PRESETS: dict[str, dict] = {
    "walk":   {"cycle": 1.0,  "type": "locomotion"},
    "run":    {"cycle": 0.6,  "type": "locomotion"},
    "idle":   {"cycle": 3.0,  "type": "breathing"},
    "wave":   {"cycle": 1.5,  "type": "gesture"},
    "jump":   {"cycle": 1.2,  "type": "jump"},
    "spin":   {"cycle": 2.0,  "type": "rotation"},
    "dance":  {"cycle": 1.6,  "type": "dance"},
    "attack": {"cycle": 1.0,  "type": "attack"},
    "fly":    {"cycle": 1.5,  "type": "fly"},
    "bounce": {"cycle": 0.6,  "type": "bounce"},
}


def _detect_preset(prompt: str) -> tuple[str, dict]:
    prompt_lower = prompt.lower()
    for kw, preset in ANIMATION_PRESETS.items():
        if kw in prompt_lower:
            return kw, preset
    return "idle", ANIMATION_PRESETS["idle"]


# ── Math ─────────────────────────────────────────────────────

def _quat(axis: list[float], angle: float) -> list[float]:
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    n = math.sqrt(sum(a * a for a in axis)) or 1.0
    return [axis[0]/n * s, axis[1]/n * s, axis[2]/n * s, c]

def _qi() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]

def _ease(t: float) -> float:
    return 4*t*t*t if t < 0.5 else 1 - (-2*t + 2)**3 / 2

def _ss(phase: float) -> float:
    """Smooth sine."""
    return math.sin(phase) * (1.0 - 0.1 * math.sin(phase * 2))


# ── Adaptive skeleton fitting ────────────────────────────────

def _fit_skeleton_to_mesh(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Analyze mesh vertices to find actual body part positions.
    Returns bone_positions [14, 3] and segment_info dict.
    """
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bsize = bmax - bmin
    center_x = (bmin[0] + bmax[0]) / 2
    center_z = (bmin[2] + bmax[2]) / 2

    # Normalize Y to [0, 1]
    y_norm = (vertices[:, 1] - bmin[1]) / max(bsize[1], 1e-6)

    # Find vertical density distribution — where is the "meat" of the mesh?
    num_slices = 20
    slice_counts = np.zeros(num_slices)
    slice_widths_x = np.zeros(num_slices)
    slice_centroids = np.zeros((num_slices, 3))

    for s in range(num_slices):
        lo, hi = s / num_slices, (s + 1) / num_slices
        mask = (y_norm >= lo) & (y_norm < hi)
        count = mask.sum()
        slice_counts[s] = count
        if count > 10:
            sv = vertices[mask]
            slice_widths_x[s] = sv[:, 0].max() - sv[:, 0].min()
            slice_centroids[s] = sv.mean(axis=0)

    # Find the widest slice (likely shoulder/chest area)
    chest_slice = int(np.argmax(slice_widths_x))
    chest_y_ratio = (chest_slice + 0.5) / num_slices

    # Find where vertex count drops significantly at top (neck/head boundary)
    # Look above chest for a narrowing
    head_start = chest_slice
    for s in range(chest_slice + 1, num_slices):
        if slice_widths_x[s] < slice_widths_x[chest_slice] * 0.5:
            head_start = s
            break

    head_y_ratio = head_start / num_slices

    # Find hip area — look below chest for where width narrows then might split (legs)
    hip_slice = max(0, chest_slice - 4)
    for s in range(chest_slice - 1, 0, -1):
        if slice_counts[s] > 0:
            hip_slice = s
            if slice_widths_x[s] < slice_widths_x[chest_slice] * 0.7:
                break

    hip_y_ratio = (hip_slice + 0.5) / num_slices

    # Detect left/right arm protrusions at chest height
    chest_mask = (y_norm >= chest_y_ratio - 0.08) & (y_norm < chest_y_ratio + 0.08)
    chest_verts = vertices[chest_mask]
    if len(chest_verts) > 50:
        left_arm_x = chest_verts[:, 0].min()
        right_arm_x = chest_verts[:, 0].max()
        # Arm roots at edges of torso
        torso_width = np.percentile(chest_verts[:, 0], 80) - np.percentile(chest_verts[:, 0], 20)
        arm_l_root_x = center_x - torso_width * 0.5
        arm_r_root_x = center_x + torso_width * 0.5
    else:
        torso_width = bsize[0] * 0.4
        arm_l_root_x = center_x - torso_width * 0.5
        arm_r_root_x = center_x + torso_width * 0.5
        left_arm_x = bmin[0]
        right_arm_x = bmax[0]

    # Detect left/right leg split at hip height
    hip_mask = (y_norm >= hip_y_ratio - 0.1) & (y_norm < hip_y_ratio + 0.05)
    hip_verts = vertices[hip_mask]
    if len(hip_verts) > 50:
        leg_l_x = np.percentile(hip_verts[:, 0], 25)
        leg_r_x = np.percentile(hip_verts[:, 0], 75)
    else:
        leg_l_x = center_x - bsize[0] * 0.15
        leg_r_x = center_x + bsize[0] * 0.15

    # Compute bone world positions based on analysis
    spine_y_ratio = (hip_y_ratio + chest_y_ratio) / 2
    neck_y_ratio = (chest_y_ratio + head_y_ratio) / 2
    head_top_ratio = min(1.0, head_y_ratio + (1.0 - head_y_ratio) * 0.6)

    def _y(ratio):
        return bmin[1] + ratio * bsize[1]

    bone_positions = np.array([
        [center_x, bmin[1], center_z],                              # 0: root
        [center_x, _y(hip_y_ratio), center_z],                     # 1: hip
        [center_x, _y(spine_y_ratio), center_z],                   # 2: spine
        [center_x, _y(chest_y_ratio), center_z],                   # 3: chest
        [center_x, _y(neck_y_ratio), center_z],                    # 4: neck
        [center_x, _y(head_top_ratio), center_z],                  # 5: head
        [arm_l_root_x, _y(chest_y_ratio), center_z],               # 6: upper_arm_l
        [(arm_l_root_x + left_arm_x) / 2, _y(chest_y_ratio - 0.05), center_z],  # 7: lower_arm_l
        [arm_r_root_x, _y(chest_y_ratio), center_z],               # 8: upper_arm_r
        [(arm_r_root_x + right_arm_x) / 2, _y(chest_y_ratio - 0.05), center_z], # 9: lower_arm_r
        [leg_l_x, _y(hip_y_ratio - 0.05), center_z],               # 10: upper_leg_l
        [leg_l_x, _y(max(0.02, hip_y_ratio * 0.3)), center_z],     # 11: lower_leg_l
        [leg_r_x, _y(hip_y_ratio - 0.05), center_z],               # 12: upper_leg_r
        [leg_r_x, _y(max(0.02, hip_y_ratio * 0.3)), center_z],     # 13: lower_leg_r
    ], dtype=np.float32)

    segment_info = {
        "hip_y": hip_y_ratio,
        "chest_y": chest_y_ratio,
        "neck_y": neck_y_ratio,
        "head_y": head_y_ratio,
        "spine_y": spine_y_ratio,
        "torso_width": torso_width,
        "center_x": center_x,
        "leg_l_x": leg_l_x,
        "leg_r_x": leg_r_x,
    }

    return bone_positions, segment_info


# ── Adaptive segment weight painting ────────────────────────

def _compute_weights(
    vertices: np.ndarray,
    bone_positions: np.ndarray,
    segment_info: dict,
    bmin: np.ndarray,
    bsize: np.ndarray,
) -> np.ndarray:
    """
    Compute per-vertex bone weights using adaptive segments.
    Uses the actual geometry analysis results for zone boundaries.
    """
    num_verts = len(vertices)
    weights = np.zeros((num_verts, NUM_BONES), dtype=np.float32)

    safe_size = np.where(bsize > 1e-6, bsize, 1.0)
    vn = (vertices - bmin) / safe_size  # normalized [0, 1]

    hip_y = segment_info["hip_y"]
    chest_y = segment_info["chest_y"]
    neck_y = segment_info["neck_y"]
    head_y = segment_info["head_y"]
    spine_y = segment_info["spine_y"]
    cx = (segment_info["center_x"] - bmin[0]) / safe_size[0]
    tw = segment_info["torso_width"] / safe_size[0]  # normalized torso width

    # Define adaptive zones per bone
    zones = {
        0:  {"y": (0.00, 0.05), "x": (0.0, 1.0), "inf": 3.0},  # root
        1:  {"y": (hip_y - 0.08, hip_y + 0.08), "x": (cx - tw*0.5, cx + tw*0.5), "inf": 1.5},  # hip
        2:  {"y": (spine_y - 0.08, spine_y + 0.08), "x": (cx - tw*0.5, cx + tw*0.5), "inf": 1.3},  # spine
        3:  {"y": (chest_y - 0.08, chest_y + 0.08), "x": (cx - tw*0.5, cx + tw*0.5), "inf": 1.3},  # chest
        4:  {"y": (neck_y - 0.05, neck_y + 0.05), "x": (cx - tw*0.3, cx + tw*0.3), "inf": 1.0},  # neck
        5:  {"y": (head_y, 1.00), "x": (cx - tw*0.4, cx + tw*0.4), "inf": 1.5},  # head
        6:  {"y": (chest_y - 0.12, chest_y + 0.08), "x": (0.0, cx - tw*0.2), "inf": 1.2},  # upper_arm_l
        7:  {"y": (chest_y - 0.20, chest_y + 0.02), "x": (0.0, cx - tw*0.3), "inf": 1.2},  # lower_arm_l
        8:  {"y": (chest_y - 0.12, chest_y + 0.08), "x": (cx + tw*0.2, 1.0), "inf": 1.2},  # upper_arm_r
        9:  {"y": (chest_y - 0.20, chest_y + 0.02), "x": (cx + tw*0.3, 1.0), "inf": 1.2},  # lower_arm_r
        10: {"y": (hip_y * 0.3, hip_y), "x": (0.0, cx), "inf": 1.3},  # upper_leg_l
        11: {"y": (0.0, hip_y * 0.4), "x": (0.0, cx), "inf": 1.3},  # lower_leg_l
        12: {"y": (hip_y * 0.3, hip_y), "x": (cx, 1.0), "inf": 1.3},  # upper_leg_r
        13: {"y": (0.0, hip_y * 0.4), "x": (cx, 1.0), "inf": 1.3},  # lower_leg_r
    }

    # Average bone spacing for sigma
    tree = KDTree(bone_positions)
    nn_dists, _ = tree.query(bone_positions, k=2)
    sigma = float(nn_dists[:, 1].mean()) * 0.7
    margin = 0.06

    for bi in range(NUM_BONES):
        z = zones[bi]
        y_lo, y_hi = z["y"]
        x_lo, x_hi = z["x"]
        inf = z["inf"]

        # Soft zone membership
        y_s = np.ones(num_verts, dtype=np.float32)
        below = vn[:, 1] < y_lo
        above = vn[:, 1] > y_hi
        y_s[below] = np.exp(-((y_lo - vn[below, 1]) / margin) ** 2)
        y_s[above] = np.exp(-((vn[above, 1] - y_hi) / margin) ** 2)

        x_s = np.ones(num_verts, dtype=np.float32)
        left = vn[:, 0] < x_lo
        right = vn[:, 0] > x_hi
        x_s[left] = np.exp(-((x_lo - vn[left, 0]) / margin) ** 2)
        x_s[right] = np.exp(-((vn[right, 0] - x_hi) / margin) ** 2)

        dists = np.linalg.norm(vertices - bone_positions[bi], axis=1)
        d_s = np.exp(-(dists / (sigma * inf)) ** 2)

        weights[:, bi] = y_s * x_s * d_s

    # Normalize
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-8] = 1.0
    weights /= row_sums

    return weights


# ── Per-bone keyframe generators ─────────────────────────────

def _gen_keyframes(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)
    s = scale

    for i, t in enumerate(times):
        p = (t / cycle) * 2 * math.pi

        if anim_type == "locomotion":
            d = {
                "hip":         lambda: (np.array([0, abs(_ss(p*2)) * s * 0.006, 0]), _quat([0,1,0], _ss(p) * 0.025)),
                "spine":       lambda: (None, _quat([0,1,0], _ss(p) * 0.018)),
                "chest":       lambda: (None, _quat([0,1,0], _ss(p) * -0.012)),
                "head":        lambda: (None, _quat([1,0,0], _ss(p*2) * 0.008)),
                "upper_leg_l": lambda: (None, _quat([1,0,0], _ss(p) * 0.22)),
                "lower_leg_l": lambda: (None, _quat([1,0,0], max(0, _ss(p - 0.5)) * 0.30)),
                "upper_leg_r": lambda: (None, _quat([1,0,0], _ss(p + math.pi) * 0.22)),
                "lower_leg_r": lambda: (None, _quat([1,0,0], max(0, _ss(p + math.pi - 0.5)) * 0.30)),
                "upper_arm_l": lambda: (None, _quat([1,0,0], _ss(p + math.pi) * 0.12)),
                "lower_arm_l": lambda: (None, _quat([1,0,0], max(0, _ss(p + math.pi)) * 0.08)),
                "upper_arm_r": lambda: (None, _quat([1,0,0], _ss(p) * 0.12)),
                "lower_arm_r": lambda: (None, _quat([1,0,0], max(0, _ss(p)) * 0.08)),
            }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

        elif anim_type == "breathing":
            d = {
                "chest": lambda: (np.array([0, math.sin(p) * s * 0.002, 0]), _quat([1,0,0], math.sin(p) * 0.006)),
                "spine": lambda: (None, _quat([1,0,0], math.sin(p) * 0.004)),
                "head":  lambda: (None, _quat([1,0,0], math.sin(p*0.5) * 0.008)),
                "neck":  lambda: (None, _quat([0,1,0], math.sin(p*0.3) * 0.004)),
            }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

        elif anim_type == "gesture":
            d = {
                "upper_arm_r": lambda: (None, _quat([0,0,1], -0.6 - 0.3 * _ease((math.sin(p)+1)/2))),
                "lower_arm_r": lambda: (None, _quat([1,0,0], -0.2 + _ss(p*2) * 0.3)),
                "head":        lambda: (None, _quat([0,1,0], _ss(p*0.8) * 0.06)),
                "spine":       lambda: (None, _quat([0,1,0], _ss(p) * 0.015)),
            }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

        elif anim_type == "jump":
            tn = (t % cycle) / cycle
            e = _ease(tn) if tn < 0.5 else _ease(1.0 - tn)
            d = {
                "root":        lambda: (np.array([0, e * s * 0.06, 0]), _qi()),
                "upper_leg_l": lambda: (None, _quat([1,0,0], -e * 0.25)),
                "upper_leg_r": lambda: (None, _quat([1,0,0], -e * 0.25)),
                "lower_leg_l": lambda: (None, _quat([1,0,0], e * 0.3)),
                "lower_leg_r": lambda: (None, _quat([1,0,0], e * 0.3)),
                "upper_arm_l": lambda: (None, _quat([0,0,1], e * 0.35)),
                "upper_arm_r": lambda: (None, _quat([0,0,1], -e * 0.35)),
                "spine":       lambda: (None, _quat([1,0,0], -e * 0.04)),
            }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

        elif anim_type == "rotation":
            if bone_name == "root":
                ro[i] = _quat([0,1,0], (t / cycle) * 2 * math.pi)

        elif anim_type == "dance":
            d = {
                "hip":         lambda: (np.array([0, abs(_ss(p*2)) * s * 0.008, 0]), _quat([0,1,0], _ss(p) * 0.10)),
                "spine":       lambda: (None, _quat([0,0,1], _ss(p) * 0.05)),
                "chest":       lambda: (None, _quat([0,1,0], _ss(p+0.5) * 0.04)),
                "upper_arm_l": lambda: (None, _quat([0,0,1], 0.25 + _ss(p) * 0.35)),
                "lower_arm_l": lambda: (None, _quat([1,0,0], -0.15 + _ss(p*2) * 0.2)),
                "upper_arm_r": lambda: (None, _quat([0,0,1], -0.25 + _ss(p+1.5) * 0.35)),
                "lower_arm_r": lambda: (None, _quat([1,0,0], -0.15 + _ss(p*2+1) * 0.2)),
                "upper_leg_l": lambda: (None, _quat([1,0,0], _ss(p) * 0.12)),
                "lower_leg_l": lambda: (None, _quat([1,0,0], max(0, _ss(p-0.3)) * 0.15)),
                "upper_leg_r": lambda: (None, _quat([1,0,0], _ss(p+math.pi) * 0.12)),
                "lower_leg_r": lambda: (None, _quat([1,0,0], max(0, _ss(p+math.pi-0.3)) * 0.15)),
                "head":        lambda: (None, _quat([0,1,0], _ss(p*2) * 0.05)),
                "neck":        lambda: (None, _quat([0,0,1], _ss(p) * 0.025)),
            }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

        elif anim_type == "attack":
            tn = (t % cycle) / cycle
            d = {}
            if tn < 0.3:
                e = _ease(tn / 0.3)
                d = {
                    "upper_arm_r": lambda: (None, _quat([1,0,0], -e * 0.6)),
                    "lower_arm_r": lambda: (None, _quat([1,0,0], -e * 0.3)),
                    "spine":       lambda: (None, _quat([0,1,0], e * 0.1)),
                }
            elif tn < 0.5:
                e = _ease((tn - 0.3) / 0.2)
                d = {
                    "upper_arm_r": lambda: (None, _quat([1,0,0], -0.6 + e * 0.9)),
                    "lower_arm_r": lambda: (None, _quat([1,0,0], -0.3 + e * 0.3)),
                    "spine":       lambda: (None, _quat([0,1,0], 0.1 - e * 0.2)),
                    "chest":       lambda: (None, _quat([1,0,0], e * 0.03)),
                }
            else:
                e = _ease((tn - 0.5) / 0.5)
                d = {
                    "upper_arm_r": lambda: (None, _quat([1,0,0], 0.3 * (1-e))),
                    "spine":       lambda: (None, _quat([0,1,0], -0.1 * (1-e))),
                }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

        elif anim_type == "fly":
            d = {
                "upper_arm_l": lambda: (None, _quat([0,0,1], 0.4 + _ss(p*2) * 0.35)),
                "upper_arm_r": lambda: (None, _quat([0,0,1], -0.4 - _ss(p*2) * 0.35)),
                "lower_arm_l": lambda: (None, _quat([0,0,1], _ss(p*2+0.3) * 0.15)),
                "lower_arm_r": lambda: (None, _quat([0,0,1], -_ss(p*2+0.3) * 0.15)),
                "root":        lambda: (np.array([0, math.sin(p) * s * 0.012, 0]), _qi()),
                "chest":       lambda: (None, _quat([1,0,0], _ss(p) * 0.015)),
            }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

        elif anim_type == "bounce":
            b = abs(math.sin(p))
            d = {
                "root":        lambda: (np.array([0, b * s * 0.03, 0]), _qi()),
                "upper_leg_l": lambda: (None, _quat([1,0,0], -b * 0.10)),
                "upper_leg_r": lambda: (None, _quat([1,0,0], -b * 0.10)),
                "lower_leg_l": lambda: (None, _quat([1,0,0], b * 0.12)),
                "lower_leg_r": lambda: (None, _quat([1,0,0], b * 0.12)),
                "spine":       lambda: (None, _quat([1,0,0], -b * 0.015)),
                "upper_arm_l": lambda: (None, _quat([0,0,1], b * 0.08)),
                "upper_arm_r": lambda: (None, _quat([0,0,1], -b * 0.08)),
            }
            if bone_name in d:
                dt, dr = d[bone_name]()
                if dt is not None: tr[i] = dt
                ro[i] = dr

    return tr, ro


# ── Main service ─────────────────────────────────────────────

class ProceduralAnimationService(AnimationService):

    def load_model(self) -> None:
        logger.info("Procedural animation service ready")

    def animate(
        self, glb_path: Path, prompt: str, output_path: Path,
        duration: float = 3.0, fps: int = 30,
    ) -> Path:
        logger.info(f"Animating {glb_path.name}: '{prompt[:60]}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        anim_name, preset = _detect_preset(prompt)
        logger.info(f"Animation: {anim_name} (cycle={preset['cycle']}s)")

        # Load mesh
        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No meshes found")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            raise ValueError(f"Unsupported: {type(scene)}")

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # Fit skeleton to geometry
        logger.info("Fitting skeleton to mesh geometry...")
        bone_positions, segment_info = _fit_skeleton_to_mesh(mesh.vertices)

        for bi, bd in enumerate(BONE_HIERARCHY):
            logger.info(f"  {bd.name:20s} pos={bone_positions[bi].round(3).tolist()}")

        bmin, bmax = mesh.bounds
        bsize = bmax - bmin
        model_height = float(bsize[1])

        # Compute weights
        logger.info("Computing adaptive segment weights...")
        weights = _compute_weights(mesh.vertices, bone_positions, segment_info, bmin, bsize)

        # Log weight distribution
        primary = weights.argmax(axis=1)
        for bi in range(NUM_BONES):
            count = (primary == bi).sum()
            pct = 100 * count / len(primary)
            logger.info(f"  {BONE_HIERARCHY[bi].name:20s} {count:5d} verts ({pct:4.1f}%)")

        # Generate keyframes
        num_frames = int(duration * fps)
        times = np.linspace(0, duration, num_frames, dtype=np.float32)

        all_trans, all_rots = [], []
        for bi, bd in enumerate(BONE_HIERARCHY):
            bt, br = _gen_keyframes(bd.name, preset["type"], times, preset["cycle"], model_height)
            all_trans.append(bt)
            all_rots.append(br)

        # Build GLB
        logger.info("Building animated GLB...")
        glb = self._build_gltf(mesh, bone_positions, weights, times, all_trans, all_rots, anim_name)

        output_path.write_bytes(glb)
        logger.info(f"Animated GLB: {output_path} ({len(glb)} bytes)")
        return output_path

    def _build_gltf(self, mesh, bone_positions, weights, times, bone_trans, bone_rots, anim_name):
        vertices = mesh.vertices.astype(np.float32)
        normals = mesh.vertex_normals.astype(np.float32)
        indices = mesh.faces.astype(np.uint32).flatten()
        nv, nf = len(vertices), len(times)

        # Skinning data
        jd = np.zeros((nv, 4), dtype=np.uint16)
        wd = np.zeros((nv, 4), dtype=np.float32)
        for v in range(nv):
            t4 = np.argsort(weights[v])[-4:][::-1]
            for k in range(4):
                jd[v,k] = t4[k]
                wd[v,k] = weights[v, t4[k]]
            ws = wd[v].sum()
            if ws > 0: wd[v] /= ws

        # IBMs
        ibms = np.zeros((NUM_BONES, 16), dtype=np.float32)
        for bi in range(NUM_BONES):
            ibm = np.eye(4, dtype=np.float32)
            ibm[0,3] = -bone_positions[bi][0]
            ibm[1,3] = -bone_positions[bi][1]
            ibm[2,3] = -bone_positions[bi][2]
            ibms[bi] = ibm.T.flatten()

        # Binary buffer
        buf = bytearray()
        bvs, accs = [], []

        def pad():
            while len(buf) % 4: buf.append(0)
        def add(d, tgt=None):
            pad(); off = len(buf); buf.extend(d)
            bv = {"buffer":0,"byteOffset":off,"byteLength":len(d)}
            if tgt: bv["target"] = tgt
            bvs.append(bv); return len(bvs)-1
        def acc(bvi,ct,cnt,at,mn=None,mx=None):
            a = {"bufferView":bvi,"componentType":ct,"count":cnt,"type":at}
            if mn: a["min"]=mn
            if mx: a["max"]=mx
            accs.append(a); return len(accs)-1

        pa = acc(add(vertices.tobytes(),34962),5126,nv,"VEC3",vertices.min(0).tolist(),vertices.max(0).tolist())
        na = acc(add(normals.tobytes(),34962),5126,nv,"VEC3")
        ia = acc(add(indices.tobytes(),34963),5125,len(indices),"SCALAR",[int(indices.min())],[int(indices.max())])
        ja = acc(add(jd.tobytes(),34962),5123,nv,"VEC4")
        wa = acc(add(wd.tobytes(),34962),5126,nv,"VEC4")
        ima = acc(add(ibms.tobytes()),5126,NUM_BONES,"MAT4")
        ta = acc(add(times.tobytes()),5126,nf,"SCALAR",[float(times[0])],[float(times[-1])])

        chs, sms = [], []
        si = 0
        for bi in range(NUM_BONES):
            bt = bone_trans[bi]
            bta = acc(add(bt.tobytes()),5126,nf,"VEC3",bt.min(0).tolist(),bt.max(0).tolist())
            sms.append({"input":ta,"output":bta,"interpolation":"LINEAR"})
            chs.append({"sampler":si,"target":{"node":bi+1,"path":"translation"}}); si+=1
            br = bone_rots[bi]
            bra = acc(add(br.tobytes()),5126,nf,"VEC4",br.min(0).tolist(),br.max(0).tolist())
            sms.append({"input":ta,"output":bra,"interpolation":"LINEAR"})
            chs.append({"sampler":si,"target":{"node":bi+1,"path":"rotation"}}); si+=1

        # Nodes
        mn = {"name":"mesh","mesh":0,"skin":0}
        bns = []
        for bi, bd in enumerate(BONE_HIERARCHY):
            lp = bone_positions[bi].copy()
            if bd.parent_idx is not None:
                lp = bone_positions[bi] - bone_positions[bd.parent_idx]
            nd = {"name":bd.name,"translation":lp.tolist()}
            ch = [j for j,b in enumerate(BONE_HIERARCHY) if b.parent_idx==bi]
            if ch: nd["children"]=[c+1 for c in ch]
            bns.append(nd)

        nodes = [mn]+bns
        roots = [bi+1 for bi,b in enumerate(BONE_HIERARCHY) if b.parent_idx is None]

        gltf = {
            "asset":{"version":"2.0","generator":"ModelGenerator"},
            "scene":0,"scenes":[{"nodes":[0]+roots}],
            "nodes":nodes,
            "meshes":[{"primitives":[{"attributes":{"POSITION":pa,"NORMAL":na,"JOINTS_0":ja,"WEIGHTS_0":wa},"indices":ia}]}],
            "skins":[{"joints":list(range(1,NUM_BONES+1)),"inverseBindMatrices":ima,"skeleton":1}],
            "animations":[{"name":anim_name,"channels":chs,"samplers":sms}],
            "buffers":[{"byteLength":len(buf)}],"bufferViews":bvs,"accessors":accs,
        }

        if hasattr(mesh.visual,'vertex_colors') and mesh.visual.vertex_colors is not None:
            try:
                c = mesh.visual.vertex_colors[:,:4].astype(np.float32)/255.0
                ca = acc(add(c.tobytes(),34962),5126,nv,"VEC4")
                gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]=ca
                gltf["buffers"][0]["byteLength"]=len(buf)
            except: pass

        pad()
        gltf["buffers"][0]["byteLength"]=len(buf)
        jb = json.dumps(gltf,separators=(",",":")).encode()
        while len(jb)%4: jb+=b" "
        tot = 12+8+len(jb)+8+len(buf)
        o = bytearray()
        o.extend(struct.pack("<III",0x46546C67,2,tot))
        o.extend(struct.pack("<II",len(jb),0x4E4F534A)); o.extend(jb)
        o.extend(struct.pack("<II",len(buf),0x004E4942)); o.extend(buf)
        return bytes(o)

    def unload_model(self):
        pass
