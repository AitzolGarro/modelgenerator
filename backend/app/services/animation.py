"""
Animation service: auto-rigs a GLB mesh and adds skeletal animation.

Adaptive approach:
1. Classify body type from mesh geometry + prompt keywords
2. Use body-type-specific skeleton template (biped, quadruped, winged_biped, serpentine, compact)
3. Fit skeleton to actual mesh geometry
4. Compute skinning weights
5. Generate body-type-appropriate keyframe animations
6. Build generic GLB (works with any bone count)
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


# ── Body type enum ────────────────────────────────────────────

class BodyType:
    BIPED        = "biped"         # humanoid: 2 legs, 2 arms
    QUADRUPED    = "quadruped"     # 4 legs, no arms (dog, horse)
    WINGED_BIPED = "winged_biped"  # 2 legs + wings (dragon, bird, angel)
    SERPENTINE   = "serpentine"    # no legs, long body (snake, worm)
    COMPACT      = "compact"       # blob-like, no clear limbs (mushroom, rock)


# ── Skeleton structure ────────────────────────────────────────

@dataclass
class BoneDef:
    name: str
    parent_idx: int | None = None


# Per-body-type skeleton hierarchies
SKELETON_BIPED: list[BoneDef] = [
    BoneDef("root",        None),  # 0
    BoneDef("hip",         0),     # 1
    BoneDef("spine",       1),     # 2
    BoneDef("chest",       2),     # 3
    BoneDef("neck",        3),     # 4
    BoneDef("head",        4),     # 5
    BoneDef("upper_arm_l", 3),     # 6
    BoneDef("lower_arm_l", 6),     # 7
    BoneDef("upper_arm_r", 3),     # 8
    BoneDef("lower_arm_r", 8),     # 9
    BoneDef("upper_leg_l", 1),     # 10
    BoneDef("lower_leg_l", 10),    # 11
    BoneDef("upper_leg_r", 1),     # 12
    BoneDef("lower_leg_r", 12),    # 13
]

SKELETON_QUADRUPED: list[BoneDef] = [
    BoneDef("root",              None),  # 0
    BoneDef("hip",               0),     # 1
    BoneDef("spine",             1),     # 2
    BoneDef("chest",             2),     # 3
    BoneDef("neck",              3),     # 4
    BoneDef("head",              4),     # 5
    BoneDef("shoulder_l",        3),     # 6
    BoneDef("upper_front_leg_l", 6),     # 7
    BoneDef("lower_front_leg_l", 7),     # 8
    BoneDef("shoulder_r",        3),     # 9
    BoneDef("upper_front_leg_r", 9),     # 10
    BoneDef("lower_front_leg_r", 10),    # 11
    BoneDef("upper_back_leg_l",  1),     # 12
    BoneDef("lower_back_leg_l",  12),    # 13
    BoneDef("upper_back_leg_r",  1),     # 14
    BoneDef("lower_back_leg_r",  14),    # 15
    BoneDef("tail_base",         1),     # 16
]

SKELETON_WINGED_BIPED: list[BoneDef] = [
    BoneDef("root",        None),  # 0
    BoneDef("hip",         0),     # 1
    BoneDef("spine",       1),     # 2
    BoneDef("chest",       2),     # 3
    BoneDef("neck",        3),     # 4
    BoneDef("head",        4),     # 5
    BoneDef("upper_arm_l", 3),     # 6
    BoneDef("lower_arm_l", 6),     # 7
    BoneDef("upper_arm_r", 3),     # 8
    BoneDef("lower_arm_r", 8),     # 9
    BoneDef("upper_leg_l", 1),     # 10
    BoneDef("lower_leg_l", 10),    # 11
    BoneDef("upper_leg_r", 1),     # 12
    BoneDef("lower_leg_r", 12),    # 13
    BoneDef("wing_l",      3),     # 14
    BoneDef("wing_tip_l",  14),    # 15
    BoneDef("wing_r",      3),     # 16
    BoneDef("wing_tip_r",  16),    # 17
    BoneDef("tail_base",   1),     # 18
]

SKELETON_SERPENTINE: list[BoneDef] = [
    BoneDef("root",      None),  # 0
    BoneDef("segment_1", 0),     # 1
    BoneDef("segment_2", 1),     # 2
    BoneDef("segment_3", 2),     # 3
    BoneDef("segment_4", 3),     # 4
    BoneDef("segment_5", 4),     # 5
    BoneDef("segment_6", 5),     # 6
    BoneDef("head",      6),     # 7
]

SKELETON_COMPACT: list[BoneDef] = [
    BoneDef("root",  None),  # 0
    BoneDef("body",  0),     # 1
    BoneDef("top",   1),     # 2
    BoneDef("base",  0),     # 3
]

SKELETONS: dict[str, list[BoneDef]] = {
    BodyType.BIPED:        SKELETON_BIPED,
    BodyType.QUADRUPED:    SKELETON_QUADRUPED,
    BodyType.WINGED_BIPED: SKELETON_WINGED_BIPED,
    BodyType.SERPENTINE:   SKELETON_SERPENTINE,
    BodyType.COMPACT:      SKELETON_COMPACT,
}


# ── Animation presets ─────────────────────────────────────────

ANIMATION_PRESETS: dict[str, dict] = {
    "walk":   {"cycle": 1.0,  "type": "walk"},
    "run":    {"cycle": 0.6,  "type": "run"},
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


# ── Math helpers ──────────────────────────────────────────────

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


def _quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]


# ── Body type classifier ──────────────────────────────────────

_PROMPT_BODY_TYPE: list[tuple[list[str], str]] = [
    # Serpentine — MUST come before "dragon" so "chinese dragon" matches here first
    (["chinese dragon", "eastern dragon", "long dragon", "sea serpent", "sea dragon",
      "noodle dragon", "serpent dragon", "wyrm", "lindworm",
      "snake", "worm", "serpent", "eel", "naga body", "centipede", "millipede",
      "chinese", "oriental dragon"], BodyType.SERPENTINE),
    # Winged biped — western dragons and winged creatures
    (["dragon", "wyvern", "pterosaur", "angel", "fairy", "winged", "pegasus", "gryphon",
      "griffin", "harpy", "phoenix", "bat creature"], BodyType.WINGED_BIPED),
    # Quadruped
    (["horse", "dog", "cat", "wolf", "bear", "lion", "tiger", "cow", "deer", "elk",
      "moose", "sheep", "goat", "boar", "fox", "rabbit", "elephant", "rhino",
      "hippo", "dinosaur", "raptor", "triceratops", "quadruped", "four-legged",
      "canine", "feline", "equine", "bovine", "pig", "spider", "crab"], BodyType.QUADRUPED),
    # Compact
    (["mushroom", "rock", "stone", "blob", "slime", "ball", "sphere", "cube",
      "crystal", "jelly", "jellyfish", "starfish"], BodyType.COMPACT),
    # Biped (last so it doesn't override winged/quadruped with generic terms)
    (["human", "person", "knight", "warrior", "wizard", "mage", "elf", "dwarf",
      "orc", "goblin", "humanoid", "robot", "android", "soldier", "archer",
      "mage", "witch", "zombie", "skeleton", "ghost"], BodyType.BIPED),
]


def _classify_body_type(vertices: np.ndarray, prompt: str) -> str:
    """
    Detect body type from mesh geometry and prompt keywords.
    Prompt-based classification overrides geometry when present.
    """
    prompt_lower = prompt.lower()

    # Check prompt keywords first — override geometry
    for keywords, body_type in _PROMPT_BODY_TYPE:
        if any(kw in prompt_lower for kw in keywords):
            logger.info(f"Body type from prompt keywords: {body_type}")
            return body_type

    # Geometry-based classification
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bsize = bmax - bmin

    width_x  = float(bsize[0])  # left-right
    height_y = float(bsize[1])  # up-down
    depth_z  = float(bsize[2])  # front-back

    # Guard against degenerate meshes
    if height_y < 1e-6:
        return BodyType.COMPACT

    aspect_xh = width_x / height_y    # wide-to-tall ratio
    aspect_zh = depth_z / height_y    # deep-to-tall ratio
    aspect_xz = width_x / max(depth_z, 1e-6)  # wide vs deep

    # Vertical mass distribution: upper vs lower half
    mid_y = (bmin[1] + bmax[1]) / 2
    upper_mask = vertices[:, 1] > mid_y
    lower_mask = ~upper_mask
    upper_count = upper_mask.sum()
    lower_count = lower_mask.sum()
    total = len(vertices)

    # Upper half fraction (bipeds tend > 0.4, quadrupeds ~0.3-0.5 but more distributed)
    upper_fraction = upper_count / max(total, 1)

    # Slice at 20 levels and analyze width
    num_slices = 20
    y_norm = (vertices[:, 1] - bmin[1]) / max(height_y, 1e-6)
    slice_widths_x = np.zeros(num_slices)
    slice_widths_z = np.zeros(num_slices)
    slice_counts   = np.zeros(num_slices)

    for s in range(num_slices):
        lo, hi = s / num_slices, (s + 1) / num_slices
        mask = (y_norm >= lo) & (y_norm < hi)
        cnt = mask.sum()
        slice_counts[s] = cnt
        if cnt > 10:
            sv = vertices[mask]
            slice_widths_x[s] = sv[:, 0].max() - sv[:, 0].min()
            slice_widths_z[s] = sv[:, 2].max() - sv[:, 2].min()

    # Widest slice position (top-heavy = biped, mid-height = quadruped)
    widest = int(np.argmax(slice_widths_x))
    widest_ratio = widest / num_slices  # 0=bottom, 1=top

    # Count protrusions at mid-height: bipeds have 2-4 lateral extensions
    mid_slices = (num_slices // 3, 2 * num_slices // 3)

    # Decision logic
    # Serpentine: very elongated, low height relative to longest axis
    longest = max(width_x, depth_z)
    if longest / max(height_y, 1e-6) > 3.0 and height_y < width_x * 0.4 and height_y < depth_z * 0.4:
        return BodyType.SERPENTINE

    # Compact: roughly spherical or blob
    dims = sorted([width_x, height_y, depth_z])
    if dims[2] / max(dims[0], 1e-6) < 2.0 and widest_ratio < 0.6:
        # All dims similar and widest point not near top
        if aspect_xh < 1.2 and aspect_zh < 1.2:
            return BodyType.COMPACT

    # Quadruped: wider in Z (depth) than X (width), lower aspect ratio, mass distributed
    if aspect_zh > 0.8 and aspect_xh < 1.2 and widest_ratio < 0.7:
        return BodyType.QUADRUPED

    # Biped: tall (height > width), widest near top half
    if height_y > width_x * 0.8 and widest_ratio > 0.5:
        return BodyType.BIPED

    # Default fallback
    if height_y > width_x:
        return BodyType.BIPED
    else:
        return BodyType.QUADRUPED


# ── Skeleton fitting functions ────────────────────────────────

def _analyze_mesh_slices(vertices: np.ndarray, num_slices: int = 20):
    """Common mesh slice analysis shared by skeleton fitters."""
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bsize = bmax - bmin
    y_norm = (vertices[:, 1] - bmin[1]) / max(float(bsize[1]), 1e-6)

    slice_counts   = np.zeros(num_slices)
    slice_widths_x = np.zeros(num_slices)
    slice_widths_z = np.zeros(num_slices)
    slice_centroids = np.zeros((num_slices, 3))

    for s in range(num_slices):
        lo, hi = s / num_slices, (s + 1) / num_slices
        mask = (y_norm >= lo) & (y_norm < hi)
        cnt = mask.sum()
        slice_counts[s] = cnt
        if cnt > 10:
            sv = vertices[mask]
            slice_widths_x[s] = sv[:, 0].max() - sv[:, 0].min()
            slice_widths_z[s] = sv[:, 2].max() - sv[:, 2].min()
            slice_centroids[s] = sv.mean(axis=0)

    return bmin, bmax, bsize, y_norm, slice_counts, slice_widths_x, slice_widths_z, slice_centroids


def _fit_biped_skeleton(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the 14-bone biped skeleton to mesh geometry."""
    bmin, bmax, bsize, y_norm, slice_counts, slice_widths_x, _, slice_centroids = \
        _analyze_mesh_slices(vertices)

    center_x = float((bmin[0] + bmax[0]) / 2)
    center_z = float((bmin[2] + bmax[2]) / 2)
    num_slices = len(slice_counts)

    # Widest slice = chest/shoulder area
    chest_slice = int(np.argmax(slice_widths_x))
    chest_y_ratio = (chest_slice + 0.5) / num_slices

    # Find head start: narrowing above chest
    head_start = chest_slice
    for s in range(chest_slice + 1, num_slices):
        if slice_widths_x[s] < slice_widths_x[chest_slice] * 0.5:
            head_start = s
            break
    head_y_ratio = head_start / num_slices

    # Hip area: below chest where width narrows
    hip_slice = max(0, chest_slice - 4)
    for s in range(chest_slice - 1, 0, -1):
        if slice_counts[s] > 0:
            hip_slice = s
            if slice_widths_x[s] < slice_widths_x[chest_slice] * 0.7:
                break
    hip_y_ratio = (hip_slice + 0.5) / num_slices

    # Arm protrusions at chest height
    chest_mask = (y_norm >= chest_y_ratio - 0.08) & (y_norm < chest_y_ratio + 0.08)
    chest_verts = vertices[chest_mask]
    if len(chest_verts) > 50:
        torso_width = np.percentile(chest_verts[:, 0], 80) - np.percentile(chest_verts[:, 0], 20)
    else:
        torso_width = float(bsize[0]) * 0.4
    arm_l_root_x = center_x - torso_width * 0.5
    arm_r_root_x = center_x + torso_width * 0.5
    left_arm_x   = float(bmin[0])
    right_arm_x  = float(bmax[0])
    if len(chest_verts) > 50:
        left_arm_x  = float(chest_verts[:, 0].min())
        right_arm_x = float(chest_verts[:, 0].max())

    # Leg split at hip height
    hip_mask = (y_norm >= hip_y_ratio - 0.1) & (y_norm < hip_y_ratio + 0.05)
    hip_verts = vertices[hip_mask]
    if len(hip_verts) > 50:
        leg_l_x = float(np.percentile(hip_verts[:, 0], 25))
        leg_r_x = float(np.percentile(hip_verts[:, 0], 75))
    else:
        leg_l_x = center_x - float(bsize[0]) * 0.15
        leg_r_x = center_x + float(bsize[0]) * 0.15

    spine_y_ratio = (hip_y_ratio + chest_y_ratio) / 2
    neck_y_ratio  = (chest_y_ratio + head_y_ratio) / 2
    head_top_ratio = min(1.0, head_y_ratio + (1.0 - head_y_ratio) * 0.6)

    def _y(ratio):
        return float(bmin[1]) + ratio * float(bsize[1])

    bone_positions = np.array([
        [center_x, float(bmin[1]), center_z],                              # 0: root
        [center_x, _y(hip_y_ratio), center_z],                             # 1: hip
        [center_x, _y(spine_y_ratio), center_z],                           # 2: spine
        [center_x, _y(chest_y_ratio), center_z],                           # 3: chest
        [center_x, _y(neck_y_ratio), center_z],                            # 4: neck
        [center_x, _y(head_top_ratio), center_z],                          # 5: head
        [arm_l_root_x, _y(chest_y_ratio), center_z],                       # 6: upper_arm_l
        [(arm_l_root_x + left_arm_x) / 2, _y(chest_y_ratio - 0.05), center_z],  # 7: lower_arm_l
        [arm_r_root_x, _y(chest_y_ratio), center_z],                       # 8: upper_arm_r
        [(arm_r_root_x + right_arm_x) / 2, _y(chest_y_ratio - 0.05), center_z], # 9: lower_arm_r
        [leg_l_x, _y(hip_y_ratio - 0.05), center_z],                      # 10: upper_leg_l
        [leg_l_x, _y(max(0.02, hip_y_ratio * 0.3)), center_z],            # 11: lower_leg_l
        [leg_r_x, _y(hip_y_ratio - 0.05), center_z],                      # 12: upper_leg_r
        [leg_r_x, _y(max(0.02, hip_y_ratio * 0.3)), center_z],            # 13: lower_leg_r
    ], dtype=np.float32)

    segment_info = {
        "body_type": BodyType.BIPED,
        "hip_y": hip_y_ratio, "chest_y": chest_y_ratio,
        "neck_y": neck_y_ratio, "head_y": head_y_ratio,
        "spine_y": spine_y_ratio, "torso_width": torso_width,
        "center_x": center_x, "leg_l_x": leg_l_x, "leg_r_x": leg_r_x,
    }
    return bone_positions, segment_info


def _fit_quadruped_skeleton(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the 17-bone quadruped skeleton to mesh geometry."""
    bmin, bmax, bsize, y_norm, slice_counts, slice_widths_x, slice_widths_z, slice_centroids = \
        _analyze_mesh_slices(vertices)

    center_x = float((bmin[0] + bmax[0]) / 2)
    center_y = float((bmin[1] + bmax[1]) / 2)
    center_z = float((bmin[2] + bmax[2]) / 2)
    num_slices = len(slice_counts)

    # For quadrupeds, the longest axis is usually Z (depth / front-back)
    # Head is at whichever end is narrower/higher
    # Analyze Z-distribution to find head end
    z_norm = (vertices[:, 2] - float(bmin[2])) / max(float(bsize[2]), 1e-6)

    # Slice along Z to find the narrower / higher end (= head)
    z_slices = 10
    z_height = np.zeros(z_slices)
    z_width_x = np.zeros(z_slices)
    for s in range(z_slices):
        lo, hi = s / z_slices, (s + 1) / z_slices
        mask = (z_norm >= lo) & (z_norm < hi)
        cnt = mask.sum()
        if cnt > 5:
            sv = vertices[mask]
            z_height[s] = sv[:, 1].max() - sv[:, 1].min()
            z_width_x[s] = sv[:, 0].max() - sv[:, 0].min()

    # Head end = the Z-extreme with higher average height
    front_height = z_height[:z_slices//3].mean()
    back_height  = z_height[2*z_slices//3:].mean()
    head_at_front = front_height >= back_height  # head at +Z if front is higher

    if head_at_front:
        head_z   = float(bmax[2]) - float(bsize[2]) * 0.05
        tail_z   = float(bmin[2]) + float(bsize[2]) * 0.05
        spine_dir = 1.0  # head → +Z
    else:
        head_z   = float(bmin[2]) + float(bsize[2]) * 0.05
        tail_z   = float(bmax[2]) - float(bsize[2]) * 0.05
        spine_dir = -1.0

    # Spine runs along Z axis at upper body height
    body_top_y = float(bmax[1]) - float(bsize[1]) * 0.15
    body_mid_y = float(bmin[1]) + float(bsize[1]) * 0.55
    body_low_y = float(bmin[1]) + float(bsize[1]) * 0.35

    # Chest near head end, hip near tail end
    chest_z    = head_z - spine_dir * float(bsize[2]) * 0.25
    spine_z    = center_z
    hip_z      = tail_z + spine_dir * float(bsize[2]) * 0.25
    neck_z     = head_z - spine_dir * float(bsize[2]) * 0.12
    head_top_y = float(bmax[1]) - float(bsize[1]) * 0.05

    # Leg positions: 4 corners of lower body
    leg_x_l    = center_x - float(bsize[0]) * 0.25
    leg_x_r    = center_x + float(bsize[0]) * 0.25
    front_leg_z = head_z - spine_dir * float(bsize[2]) * 0.3
    back_leg_z  = tail_z + spine_dir * float(bsize[2]) * 0.3
    foot_y      = float(bmin[1]) + float(bsize[1]) * 0.05

    bone_positions = np.array([
        [center_x,  float(bmin[1]), center_z],          # 0: root
        [center_x,  body_mid_y,     hip_z],             # 1: hip
        [center_x,  body_mid_y,     spine_z],           # 2: spine
        [center_x,  body_mid_y,     chest_z],           # 3: chest
        [center_x,  body_top_y,     neck_z],            # 4: neck
        [center_x,  head_top_y,     head_z],            # 5: head
        [leg_x_l,   body_mid_y,     chest_z],           # 6: shoulder_l
        [leg_x_l,   body_low_y,     front_leg_z],       # 7: upper_front_leg_l
        [leg_x_l,   foot_y,         front_leg_z],       # 8: lower_front_leg_l
        [leg_x_r,   body_mid_y,     chest_z],           # 9: shoulder_r
        [leg_x_r,   body_low_y,     front_leg_z],       # 10: upper_front_leg_r
        [leg_x_r,   foot_y,         front_leg_z],       # 11: lower_front_leg_r
        [leg_x_l,   body_low_y,     back_leg_z],        # 12: upper_back_leg_l
        [leg_x_l,   foot_y,         back_leg_z],        # 13: lower_back_leg_l
        [leg_x_r,   body_low_y,     back_leg_z],        # 14: upper_back_leg_r
        [leg_x_r,   foot_y,         back_leg_z],        # 15: lower_back_leg_r
        [center_x,  body_mid_y,     tail_z],            # 16: tail_base
    ], dtype=np.float32)

    segment_info = {
        "body_type": BodyType.QUADRUPED,
        "center_x": center_x, "center_z": center_z,
        "body_mid_y": body_mid_y, "body_top_y": body_top_y,
        "head_z": head_z, "tail_z": tail_z,
        "front_leg_z": front_leg_z, "back_leg_z": back_leg_z,
        "leg_x_l": leg_x_l, "leg_x_r": leg_x_r,
        "foot_y": foot_y,
        "spine_dir": spine_dir,
    }
    return bone_positions, segment_info


def _fit_winged_biped_skeleton(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the 19-bone winged biped skeleton to mesh geometry."""
    # Start from biped base
    bone_positions_biped, seg_biped = _fit_biped_skeleton(vertices)
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bsize = bmax - bmin

    chest_y = seg_biped["chest_y"]
    center_x = seg_biped["center_x"]
    center_z = float((bmin[2] + bmax[2]) / 2)

    # Detect wing lateral extent (wide extensions above mid-height)
    mid_y = float(bmin[1]) + float(bsize[1]) * 0.5
    upper_mask = vertices[:, 1] > mid_y
    if upper_mask.sum() > 20:
        upper_verts = vertices[upper_mask]
        wing_l_x = float(upper_verts[:, 0].min())  # leftmost extent
        wing_r_x = float(upper_verts[:, 0].max())  # rightmost extent
        wing_tip_z = center_z + float(bsize[2]) * 0.1
    else:
        wing_l_x = float(bmin[0])
        wing_r_x = float(bmax[0])
        wing_tip_z = center_z

    def _y(ratio):
        return float(bmin[1]) + ratio * float(bsize[1])

    # Wing root slightly behind chest
    wing_y     = _y(chest_y + 0.05)
    wing_tip_y = _y(chest_y - 0.1)

    # Tail: below hip, extending behind
    tail_z = center_z - float(bsize[2]) * 0.3

    bone_positions = np.vstack([
        bone_positions_biped,                                          # 0-13: biped
        [[center_x - float(bsize[0]) * 0.2, wing_y,     center_z]],  # 14: wing_l
        [[wing_l_x,                          wing_tip_y, wing_tip_z]], # 15: wing_tip_l
        [[center_x + float(bsize[0]) * 0.2, wing_y,     center_z]],  # 16: wing_r
        [[wing_r_x,                          wing_tip_y, wing_tip_z]], # 17: wing_tip_r
        [[center_x, _y(seg_biped["hip_y"]),  tail_z]],                # 18: tail_base
    ]).astype(np.float32)

    seg = {**seg_biped, "body_type": BodyType.WINGED_BIPED,
           "wing_l_x": wing_l_x, "wing_r_x": wing_r_x,
           "tail_z": tail_z}
    return bone_positions, seg


def _fit_serpentine_skeleton(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Fit the 8-bone serpentine skeleton along the mesh's actual spine curve.
    Instead of a straight line, finds the centroid path through the mesh.
    """
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bsize = bmax - bmin

    # Find longest axis (could be X or Z — serpentine bodies are horizontal)
    horiz_axes = [0, 2]  # X and Z
    axis = horiz_axes[0] if float(bsize[0]) >= float(bsize[2]) else horiz_axes[1]

    num_bones = len(SKELETON_SERPENTINE)
    num_slices = num_bones * 4  # finer slicing for smooth path

    # Slice along the longest axis and find centroid of each slice
    # This traces the actual curve of the body
    axis_norm = (vertices[:, axis] - bmin[axis]) / max(float(bsize[axis]), 1e-6)
    centroids = []

    for s in range(num_slices):
        lo, hi = s / num_slices, (s + 1) / num_slices
        mask = (axis_norm >= lo) & (axis_norm < hi)
        if mask.sum() > 5:
            centroids.append(vertices[mask].mean(axis=0))
        elif centroids:
            centroids.append(centroids[-1].copy())
        else:
            # Interpolate from bounds
            t = (lo + hi) / 2
            pos = bmin.copy()
            pos[axis] = bmin[axis] + t * bsize[axis]
            pos[1] = (bmin[1] + bmax[1]) / 2
            centroids.append(pos)

    centroids = np.array(centroids, dtype=np.float32)

    # Sample num_bones positions along the centroid path (evenly spaced)
    bone_positions = np.zeros((num_bones, 3), dtype=np.float32)
    for bi in range(num_bones):
        t = bi / max(num_bones - 1, 1)
        idx = min(int(t * (len(centroids) - 1)), len(centroids) - 1)
        bone_positions[bi] = centroids[idx]

    # Detect which end is the head: the end that's higher (Y) or narrower
    # Slice the two ends and compare
    end_0_mask = axis_norm < 0.15
    end_1_mask = axis_norm > 0.85
    end_0_y = vertices[end_0_mask, 1].mean() if end_0_mask.sum() > 5 else bmin[1]
    end_1_y = vertices[end_1_mask, 1].mean() if end_1_mask.sum() > 5 else bmin[1]
    end_0_width = (vertices[end_0_mask, :].max(0) - vertices[end_0_mask, :].min(0)).sum() if end_0_mask.sum() > 5 else 999
    end_1_width = (vertices[end_1_mask, :].max(0) - vertices[end_1_mask, :].min(0)).sum() if end_1_mask.sum() > 5 else 999

    # Head is at the higher/narrower end
    head_at_end = 1 if (end_1_y > end_0_y + bsize[1] * 0.05 or end_1_width < end_0_width * 0.8) else 0

    if head_at_end == 0:
        # Reverse bone order so head (index 7) is at the correct end
        bone_positions = bone_positions[::-1].copy()

    # Elevate the head bone slightly
    bone_positions[-1, 1] += float(bsize[1]) * 0.15

    segment_info = {
        "body_type": BodyType.SERPENTINE,
        "axis": axis,
        "length": float(bsize[axis]),
    }
    return bone_positions, segment_info


def _fit_compact_skeleton(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the 4-bone compact skeleton for blob-like shapes."""
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bsize = bmax - bmin
    center = (bmin + bmax) / 2

    bone_positions = np.array([
        [float(center[0]), float(bmin[1]), float(center[2])],             # 0: root
        [float(center[0]), float(center[1]), float(center[2])],           # 1: body
        [float(center[0]), float(bmax[1]) - float(bsize[1])*0.1, float(center[2])],  # 2: top
        [float(center[0]), float(bmin[1]) + float(bsize[1])*0.1, float(center[2])],  # 3: base
    ], dtype=np.float32)

    segment_info = {
        "body_type": BodyType.COMPACT,
        "center": center.tolist(),
        "bsize": bsize.tolist(),
    }
    return bone_positions, segment_info


# Dispatch table for skeleton fitting
_SKELETON_FITTERS = {
    BodyType.BIPED:        _fit_biped_skeleton,
    BodyType.QUADRUPED:    _fit_quadruped_skeleton,
    BodyType.WINGED_BIPED: _fit_winged_biped_skeleton,
    BodyType.SERPENTINE:   _fit_serpentine_skeleton,
    BodyType.COMPACT:      _fit_compact_skeleton,
}


# ── Adaptive segment weight painting ──────────────────────────

def _compute_weights(
    vertices: np.ndarray,
    bone_positions: np.ndarray,
    segment_info: dict,
    bmin: np.ndarray,
    bsize: np.ndarray,
) -> np.ndarray:
    """
    Generic per-vertex bone weights via Gaussian distance falloff.
    Works with any skeleton (any bone count).
    The zones dict can optionally be supplied by per-body-type logic,
    but here we use a generic distance-based approach as fallback.
    """
    num_verts  = len(vertices)
    num_bones  = len(bone_positions)
    weights    = np.zeros((num_verts, num_bones), dtype=np.float32)
    body_type  = segment_info.get("body_type", BodyType.BIPED)

    # Bone spacing for sigma
    tree = KDTree(bone_positions)
    if num_bones > 1:
        nn_dists, _ = tree.query(bone_positions, k=min(2, num_bones))
        sigma = float(nn_dists[:, -1].mean()) * 0.7
    else:
        sigma = float(bsize.max()) * 0.5
    sigma = max(sigma, 1e-4)

    if body_type == BodyType.BIPED:
        weights = _compute_weights_biped(vertices, bone_positions, segment_info, bmin, bsize, sigma)
    else:
        # Generic: pure Gaussian distance for non-biped types
        for bi in range(num_bones):
            dists = np.linalg.norm(vertices - bone_positions[bi], axis=1)
            weights[:, bi] = np.exp(-(dists / sigma) ** 2)

    # Normalize
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-8] = 1.0
    weights /= row_sums
    return weights


def _compute_weights_biped(
    vertices, bone_positions, segment_info, bmin, bsize, sigma
) -> np.ndarray:
    """Zone-aware weight painting for biped skeleton (original logic)."""
    num_verts = len(vertices)
    num_bones = len(SKELETON_BIPED)
    weights   = np.zeros((num_verts, num_bones), dtype=np.float32)

    safe_size = np.where(bsize > 1e-6, bsize, 1.0)
    vn = (vertices - bmin) / safe_size

    hip_y   = segment_info["hip_y"]
    chest_y = segment_info["chest_y"]
    neck_y  = segment_info["neck_y"]
    head_y  = segment_info["head_y"]
    spine_y = segment_info["spine_y"]
    cx = (segment_info["center_x"] - float(bmin[0])) / float(safe_size[0])
    tw = segment_info["torso_width"] / float(safe_size[0])

    zones = {
        0:  {"y": (0.00,               0.05),           "x": (0.0,         1.0),         "inf": 3.0},
        1:  {"y": (hip_y - 0.08,       hip_y + 0.08),   "x": (cx-tw*0.5,   cx+tw*0.5),  "inf": 1.5},
        2:  {"y": (spine_y - 0.08,     spine_y + 0.08), "x": (cx-tw*0.5,   cx+tw*0.5),  "inf": 1.3},
        3:  {"y": (chest_y - 0.08,     chest_y + 0.08), "x": (cx-tw*0.5,   cx+tw*0.5),  "inf": 1.3},
        4:  {"y": (neck_y - 0.05,      neck_y + 0.05),  "x": (cx-tw*0.3,   cx+tw*0.3),  "inf": 1.0},
        5:  {"y": (head_y,             1.00),            "x": (cx-tw*0.4,   cx+tw*0.4),  "inf": 1.5},
        6:  {"y": (chest_y - 0.12,     chest_y + 0.08), "x": (0.0,         cx-tw*0.2),  "inf": 1.2},
        7:  {"y": (chest_y - 0.20,     chest_y + 0.02), "x": (0.0,         cx-tw*0.3),  "inf": 1.2},
        8:  {"y": (chest_y - 0.12,     chest_y + 0.08), "x": (cx+tw*0.2,   1.0),        "inf": 1.2},
        9:  {"y": (chest_y - 0.20,     chest_y + 0.02), "x": (cx+tw*0.3,   1.0),        "inf": 1.2},
        10: {"y": (hip_y * 0.3,        hip_y),           "x": (0.0,         cx),         "inf": 1.3},
        11: {"y": (0.0,                hip_y * 0.4),     "x": (0.0,         cx),         "inf": 1.3},
        12: {"y": (hip_y * 0.3,        hip_y),           "x": (cx,          1.0),        "inf": 1.3},
        13: {"y": (0.0,                hip_y * 0.4),     "x": (cx,          1.0),        "inf": 1.3},
    }

    margin = 0.06
    for bi in range(num_bones):
        z = zones[bi]
        y_lo, y_hi = z["y"]
        x_lo, x_hi = z["x"]
        inf_val    = z["inf"]

        y_s = np.ones(num_verts, dtype=np.float32)
        below = vn[:, 1] < y_lo
        above = vn[:, 1] > y_hi
        y_s[below] = np.exp(-((y_lo - vn[below, 1]) / margin) ** 2)
        y_s[above] = np.exp(-((vn[above, 1] - y_hi) / margin) ** 2)

        x_s = np.ones(num_verts, dtype=np.float32)
        left  = vn[:, 0] < x_lo
        right = vn[:, 0] > x_hi
        x_s[left]  = np.exp(-((x_lo - vn[left, 0]) / margin) ** 2)
        x_s[right] = np.exp(-((vn[right, 0] - x_hi) / margin) ** 2)

        dists = np.linalg.norm(vertices - bone_positions[bi], axis=1)
        d_s = np.exp(-(dists / (sigma * inf_val)) ** 2)

        weights[:, bi] = y_s * x_s * d_s

    return weights


# ── Attack hand helper ────────────────────────────────────────

def _parse_attack_hand(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ["left", "izquierda", "shield", "escudo"]):
        return "left"
    return "right"


# ── Per-body-type keyframe generators ─────────────────────────

def _gen_keyframes_biped(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Original biped keyframe generator — all 14 biped bones."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)
    s  = scale

    for i, t in enumerate(times):
        p    = (t / cycle) * 2 * math.pi
        dt_v = None

        if anim_type == "walk":
            r = {
                "root":        _quat([0,0,1], _ss(p) * 0.008),
                "hip":         _quat([0,1,0], _ss(p) * 0.03),
                "spine":       _quat([0,1,0], _ss(p) * 0.02),
                "chest":       _quat([0,1,0], _ss(p) * -0.015),
                "neck":        _quat([0,1,0], _ss(p) * -0.008),
                "head":        _quat([1,0,0], _ss(p*2) * 0.01),
                "upper_leg_l": _quat([1,0,0], _ss(p) * 0.28),
                "lower_leg_l": _quat([1,0,0], max(0, _ss(p - 0.6)) * 0.40),
                "upper_leg_r": _quat([1,0,0], _ss(p + math.pi) * 0.28),
                "lower_leg_r": _quat([1,0,0], max(0, _ss(p + math.pi - 0.6)) * 0.40),
                "upper_arm_l": _quat([1,0,0], _ss(p + math.pi) * 0.18),
                "lower_arm_l": _quat([1,0,0], max(0, _ss(p + math.pi - 0.2)) * 0.12),
                "upper_arm_r": _quat([1,0,0], _ss(p) * 0.18),
                "lower_arm_r": _quat([1,0,0], max(0, _ss(p - 0.2)) * 0.12),
            }
            if bone_name == "hip":
                dt_v = np.array([0, abs(_ss(p*2)) * s * 0.008, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "run":
            r = {
                "root":        _quat([1,0,0], -0.06),
                "hip":         _quat([0,1,0], _ss(p) * 0.04),
                "spine":       _quat([1,0,0], -0.03 + _ss(p) * 0.02),
                "chest":       _quat([1,0,0], -0.02 + _ss(p) * -0.015),
                "neck":        _quat([1,0,0], 0.04),
                "head":        _quat([1,0,0], 0.03 + _ss(p*2) * 0.012),
                "upper_leg_l": _quat([1,0,0], _ss(p) * 0.45),
                "lower_leg_l": _quat([1,0,0], max(0, _ss(p - 0.4)) * 0.65),
                "upper_leg_r": _quat([1,0,0], _ss(p + math.pi) * 0.45),
                "lower_leg_r": _quat([1,0,0], max(0, _ss(p + math.pi - 0.4)) * 0.65),
                "upper_arm_l": _quat([1,0,0], _ss(p + math.pi) * 0.35),
                "lower_arm_l": _quat([1,0,0], 0.3 + _ss(p + math.pi) * 0.15),
                "upper_arm_r": _quat([1,0,0], _ss(p) * 0.35),
                "lower_arm_r": _quat([1,0,0], 0.3 + _ss(p) * 0.15),
            }
            if bone_name == "hip":
                dt_v = np.array([0, abs(_ss(p*2)) * s * 0.014, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "breathing":
            sp = math.sin(p)
            r = {
                "root":        _quat([0,0,1], sp * 0.002),
                "hip":         _quat([1,0,0], sp * 0.003),
                "spine":       _quat([1,0,0], sp * 0.005),
                "chest":       _quat([1,0,0], sp * 0.008),
                "neck":        _quat([1,0,0], sp * -0.003),
                "head":        _quat([1,0,0], math.sin(p*0.5) * 0.01),
                "upper_arm_l": _quat([0,0,1], sp * 0.005),
                "lower_arm_l": _quat([1,0,0], sp * 0.003),
                "upper_arm_r": _quat([0,0,1], sp * -0.005),
                "lower_arm_r": _quat([1,0,0], sp * 0.003),
                "upper_leg_l": _quat([1,0,0], sp * 0.002),
                "lower_leg_l": _qi(),
                "upper_leg_r": _quat([1,0,0], sp * 0.002),
                "lower_leg_r": _qi(),
            }
            if bone_name == "chest":
                dt_v = np.array([0, sp * s * 0.003, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "gesture":
            r = {
                "root":        _quat([0,1,0], _ss(p*0.3) * 0.01),
                "hip":         _quat([0,1,0], _ss(p*0.5) * 0.01),
                "spine":       _quat([0,1,0], _ss(p) * 0.02),
                "chest":       _quat([0,1,0], _ss(p) * 0.015),
                "neck":        _quat([0,1,0], _ss(p) * 0.02),
                "head":        _quat([0,1,0], _ss(p*0.8) * 0.06),
                "upper_arm_l": _quat([0,0,1], _ss(p*0.5) * 0.03),
                "lower_arm_l": _quat([1,0,0], _ss(p*0.5) * 0.02),
                "upper_arm_r": _quat([0,0,1], -0.7 - 0.3 * _ease((math.sin(p)+1)/2)),
                "lower_arm_r": _quat([1,0,0], -0.3 + _ss(p*2) * 0.35),
                "upper_leg_l": _quat([1,0,0], _ss(p*0.3) * 0.01),
                "lower_leg_l": _qi(),
                "upper_leg_r": _quat([1,0,0], _ss(p*0.3) * 0.01),
                "lower_leg_r": _qi(),
            }
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "jump":
            tn = (t % cycle) / cycle
            e  = _ease(tn) if tn < 0.5 else _ease(1.0 - tn)
            r = {
                "root":        _qi(),
                "hip":         _quat([1,0,0], -e * 0.05),
                "spine":       _quat([1,0,0], -e * 0.04),
                "chest":       _quat([1,0,0], e * 0.02),
                "neck":        _quat([1,0,0], e * 0.03),
                "head":        _quat([1,0,0], e * 0.02),
                "upper_leg_l": _quat([1,0,0], -e * 0.30),
                "lower_leg_l": _quat([1,0,0], e * 0.40),
                "upper_leg_r": _quat([1,0,0], -e * 0.30),
                "lower_leg_r": _quat([1,0,0], e * 0.40),
                "upper_arm_l": _quat([0,0,1], e * 0.40),
                "lower_arm_l": _quat([1,0,0], -e * 0.15),
                "upper_arm_r": _quat([0,0,1], -e * 0.40),
                "lower_arm_r": _quat([1,0,0], -e * 0.15),
            }
            if bone_name == "root":
                dt_v = np.array([0, e * s * 0.08, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "rotation":
            angle = (t / cycle) * 2 * math.pi
            r = {
                "root":        _quat([0,1,0], angle),
                "hip":         _qi(),
                "spine":       _qi(),
                "chest":       _qi(),
                "neck":        _qi(),
                "head":        _qi(),
                "upper_arm_l": _quat([0,0,1], math.sin(angle) * 0.05),
                "lower_arm_l": _qi(),
                "upper_arm_r": _quat([0,0,1], -math.sin(angle) * 0.05),
                "lower_arm_r": _qi(),
                "upper_leg_l": _qi(),
                "lower_leg_l": _qi(),
                "upper_leg_r": _qi(),
                "lower_leg_r": _qi(),
            }
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "dance":
            r = {
                "root":        _quat([0,0,1], _ss(p*0.5) * 0.02),
                "hip":         _quat([0,1,0], _ss(p) * 0.12),
                "spine":       _quat([0,0,1], _ss(p) * 0.06),
                "chest":       _quat([0,1,0], _ss(p+0.5) * 0.05),
                "neck":        _quat([0,0,1], _ss(p) * 0.03),
                "head":        _quat([0,1,0], _ss(p*2) * 0.06),
                "upper_arm_l": _quat([0,0,1], 0.3 + _ss(p) * 0.40),
                "lower_arm_l": _quat([1,0,0], -0.2 + _ss(p*2) * 0.25),
                "upper_arm_r": _quat([0,0,1], -0.3 + _ss(p+1.5) * 0.40),
                "lower_arm_r": _quat([1,0,0], -0.2 + _ss(p*2+1) * 0.25),
                "upper_leg_l": _quat([1,0,0], _ss(p) * 0.15),
                "lower_leg_l": _quat([1,0,0], max(0, _ss(p-0.3)) * 0.20),
                "upper_leg_r": _quat([1,0,0], _ss(p+math.pi) * 0.15),
                "lower_leg_r": _quat([1,0,0], max(0, _ss(p+math.pi-0.3)) * 0.20),
            }
            if bone_name == "hip":
                dt_v = np.array([_ss(p) * s * 0.005, abs(_ss(p*2)) * s * 0.01, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "attack":
            tn   = (t % cycle) / cycle
            hand = _parse_attack_hand(prompt)
            ua   = f"upper_arm_{hand[0]}"
            la   = f"lower_arm_{hand[0]}"
            ua_o = "upper_arm_l" if hand == "right" else "upper_arm_r"
            la_o = "lower_arm_l" if hand == "right" else "lower_arm_r"
            sd   = 1 if hand == "right" else -1

            if tn < 0.3:
                e = _ease(tn / 0.3)
                r = {
                    "root":   _quat([0,1,0], sd*e*0.04), "hip": _quat([0,1,0], sd*e*0.03),
                    "spine":  _quat([0,1,0], sd*e*0.12), "chest": _quat([0,1,0], sd*e*0.06),
                    "neck":   _quat([0,1,0], sd*e*-0.03), "head": _quat([0,1,0], sd*e*-0.04),
                    ua: _quat([1,0,0], -e*0.7), la: _quat([1,0,0], -e*0.5),
                    ua_o: _quat([1,0,0], e*0.05), la_o: _quat([1,0,0], e*0.03),
                    "upper_leg_l": _quat([1,0,0], e*0.03), "lower_leg_l": _qi(),
                    "upper_leg_r": _quat([1,0,0], -e*0.03), "lower_leg_r": _qi(),
                }
            elif tn < 0.55:
                e = _ease((tn - 0.3) / 0.25)
                r = {
                    "root":  _quat([0,1,0], sd*(0.04-e*0.10)), "hip": _quat([0,1,0], sd*(0.03-e*0.06)),
                    "spine": _quat([0,1,0], sd*(0.12-e*0.24)), "chest": _quat([0,1,0], sd*(0.06-e*0.12)),
                    "neck":  _quat([0,1,0], sd*(-0.03+e*0.06)), "head": _quat([0,1,0], sd*(-0.04+e*0.02)),
                    ua: _quat([1,0,0], -0.7+e*1.2), la: _quat([1,0,0], -0.5+e*0.7),
                    ua_o: _quat([1,0,0], 0.05-e*0.05), la_o: _quat([1,0,0], 0.03-e*0.03),
                    "upper_leg_l": _quat([1,0,0], 0.03+e*0.02), "lower_leg_l": _quat([1,0,0], e*0.03),
                    "upper_leg_r": _quat([1,0,0], -0.03-e*0.02), "lower_leg_r": _quat([1,0,0], e*0.03),
                }
            else:
                e = _ease((tn - 0.55) / 0.45)
                r = {
                    "root":  _quat([0,1,0], sd*-0.06*(1-e)), "hip": _quat([0,1,0], sd*-0.03*(1-e)),
                    "spine": _quat([0,1,0], sd*-0.12*(1-e)), "chest": _quat([0,1,0], sd*-0.06*(1-e)),
                    "neck":  _quat([0,1,0], sd*0.03*(1-e)), "head": _quat([0,1,0], sd*-0.02*(1-e)),
                    ua: _quat([1,0,0], 0.5*(1-e)), la: _quat([1,0,0], 0.2*(1-e)),
                    ua_o: _qi(), la_o: _qi(),
                    "upper_leg_l": _quat([1,0,0], 0.05*(1-e)), "lower_leg_l": _quat([1,0,0], 0.03*(1-e)),
                    "upper_leg_r": _quat([1,0,0], -0.05*(1-e)), "lower_leg_r": _quat([1,0,0], 0.03*(1-e)),
                }
            if bone_name == "hip":
                dt_v = np.array([0, 0, sd * _ss(p*0.5) * s * 0.003])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "fly":
            r = {
                "root":        _quat([1,0,0], _ss(p*0.5) * 0.02),
                "hip":         _quat([1,0,0], _ss(p) * 0.01),
                "spine":       _quat([1,0,0], _ss(p) * 0.01),
                "chest":       _quat([1,0,0], _ss(p) * 0.02),
                "neck":        _quat([1,0,0], -_ss(p) * 0.01),
                "head":        _quat([1,0,0], _ss(p*0.5) * 0.01),
                "upper_arm_l": _quat([0,0,1], 0.5 + _ss(p*2) * 0.40),
                "lower_arm_l": _quat([0,0,1], _ss(p*2+0.3) * 0.20),
                "upper_arm_r": _quat([0,0,1], -0.5 - _ss(p*2) * 0.40),
                "lower_arm_r": _quat([0,0,1], -_ss(p*2+0.3) * 0.20),
                "upper_leg_l": _quat([1,0,0], 0.1 + _ss(p) * 0.05),
                "lower_leg_l": _quat([1,0,0], _ss(p) * 0.03),
                "upper_leg_r": _quat([1,0,0], 0.1 + _ss(p+0.5) * 0.05),
                "lower_leg_r": _quat([1,0,0], _ss(p+0.5) * 0.03),
            }
            if bone_name == "root":
                dt_v = np.array([0, math.sin(p) * s * 0.015, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "bounce":
            b = abs(math.sin(p))
            r = {
                "root":        _quat([0,0,1], _ss(p*0.5) * 0.005),
                "hip":         _quat([1,0,0], -b * 0.03),
                "spine":       _quat([1,0,0], -b * 0.02),
                "chest":       _quat([1,0,0], b * 0.015),
                "neck":        _quat([1,0,0], b * 0.01),
                "head":        _quat([1,0,0], b * 0.015),
                "upper_arm_l": _quat([0,0,1], b * 0.10),
                "lower_arm_l": _quat([1,0,0], b * 0.05),
                "upper_arm_r": _quat([0,0,1], -b * 0.10),
                "lower_arm_r": _quat([1,0,0], b * 0.05),
                "upper_leg_l": _quat([1,0,0], -b * 0.12),
                "lower_leg_l": _quat([1,0,0], b * 0.15),
                "upper_leg_r": _quat([1,0,0], -b * 0.12),
                "lower_leg_r": _quat([1,0,0], b * 0.15),
            }
            if bone_name == "root":
                dt_v = np.array([0, b * s * 0.04, 0])
            ro[i] = r.get(bone_name, _qi())

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


def _gen_keyframes_quadruped(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Quadruped keyframe generator — diagonal gait walk, gallop run."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)
    s  = scale

    for i, t in enumerate(times):
        p    = (t / cycle) * 2 * math.pi
        dt_v = None

        if anim_type in ("walk", "run"):
            # Diagonal gait: front-left + back-right together, then swap
            # Phase offsets: FL=0, BR=0, FR=π, BL=π
            amp = 0.35 if anim_type == "walk" else 0.55

            r = {
                "root":              _quat([0,0,1], _ss(p) * 0.005),
                "hip":               _quat([0,1,0], _ss(p) * 0.03),
                "spine":             _quat([0,1,0], _ss(p) * 0.02),
                "chest":             _quat([0,1,0], _ss(p) * -0.02),
                "neck":              _quat([1,0,0], _ss(p) * 0.02),
                "head":              _quat([1,0,0], _ss(p*2) * 0.01),
                "shoulder_l":        _quat([1,0,0], _ss(p + math.pi) * amp * 0.3),
                "upper_front_leg_l": _quat([1,0,0], _ss(p + math.pi) * amp),
                "lower_front_leg_l": _quat([1,0,0], max(0, _ss(p + math.pi - 0.5)) * amp * 0.8),
                "shoulder_r":        _quat([1,0,0], _ss(p) * amp * 0.3),
                "upper_front_leg_r": _quat([1,0,0], _ss(p) * amp),
                "lower_front_leg_r": _quat([1,0,0], max(0, _ss(p - 0.5)) * amp * 0.8),
                "upper_back_leg_l":  _quat([1,0,0], _ss(p) * amp),
                "lower_back_leg_l":  _quat([1,0,0], max(0, _ss(p - 0.5)) * amp * 0.8),
                "upper_back_leg_r":  _quat([1,0,0], _ss(p + math.pi) * amp),
                "lower_back_leg_r":  _quat([1,0,0], max(0, _ss(p + math.pi - 0.5)) * amp * 0.8),
                "tail_base":         _quat([0,1,0], _ss(p) * 0.15),
            }
            if bone_name == "hip":
                bob = abs(_ss(p*2)) * s * 0.008
                dt_v = np.array([0, bob, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "breathing":
            sp = math.sin(p)
            r = {
                "root":              _quat([0,0,1], sp * 0.001),
                "hip":               _quat([1,0,0], sp * 0.005),
                "spine":             _quat([1,0,0], sp * 0.006),
                "chest":             _quat([1,0,0], sp * 0.008),
                "neck":              _quat([1,0,0], sp * 0.004),
                "head":              _quat([1,0,0], math.sin(p*0.5) * 0.008),
                "tail_base":         _quat([0,1,0], sp * 0.05),
            }
            if bone_name == "chest":
                dt_v = np.array([0, sp * s * 0.003, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type in ("bounce", "jump"):
            b = abs(math.sin(p)) if anim_type == "bounce" else _ease(((t % cycle) / cycle))
            r = {
                "root":              _qi(),
                "hip":               _quat([1,0,0], -b * 0.06),
                "spine":             _quat([1,0,0], -b * 0.04),
                "chest":             _quat([1,0,0], b * 0.02),
                "neck":              _quat([1,0,0], b * 0.03),
                "head":              _quat([1,0,0], b * 0.02),
                "upper_front_leg_l": _quat([1,0,0], -b * 0.25),
                "lower_front_leg_l": _quat([1,0,0], b * 0.35),
                "upper_front_leg_r": _quat([1,0,0], -b * 0.25),
                "lower_front_leg_r": _quat([1,0,0], b * 0.35),
                "upper_back_leg_l":  _quat([1,0,0], -b * 0.25),
                "lower_back_leg_l":  _quat([1,0,0], b * 0.35),
                "upper_back_leg_r":  _quat([1,0,0], -b * 0.25),
                "lower_back_leg_r":  _quat([1,0,0], b * 0.35),
                "tail_base":         _quat([1,0,0], b * 0.20),
            }
            if bone_name == "root":
                dt_v = np.array([0, b * s * 0.06, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "rotation":
            angle = (t / cycle) * 2 * math.pi
            r = {"root": _quat([0,1,0], angle)}
            ro[i] = r.get(bone_name, _qi())

        else:
            # Fallback: breathing for unsupported types
            sp = math.sin(p)
            r = {
                "spine": _quat([1,0,0], sp * 0.005),
                "chest": _quat([1,0,0], sp * 0.008),
                "head":  _quat([1,0,0], math.sin(p*0.5) * 0.01),
                "tail_base": _quat([0,1,0], sp * 0.08),
            }
            ro[i] = r.get(bone_name, _qi())

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


def _gen_keyframes_winged_biped(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Winged biped: biped base + wing flap + tail sway."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)

    # Wings and tail get special treatment; other bones delegate to biped
    WING_BONES = {"wing_l", "wing_tip_l", "wing_r", "wing_tip_r", "tail_base"}

    for i, t in enumerate(times):
        p = (t / cycle) * 2 * math.pi
        dt_v = None

        if bone_name in WING_BONES:
            if anim_type in ("fly", "walk", "run", "breathing"):
                flap_amp = 0.8 if anim_type == "fly" else 0.3
                wing_r = {
                    "wing_l":      _quat([0,0,1],  0.3 + _ss(p*2) * flap_amp),
                    "wing_tip_l":  _quat([0,0,1],  0.2 + _ss(p*2 + 0.5) * flap_amp * 0.5),
                    "wing_r":      _quat([0,0,1], -0.3 - _ss(p*2) * flap_amp),
                    "wing_tip_r":  _quat([0,0,1], -0.2 - _ss(p*2 + 0.5) * flap_amp * 0.5),
                    "tail_base":   _quat([1,0,0], _ss(p) * 0.15),
                }
                ro[i] = wing_r.get(bone_name, _qi())
                if bone_name == "wing_l" and anim_type == "fly":
                    dt_v = np.array([0, math.sin(p*2) * scale * 0.01, 0])
            else:
                # Idle / other: gentle wing droop
                r = {
                    "wing_l":    _quat([0,0,1], 0.1 + math.sin(p * 0.5) * 0.05),
                    "wing_tip_l":_quat([0,0,1], 0.05),
                    "wing_r":    _quat([0,0,1], -0.1 - math.sin(p * 0.5) * 0.05),
                    "wing_tip_r":_quat([0,0,1], -0.05),
                    "tail_base": _quat([0,1,0], _ss(p * 0.5) * 0.08),
                }
                ro[i] = r.get(bone_name, _qi())
        else:
            # Delegate to biped for non-wing bones
            bt_single, br_single = _gen_keyframes_biped(
                bone_name, anim_type, times[i:i+1], cycle, scale, prompt
            )
            ro[i] = br_single[0]
            if np.any(bt_single[0] != 0):
                dt_v = bt_single[0]

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


def _gen_keyframes_serpentine(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Serpentine animation: wave propagation along body segments.
    Walk/slither: horizontal S-wave (Y-axis rotation)
    Fly: vertical undulation (X-axis rotation) + horizontal sway
    Idle: gentle breathing wave
    """
    n = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)

    name_to_idx = {
        "root": 0, "segment_1": 1, "segment_2": 2, "segment_3": 3,
        "segment_4": 4, "segment_5": 5, "segment_6": 6, "head": 7,
    }
    bi = name_to_idx.get(bone_name, 0)
    num_seg = 8
    # Phase offset: wave propagates from tail (root=0) to head (7)
    # Each segment is offset so the wave travels along the body
    phase_offset = bi * (2 * math.pi / num_seg)
    # Amplitude decreases slightly toward head (tail whips more)
    amp_scale = 1.0 - bi * 0.04  # root=1.0, head=0.72

    for i, t in enumerate(times):
        p = (t / cycle) * 2 * math.pi

        if anim_type in ("walk", "run"):
            # Slither: horizontal S-wave (rotate around Y)
            amp_h = (0.20 if anim_type == "walk" else 0.30) * amp_scale
            wave_h = _ss(p - phase_offset)

            # Also slight vertical undulation
            amp_v = amp_h * 0.3
            wave_v = _ss(p - phase_offset + math.pi * 0.5)

            # Combine: Y-rotation (horizontal sway) + X-rotation (vertical bob)
            ry = wave_h * amp_h
            rx = wave_v * amp_v

            # Head: more pitch, less yaw — looking where it's going
            if bone_name == "head":
                rx = _ss(p * 0.5) * 0.06
                ry = wave_h * amp_h * 0.5
                tr[i] = np.array([0, abs(wave_v) * scale * 0.004, 0])

            # Root: also slight lateral translation for ground contact feel
            if bone_name == "root":
                tr[i] = np.array([wave_h * scale * 0.008, 0, 0])

            ro[i] = _quat_multiply(
                _quat([0, 1, 0], ry),
                _quat([1, 0, 0], rx),
            )

        elif anim_type == "fly":
            # Flying serpent/dragon: primary vertical undulation + horizontal sway
            # Body makes flowing S-curves in 3D
            amp_v = 0.25 * amp_scale  # vertical (pitch)
            amp_h = 0.12 * amp_scale  # horizontal (yaw)
            amp_r = 0.06 * amp_scale  # roll

            wave_v = _ss(p - phase_offset)
            wave_h = _ss(p - phase_offset + math.pi * 0.3)
            wave_r = _ss(p * 0.5 - phase_offset * 0.5)

            rx = wave_v * amp_v
            ry = wave_h * amp_h
            rz = wave_r * amp_r

            # Root: vertical bobbing for altitude variation
            if bone_name == "root":
                tr[i] = np.array([0, _ss(p) * scale * 0.015, 0])

            # Head: look forward/up, less side-to-side
            if bone_name == "head":
                rx = _ss(p * 0.5) * 0.08  # gentle pitch
                ry = wave_h * amp_h * 0.3
                rz = 0

            ro[i] = _quat_multiply(
                _quat_multiply(
                    _quat([1, 0, 0], rx),
                    _quat([0, 1, 0], ry),
                ),
                _quat([0, 0, 1], rz),
            )

        elif anim_type == "breathing":
            sp = math.sin(p + bi * math.pi * 0.3)
            rx = sp * 0.015
            ry = sp * 0.02
            if bone_name == "head":
                rx = math.sin(p * 0.5) * 0.03
                ry = math.sin(p * 0.3) * 0.015
            ro[i] = _quat_multiply(
                _quat([1, 0, 0], rx),
                _quat([0, 1, 0], ry),
            )

        elif anim_type == "attack":
            # Lunge forward then recoil
            tn = (t % cycle) / cycle
            if tn < 0.3:
                e = _ease(tn / 0.3)
                # Coil back (segments curve backward)
                ang = -e * 0.15 * (num_seg - bi) / num_seg
                ro[i] = _quat([1, 0, 0], ang)
                if bone_name == "root":
                    tr[i] = np.array([0, 0, -e * scale * 0.02])
            elif tn < 0.5:
                e = _ease((tn - 0.3) / 0.2)
                # Strike forward
                ang = (-0.15 + e * 0.4) * (num_seg - bi) / num_seg
                ro[i] = _quat([1, 0, 0], ang)
                if bone_name == "root":
                    tr[i] = np.array([0, 0, (-0.02 + e * 0.06) * scale])
                if bone_name == "head":
                    ro[i] = _quat([1, 0, 0], e * 0.3)  # head lunges forward/down
            else:
                e = _ease((tn - 0.5) / 0.5)
                ang = 0.25 * (1 - e) * (num_seg - bi) / num_seg
                ro[i] = _quat([1, 0, 0], ang)
                if bone_name == "root":
                    tr[i] = np.array([0, 0, 0.04 * (1 - e) * scale])

        elif anim_type == "rotation":
            if bone_name == "root":
                ro[i] = _quat([0, 1, 0], (t / cycle) * 2 * math.pi)

        else:
            # Default: gentle flowing sway
            wave = _ss(p - phase_offset)
            ro[i] = _quat_multiply(
                _quat([0, 1, 0], wave * 0.12 * amp_scale),
                _quat([1, 0, 0], wave * 0.04 * amp_scale),
            )

    return tr, ro


def _gen_keyframes_compact(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Compact: squash-and-stretch + gentle wobble."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)

    for i, t in enumerate(times):
        p  = (t / cycle) * 2 * math.pi
        sp = math.sin(p)
        b  = abs(math.sin(p))
        dt_v = None

        if anim_type in ("walk", "run", "bounce"):
            r = {
                "root": _quat([0,0,1], sp * 0.02),
                "body": _quat([0,0,1], sp * 0.05),
                "top":  _quat([1,0,0], sp * 0.04),
                "base": _quat([1,0,0], -sp * 0.02),
            }
            if bone_name == "body":
                dt_v = np.array([0, b * scale * 0.04, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "breathing":
            r = {
                "root": _quat([0,0,1], sp * 0.005),
                "body": _quat([1,0,0], sp * 0.008),
                "top":  _quat([1,0,0], sp * 0.01),
                "base": _qi(),
            }
            if bone_name == "body":
                dt_v = np.array([0, sp * scale * 0.005, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "rotation":
            angle = (t / cycle) * 2 * math.pi
            if bone_name == "root":
                ro[i] = _quat([0,1,0], angle)

        else:
            r = {
                "body": _quat([0,1,0], sp * 0.08),
                "top":  _quat([0,0,1], sp * 0.05),
            }
            ro[i] = r.get(bone_name, _qi())

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


# Dispatch table: (body_type) → keyframe generator
_KF_GENERATORS = {
    BodyType.BIPED:        _gen_keyframes_biped,
    BodyType.QUADRUPED:    _gen_keyframes_quadruped,
    BodyType.WINGED_BIPED: _gen_keyframes_winged_biped,
    BodyType.SERPENTINE:   _gen_keyframes_serpentine,
    BodyType.COMPACT:      _gen_keyframes_compact,
}


# ── Backwards-compatible wrapper (kept for any external callers) ───────

def _gen_keyframes(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
    body_type: str = BodyType.BIPED,
) -> tuple[np.ndarray, np.ndarray]:
    """Route to the correct per-body-type keyframe generator."""
    gen = _KF_GENERATORS.get(body_type, _gen_keyframes_biped)
    return gen(bone_name, anim_type, times, cycle, scale, prompt)


# ── Main service ──────────────────────────────────────────────

class ProceduralAnimationService(AnimationService):

    def load_model(self) -> None:
        logger.info("Procedural animation service ready (multi-body-type)")

    def animate(
        self, glb_path: Path, prompt: str, output_path: Path,
        duration: float = 3.0, fps: int = 30,
    ) -> Path:
        logger.info(f"Animating {glb_path.name}: '{prompt[:60]}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        anim_name, preset = _detect_preset(prompt)
        logger.info(f"Animation: {anim_name} (cycle={preset['cycle']}s)")

        # 1. Load mesh
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

        # 2. Classify body type
        body_type = _classify_body_type(mesh.vertices, prompt)
        logger.info(f"Detected body type: {body_type}")

        # 3. Select skeleton template and fit to mesh
        bone_hierarchy = SKELETONS[body_type]
        fitter = _SKELETON_FITTERS[body_type]
        logger.info(f"Fitting {body_type} skeleton ({len(bone_hierarchy)} bones) to mesh geometry...")
        bone_positions, segment_info = fitter(mesh.vertices)

        for bi, bd in enumerate(bone_hierarchy):
            logger.info(f"  {bd.name:24s} pos={bone_positions[bi].round(3).tolist()}")

        bmin, bmax = mesh.bounds
        bsize       = bmax - bmin
        model_height = float(bsize[1])

        # 4. Compute weights
        logger.info("Computing skinning weights...")
        weights = _compute_weights(mesh.vertices, bone_positions, segment_info, bmin, bsize)

        # Log weight distribution
        num_bones = len(bone_hierarchy)
        primary   = weights.argmax(axis=1)
        for bi in range(num_bones):
            count = (primary == bi).sum()
            pct   = 100 * count / len(primary)
            logger.info(f"  {bone_hierarchy[bi].name:24s} {count:5d} verts ({pct:4.1f}%)")

        # 5. Compute local (parent-relative) positions for bind pose
        local_positions = np.zeros_like(bone_positions)
        for bi, bd in enumerate(bone_hierarchy):
            if bd.parent_idx is not None:
                local_positions[bi] = bone_positions[bi] - bone_positions[bd.parent_idx]
            else:
                local_positions[bi] = bone_positions[bi]

        # 6. Generate keyframes
        num_frames  = int(duration * fps)
        times       = np.linspace(0, duration, num_frames, dtype=np.float32)
        gen         = _KF_GENERATORS[body_type]

        all_trans, all_rots = [], []
        for bi, bd in enumerate(bone_hierarchy):
            bt_delta, br = gen(
                bd.name, preset["type"], times, preset["cycle"], model_height, prompt
            )
            # Translation = bind pose local position + animation delta
            # (glTF animation REPLACES node translation, so bind pose must be in keyframe)
            bt = bt_delta + local_positions[bi].astype(np.float32)
            all_trans.append(bt)
            all_rots.append(br)

        # 7. Build GLB
        logger.info("Building animated GLB...")
        glb = self._build_gltf(
            mesh, bone_hierarchy, bone_positions, weights,
            times, all_trans, all_rots, anim_name
        )

        output_path.write_bytes(glb)
        logger.info(f"Animated GLB: {output_path} ({len(glb)} bytes)")
        return output_path

    def _build_gltf(
        self, mesh, bone_hierarchy: list[BoneDef], bone_positions: np.ndarray,
        weights: np.ndarray, times: np.ndarray,
        bone_trans: list, bone_rots: list, anim_name: str
    ) -> bytes:
        """Generic glTF builder — works with any skeleton (any bone count)."""
        vertices = mesh.vertices.astype(np.float32)
        normals  = mesh.vertex_normals.astype(np.float32)
        indices  = mesh.faces.astype(np.uint32).flatten()
        nv       = len(vertices)
        nf       = len(times)
        nb       = len(bone_hierarchy)  # dynamic bone count

        # Skinning data (top-4 bone influences per vertex)
        jd = np.zeros((nv, 4), dtype=np.uint16)
        wd = np.zeros((nv, 4), dtype=np.float32)
        for v in range(nv):
            t4 = np.argsort(weights[v])[-4:][::-1]
            for k in range(4):
                jd[v, k] = t4[k]
                wd[v, k] = weights[v, t4[k]]
            ws = wd[v].sum()
            if ws > 0:
                wd[v] /= ws

        # Inverse bind matrices
        ibms = np.zeros((nb, 16), dtype=np.float32)
        for bi in range(nb):
            ibm = np.eye(4, dtype=np.float32)
            ibm[0, 3] = -bone_positions[bi][0]
            ibm[1, 3] = -bone_positions[bi][1]
            ibm[2, 3] = -bone_positions[bi][2]
            ibms[bi]  = ibm.T.flatten()

        # Binary buffer construction
        buf = bytearray()
        bvs, accs = [], []

        def pad():
            while len(buf) % 4: buf.append(0)

        def add(d, tgt=None):
            pad()
            off = len(buf)
            buf.extend(d)
            bv = {"buffer": 0, "byteOffset": off, "byteLength": len(d)}
            if tgt:
                bv["target"] = tgt
            bvs.append(bv)
            return len(bvs) - 1

        def acc(bvi, ct, cnt, at, mn=None, mx=None):
            a = {"bufferView": bvi, "componentType": ct, "count": cnt, "type": at}
            if mn: a["min"] = mn
            if mx: a["max"] = mx
            accs.append(a)
            return len(accs) - 1

        pa  = acc(add(vertices.tobytes(), 34962), 5126, nv, "VEC3",
                  vertices.min(0).tolist(), vertices.max(0).tolist())
        na  = acc(add(normals.tobytes(),  34962), 5126, nv, "VEC3")
        ia  = acc(add(indices.tobytes(),  34963), 5125, len(indices), "SCALAR",
                  [int(indices.min())], [int(indices.max())])
        ja  = acc(add(jd.tobytes(),       34962), 5123, nv, "VEC4")
        wa  = acc(add(wd.tobytes(),       34962), 5126, nv, "VEC4")
        ima = acc(add(ibms.tobytes()),             5126, nb, "MAT4")
        ta  = acc(add(times.tobytes()),            5126, nf, "SCALAR",
                  [float(times[0])], [float(times[-1])])

        chs, sms = [], []
        si = 0
        for bi in range(nb):
            bt  = bone_trans[bi]
            bta = acc(add(bt.tobytes()), 5126, nf, "VEC3",
                      bt.min(0).tolist(), bt.max(0).tolist())
            sms.append({"input": ta, "output": bta, "interpolation": "LINEAR"})
            chs.append({"sampler": si, "target": {"node": bi + 1, "path": "translation"}})
            si += 1

            br  = bone_rots[bi]
            bra = acc(add(br.tobytes()), 5126, nf, "VEC4",
                      br.min(0).tolist(), br.max(0).tolist())
            sms.append({"input": ta, "output": bra, "interpolation": "LINEAR"})
            chs.append({"sampler": si, "target": {"node": bi + 1, "path": "rotation"}})
            si += 1

        # Nodes: mesh node + one node per bone
        mn  = {"name": "mesh", "mesh": 0, "skin": 0}
        bns = []
        for bi, bd in enumerate(bone_hierarchy):
            lp = bone_positions[bi].copy()
            if bd.parent_idx is not None:
                lp = bone_positions[bi] - bone_positions[bd.parent_idx]
            nd = {"name": bd.name, "translation": lp.tolist()}
            ch = [j for j, b in enumerate(bone_hierarchy) if b.parent_idx == bi]
            if ch:
                nd["children"] = [c + 1 for c in ch]
            bns.append(nd)

        nodes = [mn] + bns
        roots = [bi + 1 for bi, b in enumerate(bone_hierarchy) if b.parent_idx is None]

        gltf = {
            "asset":      {"version": "2.0", "generator": "ModelGenerator"},
            "scene":      0,
            "scenes":     [{"nodes": [0] + roots}],
            "nodes":      nodes,
            "meshes":     [{"primitives": [{"attributes": {
                "POSITION": pa, "NORMAL": na,
                "JOINTS_0": ja, "WEIGHTS_0": wa
            }, "indices": ia}]}],
            "skins":      [{"joints": list(range(1, nb + 1)),
                            "inverseBindMatrices": ima, "skeleton": 1}],
            "animations": [{"name": anim_name, "channels": chs, "samplers": sms}],
            "buffers":    [{"byteLength": len(buf)}],
            "bufferViews": bvs,
            "accessors":   accs,
        }

        # Optional vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            try:
                c  = mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255.0
                ca = acc(add(c.tobytes(), 34962), 5126, nv, "VEC4")
                gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = ca
            except Exception:
                pass

        pad()
        gltf["buffers"][0]["byteLength"] = len(buf)
        jb = json.dumps(gltf, separators=(",", ":")).encode()
        while len(jb) % 4:
            jb += b" "

        tot = 12 + 8 + len(jb) + 8 + len(buf)
        o = bytearray()
        o.extend(struct.pack("<III", 0x46546C67, 2, tot))
        o.extend(struct.pack("<II", len(jb), 0x4E4F534A))
        o.extend(jb)
        o.extend(struct.pack("<II", len(buf), 0x004E4942))
        o.extend(buf)
        return bytes(o)

    def unload_model(self):
        pass
