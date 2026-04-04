"""
Animation utilities: shared helpers for all animation services.

Extracted from animation.py so that both ProceduralAnimationService
and MocapAnimationService can import from one place without circular deps.

Includes:
- BodyType class
- BoneDef dataclass
- Skeleton templates (SKELETONS dict + individual constants)
- Animation presets + _detect_preset()
- Math helpers: _quat, _qi, _ease, _ss, _quat_multiply
- Body-type classifier: _classify_body_type(), _analyze_mesh_slices()
- Skeleton fitters: _fit_*_skeleton(), _SKELETON_FITTERS dispatch
- Skinning weight painters: _compute_weights(), _compute_weights_biped()
- Attack hand parser: _parse_attack_hand()
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from app.core.logging import get_logger

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
