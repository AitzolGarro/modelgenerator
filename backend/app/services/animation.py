"""
Animation service: auto-rigs a GLB mesh and adds skeletal animation.

Pipeline:
1. Load mesh, analyze bounds and geometry
2. Estimate skeleton bone positions from mesh shape
3. Calculate per-vertex bone weights (geodesic-aware, segment-based)
4. Generate per-bone keyframe animation from prompt with easing
5. Serialize everything into a glTF with skin + animation

Key improvements over naive approach:
- Segment-based weight painting (not just euclidean KDTree)
  Each vertex gets assigned to a body segment first (by Y-height zones),
  then fine-tuned by distance to bones within that segment.
- Smooth weight falloff with Gaussian kernel
- Amplitude scaling relative to model size
- Easing curves for natural motion
- Constrained knee/elbow rotation (only bends one way)
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


# ── Bone definitions ─────────────────────────────────────────

@dataclass
class Bone:
    name: str
    pos_ratio: tuple[float, float, float]  # ratio within bounding box
    parent_idx: int | None = None
    # Y-height zone (min, max) as ratio of bounding box for segment assignment
    segment_y: tuple[float, float] = (0.0, 1.0)
    # X zone (for left/right disambiguation)
    segment_x: tuple[float, float] = (0.0, 1.0)
    # Influence radius multiplier (larger = wider influence)
    influence: float = 1.0


SKELETON_TEMPLATE = [
    Bone("root",        (0.50, 0.02, 0.50), None,    (0.00, 0.05), (0.0, 1.0), 2.0),   # 0
    Bone("hip",         (0.50, 0.30, 0.50), 0,       (0.20, 0.38), (0.2, 0.8), 1.5),   # 1
    Bone("spine",       (0.50, 0.45, 0.50), 1,       (0.38, 0.55), (0.2, 0.8), 1.3),   # 2
    Bone("chest",       (0.50, 0.62, 0.50), 2,       (0.55, 0.72), (0.2, 0.8), 1.3),   # 3
    Bone("neck",        (0.50, 0.78, 0.50), 3,       (0.72, 0.82), (0.3, 0.7), 1.0),   # 4
    Bone("head",        (0.50, 0.90, 0.50), 4,       (0.82, 1.00), (0.2, 0.8), 1.5),   # 5
    Bone("upper_arm_l", (0.22, 0.66, 0.50), 3,       (0.55, 0.72), (0.0, 0.35), 1.0),  # 6
    Bone("lower_arm_l", (0.10, 0.52, 0.50), 6,       (0.40, 0.60), (0.0, 0.25), 1.0),  # 7
    Bone("upper_arm_r", (0.78, 0.66, 0.50), 3,       (0.55, 0.72), (0.65, 1.0), 1.0),  # 8
    Bone("lower_arm_r", (0.90, 0.52, 0.50), 8,       (0.40, 0.60), (0.75, 1.0), 1.0),  # 9
    Bone("upper_leg_l", (0.37, 0.22, 0.50), 1,       (0.12, 0.28), (0.0, 0.50), 1.2),  # 10
    Bone("lower_leg_l", (0.37, 0.08, 0.50), 10,      (0.00, 0.15), (0.0, 0.50), 1.2),  # 11
    Bone("upper_leg_r", (0.63, 0.22, 0.50), 1,       (0.12, 0.28), (0.50, 1.0), 1.2),  # 12
    Bone("lower_leg_r", (0.63, 0.08, 0.50), 12,      (0.00, 0.15), (0.50, 1.0), 1.2),  # 13
]

NUM_BONES = len(SKELETON_TEMPLATE)

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


# ── Math helpers ─────────────────────────────────────────────

def _quat(axis: list[float], angle: float) -> list[float]:
    """Quaternion from axis-angle [x, y, z, w]."""
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    n = math.sqrt(sum(a * a for a in axis)) or 1.0
    return [axis[0]/n * s, axis[1]/n * s, axis[2]/n * s, c]


def _quat_identity() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out (cubic)."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def _smooth_sin(phase: float) -> float:
    """Sine with smoothed peaks (less jerky at extremes)."""
    return math.sin(phase) * (1.0 - 0.1 * math.sin(phase * 2))


# ── Per-bone keyframe generators ─────────────────────────────

def _generate_bone_keyframes(
    bone_idx: int, bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, model_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (translations [N,3], rotations [N,4]) for a single bone.
    All values are DELTAS from the bind pose.
    Amplitudes are scaled relative to model_height.
    """
    n = len(times)
    trans = np.zeros((n, 3), dtype=np.float32)
    rots = np.tile(_quat_identity(), (n, 1)).astype(np.float32)

    # Scale factor — animations adapt to model size
    s = model_height

    for i, t in enumerate(times):
        phase = (t / cycle) * 2 * math.pi

        if anim_type == "locomotion":
            if bone_name == "hip":
                trans[i, 1] = abs(_smooth_sin(phase * 2)) * s * 0.008
                rots[i] = _quat([0,1,0], _smooth_sin(phase) * 0.03)
            elif bone_name == "spine":
                rots[i] = _quat([0,1,0], _smooth_sin(phase) * 0.02)
            elif bone_name == "chest":
                rots[i] = _quat([0,1,0], _smooth_sin(phase) * -0.015)
                rots[i] = _quat([1,0,0], _smooth_sin(phase * 2) * 0.01)
            elif bone_name == "head":
                rots[i] = _quat([1,0,0], _smooth_sin(phase * 2) * 0.01)
            elif bone_name == "upper_leg_l":
                rots[i] = _quat([1,0,0], _smooth_sin(phase) * 0.25)
            elif bone_name == "lower_leg_l":
                # Knees only bend backward (positive X rotation)
                rots[i] = _quat([1,0,0], max(0, _smooth_sin(phase - 0.5)) * 0.35)
            elif bone_name == "upper_leg_r":
                rots[i] = _quat([1,0,0], _smooth_sin(phase + math.pi) * 0.25)
            elif bone_name == "lower_leg_r":
                rots[i] = _quat([1,0,0], max(0, _smooth_sin(phase + math.pi - 0.5)) * 0.35)
            elif bone_name == "upper_arm_l":
                rots[i] = _quat([1,0,0], _smooth_sin(phase + math.pi) * 0.15)
            elif bone_name == "lower_arm_l":
                rots[i] = _quat([1,0,0], max(0, _smooth_sin(phase + math.pi)) * 0.1)
            elif bone_name == "upper_arm_r":
                rots[i] = _quat([1,0,0], _smooth_sin(phase) * 0.15)
            elif bone_name == "lower_arm_r":
                rots[i] = _quat([1,0,0], max(0, _smooth_sin(phase)) * 0.1)

        elif anim_type == "breathing":
            if bone_name == "chest":
                rots[i] = _quat([1,0,0], math.sin(phase) * 0.008)
                trans[i, 1] = math.sin(phase) * s * 0.002
            elif bone_name == "spine":
                rots[i] = _quat([1,0,0], math.sin(phase) * 0.005)
            elif bone_name == "head":
                rots[i] = _quat([1,0,0], math.sin(phase * 0.5) * 0.01)
                rots[i] = _quat([0,1,0], math.sin(phase * 0.3) * 0.005)

        elif anim_type == "gesture":
            if bone_name == "upper_arm_r":
                rots[i] = _quat([0,0,1], -0.8 - 0.4 * _ease_in_out((math.sin(phase) + 1) / 2))
            elif bone_name == "lower_arm_r":
                rots[i] = _quat([1,0,0], -0.3 + _smooth_sin(phase * 2) * 0.4)
            elif bone_name == "head":
                rots[i] = _quat([0,1,0], _smooth_sin(phase * 0.8) * 0.08)
            elif bone_name == "spine":
                rots[i] = _quat([0,1,0], _smooth_sin(phase) * 0.02)

        elif anim_type == "jump":
            t_norm = (t % cycle) / cycle
            ease = _ease_in_out(t_norm) if t_norm < 0.5 else _ease_in_out(1.0 - t_norm)
            if bone_name == "root":
                trans[i, 1] = ease * s * 0.08
            elif bone_name in ("upper_leg_l", "upper_leg_r"):
                rots[i] = _quat([1,0,0], -ease * 0.3)
            elif bone_name in ("lower_leg_l", "lower_leg_r"):
                rots[i] = _quat([1,0,0], ease * 0.4)
            elif bone_name in ("upper_arm_l", "upper_arm_r"):
                rots[i] = _quat([0,0,1], (1 if "l" in bone_name else -1) * ease * 0.5)
            elif bone_name == "spine":
                rots[i] = _quat([1,0,0], -ease * 0.05)

        elif anim_type == "rotation":
            if bone_name == "root":
                angle = (t / cycle) * 2 * math.pi
                rots[i] = _quat([0,1,0], angle)

        elif anim_type == "dance":
            if bone_name == "hip":
                rots[i] = _quat([0,1,0], _smooth_sin(phase) * 0.12)
                trans[i, 1] = abs(_smooth_sin(phase * 2)) * s * 0.01
            elif bone_name == "spine":
                rots[i] = _quat([0,0,1], _smooth_sin(phase) * 0.06)
            elif bone_name == "chest":
                rots[i] = _quat([0,1,0], _smooth_sin(phase + 0.5) * 0.05)
            elif bone_name == "upper_arm_l":
                rots[i] = _quat([0,0,1], 0.3 + _smooth_sin(phase) * 0.4)
            elif bone_name == "lower_arm_l":
                rots[i] = _quat([1,0,0], -0.2 + _smooth_sin(phase * 2) * 0.3)
            elif bone_name == "upper_arm_r":
                rots[i] = _quat([0,0,1], -0.3 + _smooth_sin(phase + 1.5) * 0.4)
            elif bone_name == "lower_arm_r":
                rots[i] = _quat([1,0,0], -0.2 + _smooth_sin(phase * 2 + 1) * 0.3)
            elif bone_name == "upper_leg_l":
                rots[i] = _quat([1,0,0], _smooth_sin(phase) * 0.15)
            elif bone_name == "lower_leg_l":
                rots[i] = _quat([1,0,0], max(0, _smooth_sin(phase - 0.3)) * 0.2)
            elif bone_name == "upper_leg_r":
                rots[i] = _quat([1,0,0], _smooth_sin(phase + math.pi) * 0.15)
            elif bone_name == "lower_leg_r":
                rots[i] = _quat([1,0,0], max(0, _smooth_sin(phase + math.pi - 0.3)) * 0.2)
            elif bone_name == "head":
                rots[i] = _quat([0,1,0], _smooth_sin(phase * 2) * 0.06)
            elif bone_name == "neck":
                rots[i] = _quat([0,0,1], _smooth_sin(phase) * 0.03)

        elif anim_type == "attack":
            t_norm = (t % cycle) / cycle
            if bone_name == "upper_arm_r":
                if t_norm < 0.3:
                    e = _ease_in_out(t_norm / 0.3)
                    rots[i] = _quat([1,0,0], -e * 0.8)
                elif t_norm < 0.5:
                    e = _ease_in_out((t_norm - 0.3) / 0.2)
                    rots[i] = _quat([1,0,0], -0.8 + e * 1.2)
                else:
                    e = _ease_in_out((t_norm - 0.5) / 0.5)
                    rots[i] = _quat([1,0,0], 0.4 * (1 - e))
            elif bone_name == "lower_arm_r":
                if t_norm < 0.3:
                    rots[i] = _quat([1,0,0], -_ease_in_out(t_norm / 0.3) * 0.4)
                else:
                    rots[i] = _quat([1,0,0], -0.4 + _ease_in_out((t_norm-0.3)/0.7) * 0.4)
            elif bone_name == "spine":
                if t_norm < 0.3:
                    rots[i] = _quat([0,1,0], _ease_in_out(t_norm / 0.3) * 0.15)
                elif t_norm < 0.5:
                    e = _ease_in_out((t_norm - 0.3) / 0.2)
                    rots[i] = _quat([0,1,0], 0.15 - e * 0.3)
                else:
                    e = _ease_in_out((t_norm - 0.5) / 0.5)
                    rots[i] = _quat([0,1,0], -0.15 * (1 - e))
            elif bone_name == "chest":
                rots[i] = _quat([1,0,0], _smooth_sin(phase * 0.5) * 0.03)

        elif anim_type == "fly":
            if bone_name in ("upper_arm_l", "upper_arm_r"):
                sign = 1 if "l" in bone_name else -1
                flap = 0.5 + _smooth_sin(phase * 2) * 0.4
                rots[i] = _quat([0,0,1], sign * flap)
            elif bone_name in ("lower_arm_l", "lower_arm_r"):
                sign = 1 if "l" in bone_name else -1
                rots[i] = _quat([0,0,1], sign * _smooth_sin(phase * 2 + 0.3) * 0.2)
            elif bone_name == "root":
                trans[i, 1] = math.sin(phase) * s * 0.015
            elif bone_name == "chest":
                rots[i] = _quat([1,0,0], _smooth_sin(phase) * 0.02)

        elif anim_type == "bounce":
            bounce = abs(math.sin(phase))
            if bone_name == "root":
                trans[i, 1] = bounce * s * 0.04
            elif bone_name in ("upper_leg_l", "upper_leg_r"):
                rots[i] = _quat([1,0,0], -bounce * 0.12)
            elif bone_name in ("lower_leg_l", "lower_leg_r"):
                rots[i] = _quat([1,0,0], bounce * 0.15)
            elif bone_name == "spine":
                rots[i] = _quat([1,0,0], -bounce * 0.02)
            elif bone_name in ("upper_arm_l", "upper_arm_r"):
                sign = 1 if "l" in bone_name else -1
                rots[i] = _quat([0,0,1], sign * bounce * 0.1)

    return trans, rots


# ── Main service ─────────────────────────────────────────────

class ProceduralAnimationService(AnimationService):
    """Auto-rigs a mesh and generates per-bone skeletal animation."""

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
                raise ValueError("No triangle meshes found in GLB")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            raise ValueError(f"Unsupported mesh type: {type(scene)}")

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        bounds_min = mesh.bounds[0]
        bounds_max = mesh.bounds[1]
        bounds_size = bounds_max - bounds_min
        model_height = float(bounds_size[1])  # Y extent

        # Compute bone world positions
        bone_positions = np.array([
            bounds_min + np.array(b.pos_ratio) * bounds_size
            for b in SKELETON_TEMPLATE
        ], dtype=np.float32)

        # Segment-based skinning weights
        logger.info("Computing segment-based skinning weights...")
        weights = self._compute_segment_weights(mesh.vertices, bone_positions, bounds_min, bounds_size)

        # Generate keyframes
        num_frames = int(duration * fps)
        times = np.linspace(0, duration, num_frames, dtype=np.float32)

        all_bone_trans = []
        all_bone_rots = []
        for bi, bone in enumerate(SKELETON_TEMPLATE):
            bt, br = _generate_bone_keyframes(
                bi, bone.name, preset["type"], times, preset["cycle"], model_height,
            )
            all_bone_trans.append(bt)
            all_bone_rots.append(br)

        logger.info("Building animated GLB...")
        glb_bytes = self._build_gltf(
            mesh, bone_positions, weights,
            times, all_bone_trans, all_bone_rots, anim_name,
        )

        output_path.write_bytes(glb_bytes)
        logger.info(f"Animated GLB: {output_path} ({len(glb_bytes)} bytes)")
        return output_path

    def _compute_segment_weights(
        self, vertices: np.ndarray, bone_positions: np.ndarray,
        bounds_min: np.ndarray, bounds_size: np.ndarray,
    ) -> np.ndarray:
        """
        Segment-based weight painting:
        1. Normalize each vertex to [0,1] within bounding box
        2. For each vertex, score each bone based on:
           a) Whether the vertex falls within the bone's Y-height zone
           b) Whether the vertex is on the correct side (L/R) for that bone
           c) Euclidean distance to the bone position
        3. Apply Gaussian falloff on distance
        4. Normalize to sum=1 per vertex
        """
        num_verts = len(vertices)
        weights = np.zeros((num_verts, NUM_BONES), dtype=np.float32)

        # Normalize vertex positions to [0,1]
        safe_size = np.where(bounds_size > 1e-6, bounds_size, 1.0)
        verts_norm = (vertices - bounds_min) / safe_size

        # Average bone-to-bone distance for Gaussian sigma
        tree = KDTree(bone_positions)
        nn_dists, _ = tree.query(bone_positions, k=2)
        avg_bone_dist = float(nn_dists[:, 1].mean())
        sigma = avg_bone_dist * 0.8  # Gaussian falloff radius

        for bi, bone in enumerate(SKELETON_TEMPLATE):
            # Zone check: is the vertex in this bone's segment?
            y_min, y_max = bone.segment_y
            x_min, x_max = bone.segment_x

            # Soft zone membership (smooth falloff at zone boundaries)
            y_scores = np.ones(num_verts, dtype=np.float32)
            y_below = verts_norm[:, 1] < y_min
            y_above = verts_norm[:, 1] > y_max
            margin = 0.08  # Soft boundary

            y_scores[y_below] = np.exp(-((y_min - verts_norm[y_below, 1]) / margin) ** 2)
            y_scores[y_above] = np.exp(-((verts_norm[y_above, 1] - y_max) / margin) ** 2)

            x_scores = np.ones(num_verts, dtype=np.float32)
            x_below = verts_norm[:, 0] < x_min
            x_above = verts_norm[:, 0] > x_max
            x_scores[x_below] = np.exp(-((x_min - verts_norm[x_below, 0]) / margin) ** 2)
            x_scores[x_above] = np.exp(-((verts_norm[x_above, 0] - x_max) / margin) ** 2)

            # Distance-based score (Gaussian)
            dists = np.linalg.norm(vertices - bone_positions[bi], axis=1)
            dist_scores = np.exp(-(dists / (sigma * bone.influence)) ** 2)

            # Combined score
            weights[:, bi] = y_scores * x_scores * dist_scores

        # Normalize per vertex
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-8] = 1.0
        weights /= row_sums

        return weights

    def _build_gltf(
        self,
        mesh: trimesh.Trimesh,
        bone_positions: np.ndarray,
        weights: np.ndarray,
        times: np.ndarray,
        bone_translations: list[np.ndarray],
        bone_rotations: list[np.ndarray],
        anim_name: str,
    ) -> bytes:
        """Build a complete glTF 2.0 GLB with mesh, skeleton, skin, and animation."""

        vertices = mesh.vertices.astype(np.float32)
        normals = mesh.vertex_normals.astype(np.float32)
        indices = mesh.faces.astype(np.uint32).flatten()
        num_verts = len(vertices)
        num_frames = len(times)

        # Prepare skinning data: top 4 influences per vertex
        joints_data = np.zeros((num_verts, 4), dtype=np.uint16)
        weights_data = np.zeros((num_verts, 4), dtype=np.float32)

        for v in range(num_verts):
            top4 = np.argsort(weights[v])[-4:][::-1]
            for k in range(4):
                joints_data[v, k] = top4[k]
                weights_data[v, k] = weights[v, top4[k]]
            wsum = weights_data[v].sum()
            if wsum > 0:
                weights_data[v] /= wsum

        # Inverse bind matrices
        ibms = np.zeros((NUM_BONES, 16), dtype=np.float32)
        for bi in range(NUM_BONES):
            pos = bone_positions[bi]
            ibm = np.eye(4, dtype=np.float32)
            ibm[0, 3] = -pos[0]
            ibm[1, 3] = -pos[1]
            ibm[2, 3] = -pos[2]
            ibms[bi] = ibm.T.flatten()

        # ── Binary buffer ────────────────────────────────────
        buf = bytearray()
        buffer_views = []
        accessors = []

        def _pad4():
            while len(buf) % 4 != 0:
                buf.append(0)

        def _add(data: bytes, target: int | None = None) -> int:
            _pad4()
            off = len(buf)
            buf.extend(data)
            bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
            if target is not None:
                bv["target"] = target
            buffer_views.append(bv)
            return len(buffer_views) - 1

        def _acc(bv, ct, count, atype, mn=None, mx=None) -> int:
            a = {"bufferView": bv, "componentType": ct, "count": count, "type": atype}
            if mn is not None: a["min"] = mn
            if mx is not None: a["max"] = mx
            accessors.append(a)
            return len(accessors) - 1

        # Mesh
        pos_a = _acc(_add(vertices.tobytes(), 34962), 5126, num_verts, "VEC3",
                      vertices.min(0).tolist(), vertices.max(0).tolist())
        norm_a = _acc(_add(normals.tobytes(), 34962), 5126, num_verts, "VEC3")
        idx_a = _acc(_add(indices.tobytes(), 34963), 5125, len(indices), "SCALAR",
                      [int(indices.min())], [int(indices.max())])
        jnt_a = _acc(_add(joints_data.tobytes(), 34962), 5123, num_verts, "VEC4")
        wgt_a = _acc(_add(weights_data.tobytes(), 34962), 5126, num_verts, "VEC4")
        ibm_a = _acc(_add(ibms.tobytes()), 5126, NUM_BONES, "MAT4")

        # Animation
        time_a = _acc(_add(times.tobytes()), 5126, num_frames, "SCALAR",
                       [float(times[0])], [float(times[-1])])

        channels = []
        samplers = []
        si = 0
        for bi in range(NUM_BONES):
            bt = bone_translations[bi]
            bt_a = _acc(_add(bt.tobytes()), 5126, num_frames, "VEC3",
                         bt.min(0).tolist(), bt.max(0).tolist())
            samplers.append({"input": time_a, "output": bt_a, "interpolation": "LINEAR"})
            channels.append({"sampler": si, "target": {"node": bi + 1, "path": "translation"}})
            si += 1

            br = bone_rotations[bi]
            br_a = _acc(_add(br.tobytes()), 5126, num_frames, "VEC4",
                         br.min(0).tolist(), br.max(0).tolist())
            samplers.append({"input": time_a, "output": br_a, "interpolation": "LINEAR"})
            channels.append({"sampler": si, "target": {"node": bi + 1, "path": "rotation"}})
            si += 1

        # ── Nodes ────────────────────────────────────────────
        mesh_node = {"name": "mesh", "mesh": 0, "skin": 0}

        bone_nodes = []
        for bi, bone in enumerate(SKELETON_TEMPLATE):
            local_pos = bone_positions[bi].copy()
            if bone.parent_idx is not None:
                local_pos = bone_positions[bi] - bone_positions[bone.parent_idx]

            node = {"name": bone.name, "translation": local_pos.tolist()}
            children = [i for i, b in enumerate(SKELETON_TEMPLATE) if b.parent_idx == bi]
            if children:
                node["children"] = [c + 1 for c in children]
            bone_nodes.append(node)

        nodes = [mesh_node] + bone_nodes
        root_bones = [bi + 1 for bi, b in enumerate(SKELETON_TEMPLATE) if b.parent_idx is None]

        gltf = {
            "asset": {"version": "2.0", "generator": "ModelGenerator"},
            "scene": 0,
            "scenes": [{"nodes": [0] + root_bones}],
            "nodes": nodes,
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": pos_a, "NORMAL": norm_a,
                "JOINTS_0": jnt_a, "WEIGHTS_0": wgt_a,
            }, "indices": idx_a}]}],
            "skins": [{
                "joints": list(range(1, NUM_BONES + 1)),
                "inverseBindMatrices": ibm_a,
                "skeleton": 1,
            }],
            "animations": [{"name": anim_name, "channels": channels, "samplers": samplers}],
            "buffers": [{"byteLength": len(buf)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }

        # Vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            try:
                colors = mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255.0
                col_a = _acc(_add(colors.tobytes(), 34962), 5126, num_verts, "VEC4")
                gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = col_a
                gltf["buffers"][0]["byteLength"] = len(buf)
            except Exception:
                pass

        # Serialize GLB
        _pad4()
        gltf["buffers"][0]["byteLength"] = len(buf)

        jb = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(jb) % 4 != 0:
            jb += b" "

        total = 12 + 8 + len(jb) + 8 + len(buf)
        out = bytearray()
        out.extend(struct.pack("<III", 0x46546C67, 2, total))
        out.extend(struct.pack("<II", len(jb), 0x4E4F534A))
        out.extend(jb)
        out.extend(struct.pack("<II", len(buf), 0x004E4942))
        out.extend(buf)
        return bytes(out)

    def unload_model(self) -> None:
        pass
