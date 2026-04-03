"""
Animation service: auto-rigs a GLB mesh and adds skeletal animation.

Pipeline:
1. Load mesh, analyze bounds and geometry
2. Estimate skeleton bone positions from mesh shape
3. Calculate per-vertex bone weights (proximity-based skinning)
4. Generate per-bone keyframe animation from prompt
5. Serialize everything into a glTF with skin + animation

Supported animation types (detected from prompt):
  walk, run, idle, wave, jump, spin, dance, attack, fly, bounce
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
    # Position relative to mesh bounds (0-1 normalized)
    pos_ratio: tuple[float, float, float]
    parent_idx: int | None = None
    # Children filled at runtime
    children: list[int] = field(default_factory=list)


# Skeleton template — positions are ratios of bounding box (x, y, z)
# y is up, model centered at origin
SKELETON_TEMPLATE = [
    Bone("root",       (0.5, 0.0, 0.5)),                    # 0
    Bone("hip",        (0.5, 0.30, 0.5), parent_idx=0),     # 1
    Bone("spine",      (0.5, 0.50, 0.5), parent_idx=1),     # 2
    Bone("chest",      (0.5, 0.65, 0.5), parent_idx=2),     # 3
    Bone("neck",       (0.5, 0.80, 0.5), parent_idx=3),     # 4
    Bone("head",       (0.5, 0.92, 0.5), parent_idx=4),     # 5
    Bone("upper_arm_l",(0.25, 0.70, 0.5), parent_idx=3),    # 6
    Bone("lower_arm_l",(0.10, 0.55, 0.5), parent_idx=6),    # 7
    Bone("upper_arm_r",(0.75, 0.70, 0.5), parent_idx=3),    # 8
    Bone("lower_arm_r",(0.90, 0.55, 0.5), parent_idx=8),    # 9
    Bone("upper_leg_l",(0.35, 0.25, 0.5), parent_idx=1),    # 10
    Bone("lower_leg_l",(0.35, 0.10, 0.5), parent_idx=10),   # 11
    Bone("upper_leg_r",(0.65, 0.25, 0.5), parent_idx=1),    # 12
    Bone("lower_leg_r",(0.65, 0.10, 0.5), parent_idx=13),   # 13  (typo fix below)
]
# Fix: lower_leg_r parent should be upper_leg_r (12)
SKELETON_TEMPLATE[13] = Bone("lower_leg_r", (0.65, 0.10, 0.5), parent_idx=12)

NUM_BONES = len(SKELETON_TEMPLATE)


# ── Animation presets ────────────────────────────────────────

ANIMATION_PRESETS: dict[str, dict] = {
    "walk": {"cycle": 1.0, "type": "locomotion"},
    "run":  {"cycle": 0.6, "type": "locomotion"},
    "idle": {"cycle": 2.5, "type": "breathing"},
    "wave": {"cycle": 1.5, "type": "gesture"},
    "jump": {"cycle": 1.0, "type": "jump"},
    "spin": {"cycle": 2.0, "type": "rotation"},
    "dance":{"cycle": 2.0, "type": "dance"},
    "attack":{"cycle": 0.8, "type": "attack"},
    "fly":  {"cycle": 1.5, "type": "fly"},
    "bounce":{"cycle": 0.5, "type": "bounce"},
}


def _detect_preset(prompt: str) -> tuple[str, dict]:
    prompt_lower = prompt.lower()
    for kw, preset in ANIMATION_PRESETS.items():
        if kw in prompt_lower:
            return kw, preset
    return "idle", ANIMATION_PRESETS["idle"]


def _quat(axis: list[float], angle: float) -> list[float]:
    """Quaternion from axis-angle (x, y, z, w)."""
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    n = math.sqrt(sum(a * a for a in axis)) or 1.0
    return [axis[0]/n * s, axis[1]/n * s, axis[2]/n * s, c]


def _quat_identity() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]


# ── Per-bone keyframe generators ─────────────────────────────

def _generate_bone_keyframes(
    bone_idx: int, bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (translations [N,3], rotations [N,4]) for a single bone.
    Translations are relative offsets from bind pose.
    """
    n = len(times)
    trans = np.zeros((n, 3), dtype=np.float32)
    rots = np.tile(_quat_identity(), (n, 1)).astype(np.float32)

    for i, t in enumerate(times):
        phase = (t / cycle) * 2 * math.pi

        if anim_type == "locomotion":
            if bone_name == "hip":
                trans[i, 1] = abs(math.sin(phase * 2)) * 0.02
            elif bone_name == "upper_leg_l":
                rots[i] = _quat([1,0,0], math.sin(phase) * 0.5)
            elif bone_name == "lower_leg_l":
                rots[i] = _quat([1,0,0], max(0, math.sin(phase - 0.3)) * 0.6)
            elif bone_name == "upper_leg_r":
                rots[i] = _quat([1,0,0], math.sin(phase + math.pi) * 0.5)
            elif bone_name == "lower_leg_r":
                rots[i] = _quat([1,0,0], max(0, math.sin(phase + math.pi - 0.3)) * 0.6)
            elif bone_name == "upper_arm_l":
                rots[i] = _quat([1,0,0], math.sin(phase + math.pi) * 0.3)
            elif bone_name == "upper_arm_r":
                rots[i] = _quat([1,0,0], math.sin(phase) * 0.3)
            elif bone_name == "spine":
                rots[i] = _quat([0,1,0], math.sin(phase) * 0.05)
            elif bone_name == "chest":
                rots[i] = _quat([0,1,0], math.sin(phase) * -0.03)

        elif anim_type == "breathing":
            if bone_name == "chest":
                rots[i] = _quat([1,0,0], math.sin(phase) * 0.015)
                trans[i, 1] = math.sin(phase) * 0.005
            elif bone_name == "spine":
                rots[i] = _quat([1,0,0], math.sin(phase) * 0.01)
            elif bone_name == "head":
                rots[i] = _quat([1,0,0], math.sin(phase * 0.5) * 0.02)

        elif anim_type == "gesture":
            if bone_name == "upper_arm_r":
                rots[i] = _quat([0,0,1], -1.2 + math.sin(phase) * 0.4)
            elif bone_name == "lower_arm_r":
                rots[i] = _quat([0,0,1], math.sin(phase * 2) * 0.3)
            elif bone_name == "head":
                rots[i] = _quat([0,1,0], math.sin(phase) * 0.1)

        elif anim_type == "jump":
            t_norm = (t % cycle) / cycle
            if bone_name == "root":
                # Parabolic jump
                trans[i, 1] = max(0, 4 * t_norm * (1 - t_norm)) * 0.15
            elif bone_name in ("upper_leg_l", "upper_leg_r"):
                rots[i] = _quat([1,0,0], -0.3 * (1 - 4 * t_norm * (1 - t_norm)))
            elif bone_name in ("upper_arm_l", "upper_arm_r"):
                rots[i] = _quat([1,0,0], 0.5 * (4 * t_norm * (1 - t_norm)))

        elif anim_type == "rotation":
            if bone_name == "root":
                angle = (t / (cycle * 1.0)) * 2 * math.pi
                rots[i] = _quat([0,1,0], angle)

        elif anim_type == "dance":
            if bone_name == "hip":
                rots[i] = _quat([0,1,0], math.sin(phase) * 0.2)
                trans[i, 1] = abs(math.sin(phase * 2)) * 0.02
            elif bone_name == "upper_arm_l":
                rots[i] = _quat([0,0,1], 0.5 + math.sin(phase) * 0.8)
            elif bone_name == "upper_arm_r":
                rots[i] = _quat([0,0,1], -0.5 + math.sin(phase + 1) * 0.8)
            elif bone_name in ("upper_leg_l", "upper_leg_r"):
                offset = 0 if "l" in bone_name else math.pi
                rots[i] = _quat([1,0,0], math.sin(phase + offset) * 0.3)
            elif bone_name == "head":
                rots[i] = _quat([0,1,0], math.sin(phase * 2) * 0.15)
            elif bone_name == "chest":
                rots[i] = _quat([0,0,1], math.sin(phase) * 0.1)

        elif anim_type == "attack":
            t_norm = (t % cycle) / cycle
            if bone_name == "upper_arm_r":
                # Wind up then swing
                if t_norm < 0.4:
                    rots[i] = _quat([1,0,0], -t_norm / 0.4 * 1.5)
                else:
                    progress = (t_norm - 0.4) / 0.6
                    rots[i] = _quat([1,0,0], -1.5 + progress * 2.5)
            elif bone_name == "spine":
                if t_norm < 0.4:
                    rots[i] = _quat([0,1,0], t_norm / 0.4 * 0.3)
                else:
                    rots[i] = _quat([0,1,0], 0.3 - ((t_norm-0.4)/0.6) * 0.6)

        elif anim_type == "fly":
            if bone_name in ("upper_arm_l", "upper_arm_r"):
                sign = 1 if "l" in bone_name else -1
                rots[i] = _quat([0,0,1], sign * (0.8 + math.sin(phase * 2) * 0.6))
            elif bone_name == "root":
                trans[i, 1] = math.sin(phase) * 0.03

        elif anim_type == "bounce":
            if bone_name == "root":
                trans[i, 1] = abs(math.sin(phase)) * 0.08
            elif bone_name in ("upper_leg_l", "upper_leg_r"):
                rots[i] = _quat([1,0,0], -abs(math.sin(phase)) * 0.2)

    return trans, rots


# ── Main service ─────────────────────────────────────────────

class ProceduralAnimationService(AnimationService):
    """
    Auto-rigs a mesh and generates per-bone skeletal animation.
    Outputs a valid glTF 2.0 GLB with skin, joints, and animation.
    """

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

        # Compute skeleton positions in world space
        bounds_min = mesh.bounds[0]
        bounds_max = mesh.bounds[1]
        bounds_size = bounds_max - bounds_min

        bone_positions = []
        for bone in SKELETON_TEMPLATE:
            pos = bounds_min + np.array(bone.pos_ratio) * bounds_size
            bone_positions.append(pos)
        bone_positions = np.array(bone_positions, dtype=np.float32)

        # Calculate vertex-bone weights using proximity
        logger.info("Computing skinning weights...")
        weights = self._compute_skinning_weights(mesh.vertices, bone_positions)

        # Generate per-bone keyframes
        num_frames = int(duration * fps)
        times = np.linspace(0, duration, num_frames, dtype=np.float32)

        all_bone_trans = []
        all_bone_rots = []
        for bi, bone in enumerate(SKELETON_TEMPLATE):
            bt, br = _generate_bone_keyframes(
                bi, bone.name, preset["type"], times, preset["cycle"]
            )
            all_bone_trans.append(bt)
            all_bone_rots.append(br)

        # Build glTF
        logger.info("Building animated GLB...")
        glb_bytes = self._build_gltf(
            mesh, bone_positions, weights,
            times, all_bone_trans, all_bone_rots,
            anim_name,
        )

        output_path.write_bytes(glb_bytes)
        logger.info(f"Animated GLB: {output_path} ({len(glb_bytes)} bytes)")
        return output_path

    def _compute_skinning_weights(
        self, vertices: np.ndarray, bone_positions: np.ndarray,
        max_influences: int = 4,
    ) -> np.ndarray:
        """
        Compute per-vertex bone weights using inverse-distance weighting.
        Each vertex gets weights for the closest `max_influences` bones.
        Returns [num_verts, num_bones] weight matrix (sparse-ish).
        """
        tree = KDTree(bone_positions)
        dists, indices = tree.query(vertices, k=max_influences)

        # Inverse distance weighting
        # Add small epsilon to avoid division by zero
        inv_dists = 1.0 / (dists + 1e-6)

        # Normalize per vertex
        row_sums = inv_dists.sum(axis=1, keepdims=True)
        inv_dists /= row_sums

        # Build dense weight matrix
        num_verts = len(vertices)
        weights = np.zeros((num_verts, NUM_BONES), dtype=np.float32)
        for v in range(num_verts):
            for k in range(max_influences):
                bone_idx = indices[v, k]
                weights[v, bone_idx] = inv_dists[v, k]

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

        # Prepare skinning data (4 joints + 4 weights per vertex)
        joints_data = np.zeros((num_verts, 4), dtype=np.uint16)
        weights_data = np.zeros((num_verts, 4), dtype=np.float32)

        for v in range(num_verts):
            # Get top 4 bone influences
            bone_weights = weights[v]
            top4 = np.argsort(bone_weights)[-4:][::-1]
            for k in range(4):
                bi = top4[k]
                joints_data[v, k] = bi
                weights_data[v, k] = bone_weights[bi]
            # Re-normalize
            wsum = weights_data[v].sum()
            if wsum > 0:
                weights_data[v] /= wsum

        # Inverse bind matrices (transform from mesh space to bone-local space)
        # IBM = inverse of the bone's global transform (world-space bind pose matrix)
        # For translation-only bones: IBM negates the WORLD position of the bone.
        ibms = np.zeros((NUM_BONES, 16), dtype=np.float32)
        for bi in range(NUM_BONES):
            pos = bone_positions[bi]  # world-space bind pose position
            ibm = np.eye(4, dtype=np.float32)
            ibm[0, 3] = -pos[0]
            ibm[1, 3] = -pos[1]
            ibm[2, 3] = -pos[2]
            ibms[bi] = ibm.T.flatten()  # Column-major for glTF

        # ── Pack binary buffer ───────────────────────────────
        buf = bytearray()
        buffer_views = []
        accessors = []

        def _pad4(b: bytearray):
            while len(b) % 4 != 0:
                b.append(0)

        def _add_data(data: bytes, target: int | None = None) -> int:
            """Add data to buffer, return bufferView index."""
            _pad4(buf)
            offset = len(buf)
            buf.extend(data)
            bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
            if target is not None:
                bv["target"] = target
            buffer_views.append(bv)
            return len(buffer_views) - 1

        def _add_accessor(bv_idx, comp_type, count, acc_type, min_val=None, max_val=None) -> int:
            acc = {"bufferView": bv_idx, "componentType": comp_type, "count": count, "type": acc_type}
            if min_val is not None:
                acc["min"] = min_val
            if max_val is not None:
                acc["max"] = max_val
            accessors.append(acc)
            return len(accessors) - 1

        # Mesh data
        pos_bv = _add_data(vertices.tobytes(), target=34962)
        pos_acc = _add_accessor(pos_bv, 5126, num_verts, "VEC3",
                                vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist())

        norm_bv = _add_data(normals.tobytes(), target=34962)
        norm_acc = _add_accessor(norm_bv, 5126, num_verts, "VEC3")

        idx_bv = _add_data(indices.tobytes(), target=34963)
        idx_acc = _add_accessor(idx_bv, 5125, len(indices), "SCALAR",
                                [int(indices.min())], [int(indices.max())])

        # Skinning data
        joints_bv = _add_data(joints_data.tobytes(), target=34962)
        joints_acc = _add_accessor(joints_bv, 5123, num_verts, "VEC4")  # UNSIGNED_SHORT

        weights_bv = _add_data(weights_data.tobytes(), target=34962)
        weights_acc = _add_accessor(weights_bv, 5126, num_verts, "VEC4")

        # Inverse bind matrices
        ibm_bv = _add_data(ibms.tobytes())
        ibm_acc = _add_accessor(ibm_bv, 5126, NUM_BONES, "MAT4")

        # Animation data — time + per-bone translation + rotation
        time_bv = _add_data(times.tobytes())
        time_acc = _add_accessor(time_bv, 5126, num_frames, "SCALAR",
                                 [float(times[0])], [float(times[-1])])

        anim_channels = []
        anim_samplers = []
        sampler_idx = 0

        for bi in range(NUM_BONES):
            # Translation
            bt = bone_translations[bi]
            bt_bv = _add_data(bt.tobytes())
            bt_acc = _add_accessor(bt_bv, 5126, num_frames, "VEC3",
                                   bt.min(axis=0).tolist(), bt.max(axis=0).tolist())
            anim_samplers.append({"input": time_acc, "output": bt_acc, "interpolation": "LINEAR"})
            # Node index for this bone = bi + 1 (node 0 is mesh)
            anim_channels.append({"sampler": sampler_idx, "target": {"node": bi + 1, "path": "translation"}})
            sampler_idx += 1

            # Rotation
            br = bone_rotations[bi]
            br_bv = _add_data(br.tobytes())
            br_acc = _add_accessor(br_bv, 5126, num_frames, "VEC4",
                                   br.min(axis=0).tolist(), br.max(axis=0).tolist())
            anim_samplers.append({"input": time_acc, "output": br_acc, "interpolation": "LINEAR"})
            anim_channels.append({"sampler": sampler_idx, "target": {"node": bi + 1, "path": "rotation"}})
            sampler_idx += 1

        # ── Build glTF JSON ──────────────────────────────────

        # Nodes: [mesh_node, bone0, bone1, ..., boneN]
        mesh_node = {
            "name": "mesh",
            "mesh": 0,
            "skin": 0,
        }

        bone_nodes = []
        for bi, bone in enumerate(SKELETON_TEMPLATE):
            if bone.parent_idx is None:
                # Root bone: translation is absolute world position
                local_pos = bone_positions[bi]
            else:
                # Child bone: translation is relative to parent bone world position
                local_pos = bone_positions[bi] - bone_positions[bone.parent_idx]

            node = {
                "name": bone.name,
                "translation": local_pos.tolist(),
            }
            # Find children
            children = [i for i, b in enumerate(SKELETON_TEMPLATE) if b.parent_idx == bi]
            if children:
                node["children"] = [c + 1 for c in children]  # +1 because node 0 is mesh
            bone_nodes.append(node)

        nodes = [mesh_node] + bone_nodes

        # Find root bones (no parent) — node indices are bi+1 (node 0 is mesh)
        root_bone_node_indices = [bi + 1 for bi, b in enumerate(SKELETON_TEMPLATE) if b.parent_idx is None]

        # Scene contains: mesh node + root bone nodes
        # The skeleton hierarchy is expressed via children arrays in bone nodes
        gltf = {
            "asset": {"version": "2.0", "generator": "ModelGenerator"},
            "scene": 0,
            "scenes": [{"nodes": [0] + root_bone_node_indices}],
            "nodes": nodes,
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": pos_acc,
                        "NORMAL": norm_acc,
                        "JOINTS_0": joints_acc,
                        "WEIGHTS_0": weights_acc,
                    },
                    "indices": idx_acc,
                }]
            }],
            "skins": [{
                "joints": list(range(1, NUM_BONES + 1)),  # bone node indices
                "inverseBindMatrices": ibm_acc,
                "skeleton": 1,  # root bone node
            }],
            "animations": [{
                "name": anim_name,
                "channels": anim_channels,
                "samplers": anim_samplers,
            }],
            "buffers": [{"byteLength": len(buf)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }

        # Vertex colors if available
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            try:
                colors = (mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255.0)
                color_bv = _add_data(colors.tobytes(), target=34962)
                color_acc = _add_accessor(color_bv, 5126, num_verts, "VEC4")
                gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = color_acc
                gltf["buffers"][0]["byteLength"] = len(buf)
            except Exception:
                pass

        # ── Serialize GLB ────────────────────────────────────
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

    def unload_model(self) -> None:
        pass
