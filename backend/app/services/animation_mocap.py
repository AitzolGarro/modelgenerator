"""
MocapAnimationService: BVH mocap retargeting for biped meshes.

Uses CMU motion capture clips (6 synthetic BVH files) and retargets
the CMU 31-joint skeleton → our 14-bone biped skeleton.

Non-biped body types fall back to ProceduralAnimationService.
"""

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from app.core.logging import get_logger
from app.services.animation import ProceduralAnimationService
from app.services.animation_utils import (
    BodyType,
    SKELETONS,
    _classify_body_type,
    _SKELETON_FITTERS,
    _compute_weights,
)

try:
    import bvh as bvh_lib
    _BVH_AVAILABLE = True
except ImportError:
    _BVH_AVAILABLE = False

import trimesh

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────

CMU_ACTOR_HEIGHT = 1.80   # meters (180 cm reference height)
LOOP_BLEND_RATIO = 0.10   # blend last 10% of frames toward frame 0

# ── Bone mapping: our 14-bone name → (CMU joint names, operation) ──────────
# Operations:
#   "translate" — extract root XYZ position channels
#   "direct"    — single joint, euler → quat
#   "compose"   — multiple joints, quat multiply chain

BONE_MAP: dict[str, tuple[list[str], str]] = {
    "root":        (["Hips"],                              "translate"),
    "hip":         (["Hips"],                              "direct"),
    "spine":       (["LowerBack"],                         "direct"),
    "chest":       (["Spine1"],                            "direct"),
    "neck":        (["Neck"],                              "direct"),
    "head":        (["Head"],                              "direct"),
    "upper_arm_l": (["LeftShoulder", "LeftArm"],           "compose"),
    "lower_arm_l": (["LeftForeArm"],                       "direct"),
    "upper_arm_r": (["RightShoulder", "RightArm"],         "compose"),
    "lower_arm_r": (["RightForeArm"],                      "direct"),
    "upper_leg_l": (["LeftUpLeg"],                         "direct"),
    "lower_leg_l": (["LeftLeg"],                           "direct"),
    "upper_leg_r": (["RightUpLeg"],                        "direct"),
    "lower_leg_r": (["RightLeg"],                          "direct"),
}

# ── Prompt → clip keyword mapping ──────────────────────────────

CLIP_KEYWORDS: dict[str, list[str]] = {
    "walk":   ["walk", "stroll", "march", "patrol"],
    "run":    ["run", "sprint", "jog", "chase"],
    "idle":   ["idle", "stand", "wait", "rest"],
    "jump":   ["jump", "leap", "hop", "bounce"],
    "attack": ["attack", "fight", "hit", "strike", "punch", "slash",
               "left hand", "right hand"],
    "dance":  ["dance", "celebrate", "groove"],
}

# Directory containing BVH files (relative to this file's parent)
_MOCAP_DATA_DIR = Path(__file__).parent / "mocap_data"


# ── MocapAnimationService ─────────────────────────────────────

class MocapAnimationService(ProceduralAnimationService):
    """
    Animation service using BVH mocap retargeting for biped meshes.
    Falls back to procedural for non-biped body types.
    """

    def __init__(self) -> None:
        super().__init__()
        self._clips: dict = {}  # clip_name → bvh.Bvh object
        if _BVH_AVAILABLE:
            self._load_clips()
        else:
            logger.warning("bvh library not available — mocap disabled, using procedural fallback")

    def load_model(self) -> None:
        logger.info(f"Mocap animation service ready ({len(self._clips)} clips loaded)")

    # ── Clip loading ───────────────────────────────────────────

    def _load_clips(self) -> None:
        """Parse all BVH files in mocap_data/ at init. Skip on parse error."""
        if not _MOCAP_DATA_DIR.is_dir():
            logger.warning(f"mocap_data/ directory not found: {_MOCAP_DATA_DIR}")
            return

        for bvh_path in sorted(_MOCAP_DATA_DIR.glob("*.bvh")):
            clip_name = bvh_path.stem  # e.g. "walk"
            try:
                text = bvh_path.read_text()
                clip = bvh_lib.Bvh(text)
                self._clips[clip_name] = clip
                logger.info(
                    f"Loaded BVH clip '{clip_name}': "
                    f"{clip.nframes} frames @ {clip.frame_time:.4f}s"
                )
            except Exception as exc:
                logger.warning(f"Failed to parse {bvh_path.name}: {exc} — skipping")

    # ── Clip selection ─────────────────────────────────────────

    def _select_clip(self, prompt: str) -> str:
        """Match prompt keywords → clip name. Default: 'idle'."""
        prompt_lower = prompt.lower()
        for clip_name, keywords in CLIP_KEYWORDS.items():
            if any(kw in prompt_lower for kw in keywords):
                if clip_name in self._clips:
                    return clip_name
        return "idle" if "idle" in self._clips else next(iter(self._clips), "idle")

    # ── Euler → quaternion conversion ──────────────────────────

    def _euler_to_quat(self, joint_name: str, clip, frame_idx: int) -> np.ndarray:
        """
        Extract Euler rotation channels for a joint at a given frame,
        detect channel order dynamically, convert to unit quaternion [x,y,z,w].
        """
        channels = clip.joint_channels(joint_name)
        # Filter rotation channels only
        rot_channels = [c for c in channels if "rotation" in c.lower()]
        if not rot_channels:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Build scipy axes string: "Zrotation" → "Z", etc.
        axes = "".join(c[0] for c in rot_channels)  # e.g. "ZXY"

        # Extract values
        values = clip.frame_joint_channels(frame_idx, joint_name, rot_channels)
        angles_deg = np.array(values, dtype=np.float64)

        # Convert via scipy Rotation
        rot = Rotation.from_euler(axes, angles_deg, degrees=True)
        q = rot.as_quat()  # scipy returns [x, y, z, w]

        # Normalize
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0])

        return q.astype(np.float32)

    # ── Per-frame retargeting ──────────────────────────────────

    def _retarget_frame(
        self, clip, frame_idx: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Retarget one BVH frame → dict of bone_name → quaternion,
        plus root translation vector [x, y, z] in CMU cm units.
        """
        bone_quats: dict[str, np.ndarray] = {}

        # Root translation (only from "root" bone mapping)
        root_joints, root_op = BONE_MAP["root"]
        root_joint = root_joints[0]  # "Hips"
        channels = clip.joint_channels(root_joint)
        pos_channels = [c for c in channels if "position" in c.lower()]
        if pos_channels:
            pos_vals = clip.frame_joint_channels(frame_idx, root_joint, pos_channels)
            root_trans = np.array(pos_vals, dtype=np.float32)
        else:
            root_trans = np.zeros(3, dtype=np.float32)

        # Rotation for all 14 bones
        for bone_name, (cmu_joints, operation) in BONE_MAP.items():
            if operation == "translate":
                # Root translation handled above; rotation of root is "hip" bone
                bone_quats[bone_name] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                continue

            if operation == "direct":
                joint = cmu_joints[0]
                q = self._euler_to_quat(joint, clip, frame_idx)

            elif operation == "compose":
                # Multiply quaternions: q(A) * q(B) for each joint in chain
                q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                for joint in cmu_joints:
                    qi = self._euler_to_quat(joint, clip, frame_idx)
                    # scipy-convention multiply: result = q * qi
                    q_r = Rotation.from_quat(q)
                    q_i = Rotation.from_quat(qi)
                    q = (q_r * q_i).as_quat().astype(np.float32)
                # Normalize
                norm = np.linalg.norm(q)
                if norm > 1e-10:
                    q = q / norm

            else:
                q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            bone_quats[bone_name] = q

        return bone_quats, root_trans

    # ── Full clip retargeting ──────────────────────────────────

    def _retarget_clip(
        self,
        clip,
        mesh_height: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Retarget entire BVH clip to produce animation data for our 14-bone skeleton.

        Returns:
            bone_trans_list: list of (nframes, 3) arrays — root translation per bone
                             (only root/hip have non-zero translations)
            bone_rots_list:  list of (nframes, 4) arrays — quaternions [x,y,z,w]
            times:           (nframes,) float32 array of timestamps
        """
        bone_names = [bd.name for bd in SKELETONS[BodyType.BIPED]]
        nframes_raw = clip.nframes
        frame_time = float(clip.frame_time)

        # Drop last frame (CMU duplicate convention)
        nframes = max(1, nframes_raw - 1)

        # Collect per-frame data
        all_quats: list[dict[str, np.ndarray]] = []
        all_root_trans: list[np.ndarray] = []

        for fi in range(nframes):
            bone_quats, root_trans = self._retarget_frame(clip, fi)
            all_quats.append(bone_quats)
            all_root_trans.append(root_trans)

        # Scale root translation: relative displacement (frame[i] - frame[0])
        # Scale from CMU cm → mesh units using mesh_height / cmu_actor_height_m
        cmu_root_0 = all_root_trans[0].copy()
        scale = mesh_height / CMU_ACTOR_HEIGHT  # mesh_height is in mesh units

        # CMU uses cm, our mesh uses scene units (typically meters).
        # CMU actor ~180cm → if mesh_height ≈ 1.8 (in meters), scale ≈ 1.0/100 effectively
        # But mesh_height is already in mesh-scene units — we need to normalize CMU units too.
        # CMU positions are in cm (typically 60–130cm Y). mesh_height in mesh bounding box units.
        # Example: mesh height = 1.8 → CMU height 180cm → scale = 1.8/180 = 0.01
        # scale = mesh_height / (CMU_ACTOR_HEIGHT * 100)  # 100 = cm→m conversion
        scale_trans = mesh_height / (CMU_ACTOR_HEIGHT * 100.0)

        # Build per-bone arrays
        bone_rots_list = []
        bone_trans_list = []

        for bone_name in bone_names:
            rots = np.zeros((nframes, 4), dtype=np.float32)
            trans = np.zeros((nframes, 3), dtype=np.float32)

            for fi in range(nframes):
                rots[fi] = all_quats[fi].get(bone_name, np.array([0,0,0,1], dtype=np.float32))

                if bone_name == "root":
                    rel = all_root_trans[fi] - cmu_root_0
                    # CMU: X=right, Y=up, Z=forward — glTF: X=right, Y=up, Z=back
                    # Negate Z to flip from CMU to glTF forward convention
                    trans[fi] = np.array([
                        rel[0] * scale_trans,
                        rel[1] * scale_trans,
                        -rel[2] * scale_trans,
                    ], dtype=np.float32)

            # Loop blend: SLERP last 10% toward frame 0 quaternion
            blend_start = int(nframes * (1.0 - LOOP_BLEND_RATIO))
            blend_start = max(0, min(blend_start, nframes - 1))

            if blend_start < nframes - 1:
                q0 = Rotation.from_quat(rots[0])
                key_times = np.array([0.0, 1.0])
                key_rots = Rotation.concatenate([
                    Rotation.from_quat(rots[blend_start]),
                    q0,
                ])
                slerp = Slerp(key_times, key_rots)
                blend_t = np.linspace(0.0, 1.0, nframes - blend_start)
                blended = slerp(blend_t)
                rots[blend_start:] = blended.as_quat().astype(np.float32)

            bone_rots_list.append(rots)
            bone_trans_list.append(trans)

        times = np.arange(nframes, dtype=np.float32) * frame_time
        return bone_trans_list, bone_rots_list, times

    # ── Main animate() override ────────────────────────────────

    def animate(
        self,
        glb_path: Path,
        prompt: str,
        output_path: Path,
        duration: float = 3.0,
        fps: int = 30,
    ) -> Path:
        """
        Override: use mocap retargeting for biped, procedural for all other types.
        """
        # If no clips loaded, fall back to procedural
        if not self._clips:
            logger.warning("No BVH clips available — using procedural fallback")
            return super().animate(glb_path, prompt, output_path, duration, fps)

        logger.info(f"MocapAnimationService.animate: {glb_path.name} — '{prompt[:60]}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load mesh
        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No meshes found in GLB")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            raise ValueError(f"Unsupported mesh type: {type(scene)}")

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # 2. Classify body type
        body_type = _classify_body_type(mesh.vertices, prompt)
        logger.info(f"Detected body type: {body_type}")

        # 3. Non-biped → procedural fallback
        if body_type != BodyType.BIPED:
            logger.info(f"Non-biped ({body_type}) — delegating to procedural service")
            return super().animate(glb_path, prompt, output_path, duration, fps)

        # 4. Biped mocap path
        clip_name = self._select_clip(prompt)
        logger.info(f"Selected mocap clip: {clip_name}")

        clip = self._clips[clip_name]

        # 5. Fit biped skeleton to mesh
        bone_hierarchy = SKELETONS[BodyType.BIPED]
        bone_positions, segment_info = _SKELETON_FITTERS[BodyType.BIPED](mesh.vertices)

        bmin, bmax = mesh.bounds
        bsize = bmax - bmin
        model_height = float(bsize[1])
        logger.info(f"Model height: {model_height:.4f} scene units")

        # 6. Compute skinning weights
        logger.info("Computing skinning weights...")
        weights = _compute_weights(mesh.vertices, bone_positions, segment_info, bmin, bsize)

        # 7. Compute local (parent-relative) bone positions for bind pose
        local_positions = np.zeros_like(bone_positions)
        for bi, bd in enumerate(bone_hierarchy):
            if bd.parent_idx is not None:
                local_positions[bi] = bone_positions[bi] - bone_positions[bd.parent_idx]
            else:
                local_positions[bi] = bone_positions[bi]

        # 8. Retarget BVH clip
        logger.info(f"Retargeting '{clip_name}' clip ({clip.nframes} frames)...")
        bone_trans_list, bone_rots_list, times = self._retarget_clip(clip, model_height)

        # 9. Add bind-pose local positions to retargeted translations
        # (glTF animation REPLACES node translation — bind pose must be in keyframes)
        all_trans = []
        for bi, bd in enumerate(bone_hierarchy):
            bt_delta = bone_trans_list[bi]
            bt = bt_delta + local_positions[bi].astype(np.float32)
            all_trans.append(bt)

        # 10. Build GLB
        logger.info("Building animated GLB from mocap data...")
        glb = self._build_gltf(
            mesh, bone_hierarchy, bone_positions, weights,
            times, all_trans, bone_rots_list, clip_name
        )

        output_path.write_bytes(glb)
        logger.info(f"Mocap animated GLB: {output_path} ({len(glb)} bytes)")
        return output_path
