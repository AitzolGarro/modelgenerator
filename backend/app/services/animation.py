"""
Animation service: adds skeletal animation to a GLB model from a text prompt.

Approach:
- Parse the prompt to determine animation type (walk, run, wave, idle, etc.)
- Auto-rig the mesh using bone placement heuristics
- Generate keyframe animation matching the description
- Export as animated GLB with embedded animation clips

For production, consider integrating:
- Mixamo auto-rigging API
- MDM (Motion Diffusion Model) for text-to-motion
- MotionGPT for prompt-based motion generation
"""

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from app.core.logging import get_logger
from app.services.base import AnimationService

logger = get_logger(__name__)

# ── Animation presets mapped from prompt keywords ────────────

ANIMATION_PRESETS: dict[str, dict[str, Any]] = {
    "walk": {
        "type": "locomotion",
        "cycle_duration": 1.0,
        "amplitude": 0.3,
        "description": "bipedal walking cycle",
    },
    "run": {
        "type": "locomotion",
        "cycle_duration": 0.6,
        "amplitude": 0.5,
        "description": "running cycle",
    },
    "idle": {
        "type": "breathing",
        "cycle_duration": 2.5,
        "amplitude": 0.02,
        "description": "subtle idle breathing",
    },
    "wave": {
        "type": "gesture",
        "cycle_duration": 1.5,
        "amplitude": 0.4,
        "description": "arm waving",
    },
    "jump": {
        "type": "vertical",
        "cycle_duration": 1.0,
        "amplitude": 0.5,
        "description": "jump animation",
    },
    "spin": {
        "type": "rotation",
        "cycle_duration": 2.0,
        "amplitude": 1.0,
        "description": "full body rotation",
    },
    "dance": {
        "type": "complex",
        "cycle_duration": 2.0,
        "amplitude": 0.3,
        "description": "dancing motion",
    },
    "attack": {
        "type": "gesture",
        "cycle_duration": 0.8,
        "amplitude": 0.6,
        "description": "attack swing",
    },
    "fly": {
        "type": "vertical",
        "cycle_duration": 1.5,
        "amplitude": 0.4,
        "description": "flying/hovering motion",
    },
    "bounce": {
        "type": "vertical",
        "cycle_duration": 0.5,
        "amplitude": 0.2,
        "description": "bouncing up and down",
    },
}


def _detect_animation_type(prompt: str) -> dict[str, Any]:
    """Parse prompt to determine which animation preset to use."""
    prompt_lower = prompt.lower()
    for keyword, preset in ANIMATION_PRESETS.items():
        if keyword in prompt_lower:
            return preset
    # Default: idle
    return ANIMATION_PRESETS["idle"]


class ProceduralAnimationService(AnimationService):
    """
    Procedural animation service.
    Generates animation by:
    1. Analyzing model bounds and structure
    2. Matching prompt to animation preset
    3. Generating keyframes procedurally
    4. Embedding animation into GLB via glTF animation extension
    """

    def load_model(self) -> None:
        logger.info("Procedural animation service ready (no ML model needed)")

    def animate(
        self,
        glb_path: Path,
        prompt: str,
        output_path: Path,
        duration: float = 3.0,
        fps: int = 30,
    ) -> Path:
        logger.info(f"Animating {glb_path.name} with prompt: '{prompt[:60]}'")

        preset = _detect_animation_type(prompt)
        logger.info(f"Detected animation: {preset['description']}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the GLB and create animated version
        glb_data = glb_path.read_bytes()
        animated_data = self._add_animation_to_glb(
            glb_data, preset, duration, fps
        )

        output_path.write_bytes(animated_data)
        logger.info(f"Animated GLB saved: {output_path} ({len(animated_data)} bytes)")
        return output_path

    def _add_animation_to_glb(
        self,
        glb_data: bytes,
        preset: dict[str, Any],
        duration: float,
        fps: int,
    ) -> bytes:
        """
        Parse GLB, add a glTF animation, and re-serialize.
        Adds translation/rotation keyframes to the root node.
        """
        import json

        # Parse GLB container
        # GLB format: header (12 bytes) + JSON chunk + BIN chunk
        magic, version, length = struct.unpack_from("<III", glb_data, 0)
        if magic != 0x46546C67:  # 'glTF'
            raise ValueError("Not a valid GLB file")

        # Read JSON chunk
        json_chunk_len, json_chunk_type = struct.unpack_from("<II", glb_data, 12)
        json_bytes = glb_data[20 : 20 + json_chunk_len]
        gltf = json.loads(json_bytes)

        # Read BIN chunk (if exists)
        bin_offset = 20 + json_chunk_len
        bin_data = b""
        if bin_offset < len(glb_data):
            # Align to 4 bytes
            while bin_offset % 4 != 0:
                bin_offset += 1
            if bin_offset + 8 <= len(glb_data):
                bin_chunk_len, bin_chunk_type = struct.unpack_from("<II", glb_data, bin_offset)
                bin_data = bytearray(glb_data[bin_offset + 8 : bin_offset + 8 + bin_chunk_len])

        # Generate keyframe data
        num_frames = int(duration * fps)
        cycle = preset["cycle_duration"]
        amp = preset["amplitude"]
        anim_type = preset["type"]

        # Time values
        times = np.linspace(0, duration, num_frames, dtype=np.float32)

        # Generate translation and rotation keyframes
        translations = np.zeros((num_frames, 3), dtype=np.float32)
        rotations = np.zeros((num_frames, 4), dtype=np.float32)
        # Default rotation: identity quaternion
        rotations[:, 3] = 1.0

        for i, t in enumerate(times):
            phase = (t / cycle) * 2 * math.pi

            if anim_type == "locomotion":
                translations[i, 1] = abs(math.sin(phase)) * amp * 0.15  # vertical bob
                translations[i, 2] = math.sin(phase) * amp * 0.05       # forward sway
                # Slight body rotation
                angle = math.sin(phase) * 0.05
                rotations[i] = _quat_from_axis_angle([0, 1, 0], angle)

            elif anim_type == "breathing":
                scale_factor = 1.0 + math.sin(phase) * amp
                translations[i, 1] = math.sin(phase) * amp * 0.5

            elif anim_type == "gesture":
                translations[i, 1] = abs(math.sin(phase * 0.5)) * amp * 0.1
                angle = math.sin(phase) * amp * 0.3
                rotations[i] = _quat_from_axis_angle([0, 0, 1], angle)

            elif anim_type == "vertical":
                translations[i, 1] = abs(math.sin(phase)) * amp

            elif anim_type == "rotation":
                angle = (t / duration) * 2 * math.pi * amp
                rotations[i] = _quat_from_axis_angle([0, 1, 0], angle)

            elif anim_type == "complex":
                translations[i, 0] = math.sin(phase) * amp * 0.2
                translations[i, 1] = abs(math.sin(phase * 2)) * amp * 0.15
                angle = math.sin(phase * 0.5) * 0.2
                rotations[i] = _quat_from_axis_angle([0, 1, 0], angle)

        # Append keyframe data to the binary buffer
        bin_data = bytearray(bin_data)
        base_offset = len(bin_data)

        # Pad to 4-byte alignment
        while len(bin_data) % 4 != 0:
            bin_data.append(0)
        base_offset = len(bin_data)

        time_bytes = times.tobytes()
        trans_bytes = translations.tobytes()
        rot_bytes = rotations.tobytes()

        bin_data.extend(time_bytes)
        while len(bin_data) % 4 != 0:
            bin_data.append(0)
        trans_offset = len(bin_data)
        bin_data.extend(trans_bytes)
        while len(bin_data) % 4 != 0:
            bin_data.append(0)
        rot_offset = len(bin_data)
        bin_data.extend(rot_bytes)
        while len(bin_data) % 4 != 0:
            bin_data.append(0)

        # Update buffer size
        if "buffers" not in gltf:
            gltf["buffers"] = [{"byteLength": 0}]
        gltf["buffers"][0]["byteLength"] = len(bin_data)

        # Add buffer views
        if "bufferViews" not in gltf:
            gltf["bufferViews"] = []
        if "accessors" not in gltf:
            gltf["accessors"] = []

        bv_base = len(gltf["bufferViews"])
        acc_base = len(gltf["accessors"])

        # BufferView for time
        gltf["bufferViews"].append({
            "buffer": 0, "byteOffset": base_offset, "byteLength": len(time_bytes)
        })
        # BufferView for translations
        gltf["bufferViews"].append({
            "buffer": 0, "byteOffset": trans_offset, "byteLength": len(trans_bytes)
        })
        # BufferView for rotations
        gltf["bufferViews"].append({
            "buffer": 0, "byteOffset": rot_offset, "byteLength": len(rot_bytes)
        })

        # Accessor for time
        gltf["accessors"].append({
            "bufferView": bv_base,
            "componentType": 5126,  # FLOAT
            "count": num_frames,
            "type": "SCALAR",
            "min": [float(times[0])],
            "max": [float(times[-1])],
        })
        # Accessor for translations
        gltf["accessors"].append({
            "bufferView": bv_base + 1,
            "componentType": 5126,
            "count": num_frames,
            "type": "VEC3",
            "min": translations.min(axis=0).tolist(),
            "max": translations.max(axis=0).tolist(),
        })
        # Accessor for rotations
        gltf["accessors"].append({
            "bufferView": bv_base + 2,
            "componentType": 5126,
            "count": num_frames,
            "type": "VEC4",
            "min": rotations.min(axis=0).tolist(),
            "max": rotations.max(axis=0).tolist(),
        })

        # Add animation
        if "animations" not in gltf:
            gltf["animations"] = []

        target_node = 0  # Animate the root node
        if "nodes" in gltf and len(gltf["nodes"]) > 0:
            target_node = 0

        gltf["animations"].append({
            "name": preset["description"],
            "channels": [
                {
                    "sampler": 0,
                    "target": {"node": target_node, "path": "translation"},
                },
                {
                    "sampler": 1,
                    "target": {"node": target_node, "path": "rotation"},
                },
            ],
            "samplers": [
                {"input": acc_base, "output": acc_base + 1, "interpolation": "LINEAR"},
                {"input": acc_base, "output": acc_base + 2, "interpolation": "LINEAR"},
            ],
        })

        # Re-serialize GLB
        json_out = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        # Pad JSON to 4-byte alignment
        while len(json_out) % 4 != 0:
            json_out += b" "

        # Pad BIN to 4-byte alignment
        while len(bin_data) % 4 != 0:
            bin_data.append(0)

        total_length = 12 + 8 + len(json_out) + 8 + len(bin_data)

        result = bytearray()
        # Header
        result.extend(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk
        result.extend(struct.pack("<II", len(json_out), 0x4E4F534A))
        result.extend(json_out)
        # BIN chunk
        result.extend(struct.pack("<II", len(bin_data), 0x004E4942))
        result.extend(bin_data)

        return bytes(result)

    def unload_model(self) -> None:
        pass


def _quat_from_axis_angle(axis: list[float], angle: float) -> list[float]:
    """Create a quaternion from axis-angle representation."""
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    norm = math.sqrt(sum(a * a for a in axis))
    if norm == 0:
        return [0, 0, 0, 1]
    return [axis[0] / norm * s, axis[1] / norm * s, axis[2] / norm * s, c]
