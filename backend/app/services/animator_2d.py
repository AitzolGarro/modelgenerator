"""
2D Animator Service.

Generates sprite-sheet animations from a 2D character model (model.json + part PNGs).

Animation keyframes:
    Each keyframe is a 4-tuple: (time_seconds, translate_x_pixels, translate_y_pixels, rotation_degrees)
    Rotation is applied around the part's pivot point.

Easing: ease-in-out (smoothstep) between adjacent keyframes.

Output:
    - sprite_sheet.png  — horizontal strip of FRAME_SIZE × FRAME_SIZE frames
    - animation.json    — metadata for game engines
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Keyframe type ─────────────────────────────────────────────

class Keyframe(NamedTuple):
    time: float          # seconds
    tx: float            # translate X pixels
    ty: float            # translate Y pixels
    rot: float           # rotation degrees (CCW)


# ── Animation presets ────────────────────────────────────────
#
# Each part has a list of Keyframes.  Parts not listed use zero keyframes.
# time 0.0 and time == duration are the same pose (loop).

_KF = Keyframe  # alias for brevity

ANIMATION_PRESETS: dict[str, dict] = {

    # ── Idle: gentle breathing, slight head tilt ─────────────
    "idle": {
        "duration": 2.0, "fps": 12, "loop": True,
        "parts": {
            "head":      [_KF(0.0, 0, 0, 0), _KF(0.5, 0, -2, -1.5), _KF(1.0, 0, -3, 0),
                          _KF(1.5, 0, -2, 1.5), _KF(2.0, 0, 0, 0)],
            "torso":     [_KF(0.0, 0, 0, 0), _KF(1.0, 0, -2, 0), _KF(2.0, 0, 0, 0)],
            "arm_left":  [_KF(0.0, 0, 0, 2), _KF(1.0, 0, -2, 3), _KF(2.0, 0, 0, 2)],
            "arm_right": [_KF(0.0, 0, 0, -2), _KF(1.0, 0, -2, -3), _KF(2.0, 0, 0, -2)],
            "leg_left":  [_KF(0.0, 0, 0, 0), _KF(1.0, 0, -1, 0), _KF(2.0, 0, 0, 0)],
            "leg_right": [_KF(0.0, 0, 0, 0), _KF(1.0, 0, -1, 0), _KF(2.0, 0, 0, 0)],
        },
    },

    # ── Walk: alternating legs, arm swing, head bob ──────────
    "walk": {
        "duration": 0.8, "fps": 12, "loop": True,
        "parts": {
            "head":      [_KF(0.0, 0, 0, 0), _KF(0.2, 0, -2, 0), _KF(0.4, 0, 0, 0),
                          _KF(0.6, 0, -2, 0), _KF(0.8, 0, 0, 0)],
            "torso":     [_KF(0.0, 0, 0, 2), _KF(0.4, 0, 0, -2), _KF(0.8, 0, 0, 2)],
            "arm_left":  [_KF(0.0, 0, 0, -28), _KF(0.4, 0, 0, 28), _KF(0.8, 0, 0, -28)],
            "arm_right": [_KF(0.0, 0, 0, 28), _KF(0.4, 0, 0, -28), _KF(0.8, 0, 0, 28)],
            "leg_left":  [_KF(0.0, 0, 0, -22), _KF(0.2, 0, -8, -32), _KF(0.4, 0, 0, 22),
                          _KF(0.6, 0, 0, 10), _KF(0.8, 0, 0, -22)],
            "leg_right": [_KF(0.0, 0, 0, 22), _KF(0.2, 0, 0, 10), _KF(0.4, 0, 0, -22),
                          _KF(0.6, 0, -8, -32), _KF(0.8, 0, 0, 22)],
        },
    },

    # ── Run: faster stride, higher leg lift ──────────────────
    "run": {
        "duration": 0.5, "fps": 12, "loop": True,
        "parts": {
            "head":      [_KF(0.0, 0, -2, 0), _KF(0.25, 0, 2, 0), _KF(0.5, 0, -2, 0)],
            "torso":     [_KF(0.0, 0, 0, 5), _KF(0.25, 0, 0, -5), _KF(0.5, 0, 0, 5)],
            "arm_left":  [_KF(0.0, 0, 0, -45), _KF(0.25, 0, 0, 45), _KF(0.5, 0, 0, -45)],
            "arm_right": [_KF(0.0, 0, 0, 45), _KF(0.25, 0, 0, -45), _KF(0.5, 0, 0, 45)],
            "leg_left":  [_KF(0.0, 0, 0, -40), _KF(0.125, 0, -18, -55),
                          _KF(0.25, 0, 0, 40), _KF(0.5, 0, 0, -40)],
            "leg_right": [_KF(0.0, 0, 0, 40), _KF(0.25, 0, 0, -40),
                          _KF(0.375, 0, -18, -55), _KF(0.5, 0, 0, 40)],
        },
    },

    # ── Attack: wind-up, fast swing, follow-through ──────────
    "attack": {
        "duration": 0.9, "fps": 12, "loop": False,
        "parts": {
            "head":      [_KF(0.0, 0, 0, 0), _KF(0.15, 0, 0, -8), _KF(0.35, 0, 0, 8), _KF(0.9, 0, 0, 0)],
            "torso":     [_KF(0.0, 0, 0, 0), _KF(0.15, -8, 0, -12), _KF(0.35, 8, 0, 20), _KF(0.9, 0, 0, 0)],
            # Primary attack arm swings forward aggressively
            "arm_right": [_KF(0.0, 0, 0, 0), _KF(0.15, -12, 0, -70),
                          _KF(0.30, 12, 0, 80), _KF(0.5, 0, 0, 30), _KF(0.9, 0, 0, 0)],
            "arm_left":  [_KF(0.0, 0, 0, 0), _KF(0.15, 8, 0, 30), _KF(0.35, -8, 0, -10), _KF(0.9, 0, 0, 0)],
            "leg_left":  [_KF(0.0, 0, 0, 0), _KF(0.15, -6, 0, -8), _KF(0.9, 0, 0, 0)],
            "leg_right": [_KF(0.0, 0, 0, 0), _KF(0.15, 6, 0, 8), _KF(0.9, 0, 0, 0)],
        },
    },

    # ── Jump: crouch, airborne, land ─────────────────────────
    "jump": {
        "duration": 1.0, "fps": 12, "loop": False,
        "parts": {
            "head":      [_KF(0.0, 0, 0, 0), _KF(0.2, 0, 4, 0), _KF(0.4, 0, -20, 0),
                          _KF(0.7, 0, -18, 0), _KF(0.9, 0, 4, 0), _KF(1.0, 0, 0, 0)],
            "torso":     [_KF(0.0, 0, 0, 0), _KF(0.2, 0, 6, 5), _KF(0.4, 0, -20, -5),
                          _KF(0.7, 0, -18, 0), _KF(0.9, 0, 6, 0), _KF(1.0, 0, 0, 0)],
            "arm_left":  [_KF(0.0, 0, 0, 5), _KF(0.2, 0, 0, 50), _KF(0.4, 0, 0, -40),
                          _KF(0.7, 0, 0, -35), _KF(0.9, 0, 0, 10), _KF(1.0, 0, 0, 5)],
            "arm_right": [_KF(0.0, 0, 0, -5), _KF(0.2, 0, 0, -50), _KF(0.4, 0, 0, 40),
                          _KF(0.7, 0, 0, 35), _KF(0.9, 0, 0, -10), _KF(1.0, 0, 0, -5)],
            "leg_left":  [_KF(0.0, 0, 0, 0), _KF(0.2, 0, 0, 30), _KF(0.4, 0, 0, -10),
                          _KF(0.7, 0, 0, -5), _KF(0.9, 0, 0, 20), _KF(1.0, 0, 0, 0)],
            "leg_right": [_KF(0.0, 0, 0, 0), _KF(0.2, 0, 0, -30), _KF(0.4, 0, 0, 10),
                          _KF(0.7, 0, 0, 5), _KF(0.9, 0, 0, -20), _KF(1.0, 0, 0, 0)],
        },
    },

    # ── Dance: energetic Honkai Star Rail-style pop ──────────
    "dance": {
        "duration": 2.0, "fps": 12, "loop": True,
        "parts": {
            "head":      [_KF(0.0, 0, 0, 0), _KF(0.25, 4, -4, -12), _KF(0.5, 0, 0, 0),
                          _KF(0.75, -4, -4, 12), _KF(1.0, 0, 0, 0), _KF(1.25, 4, -4, -12),
                          _KF(1.5, 0, 0, 0), _KF(1.75, -4, -4, 12), _KF(2.0, 0, 0, 0)],
            "torso":     [_KF(0.0, 0, 0, 0), _KF(0.5, 0, -4, 8), _KF(1.0, 0, 0, -8),
                          _KF(1.5, 0, -4, 8), _KF(2.0, 0, 0, 0)],
            "arm_left":  [_KF(0.0, 0, 0, 10), _KF(0.25, -6, -12, -40), _KF(0.5, 0, 0, 10),
                          _KF(0.75, 6, 0, 60), _KF(1.0, 0, 0, 10), _KF(1.25, -6, -12, -40),
                          _KF(1.5, 0, 0, 10), _KF(1.75, 6, 0, 60), _KF(2.0, 0, 0, 10)],
            "arm_right": [_KF(0.0, 0, 0, -10), _KF(0.25, 6, 0, -60), _KF(0.5, 0, 0, -10),
                          _KF(0.75, -6, -12, 40), _KF(1.0, 0, 0, -10), _KF(1.25, 6, 0, -60),
                          _KF(1.5, 0, 0, -10), _KF(1.75, -6, -12, 40), _KF(2.0, 0, 0, -10)],
            "leg_left":  [_KF(0.0, 0, 0, 0), _KF(0.25, -8, 0, -20), _KF(0.5, 0, 0, 0),
                          _KF(0.75, 8, 0, 20), _KF(1.0, 0, 0, 0), _KF(1.5, 0, 0, 0), _KF(2.0, 0, 0, 0)],
            "leg_right": [_KF(0.0, 0, 0, 0), _KF(0.25, 8, 0, 20), _KF(0.5, 0, 0, 0),
                          _KF(0.75, -8, 0, -20), _KF(1.0, 0, 0, 0), _KF(1.5, 0, 0, 0), _KF(2.0, 0, 0, 0)],
        },
    },

    # ── Wave: friendly greeting ──────────────────────────────
    "wave": {
        "duration": 1.2, "fps": 12, "loop": True,
        "parts": {
            "head":      [_KF(0.0, 0, 0, 0), _KF(0.3, 0, 0, -6), _KF(0.6, 0, 0, 6), _KF(1.2, 0, 0, 0)],
            "torso":     [_KF(0.0, 0, 0, 0), _KF(0.6, 0, 0, -4), _KF(1.2, 0, 0, 0)],
            # Right arm waves back and forth
            "arm_right": [_KF(0.0, 0, 0, -60), _KF(0.3, 0, 0, -90), _KF(0.6, 0, 0, -60),
                          _KF(0.9, 0, 0, -90), _KF(1.2, 0, 0, -60)],
            "arm_left":  [_KF(0.0, 0, 0, 8), _KF(1.2, 0, 0, 8)],
            "leg_left":  [_KF(0.0, 0, 0, 0), _KF(1.2, 0, 0, 0)],
            "leg_right": [_KF(0.0, 0, 0, 0), _KF(1.2, 0, 0, 0)],
        },
    },

    # ── Hurt: recoil and stagger ─────────────────────────────
    "hurt": {
        "duration": 0.7, "fps": 12, "loop": False,
        "parts": {
            "head":      [_KF(0.0, 0, 0, 0), _KF(0.1, -8, -4, 15), _KF(0.25, 6, 0, -8),
                          _KF(0.5, -2, 0, 4), _KF(0.7, 0, 0, 0)],
            "torso":     [_KF(0.0, 0, 0, 0), _KF(0.1, -12, 0, 20), _KF(0.3, 6, 0, -8),
                          _KF(0.7, 0, 0, 0)],
            "arm_left":  [_KF(0.0, 0, 0, 0), _KF(0.1, -8, -4, -40), _KF(0.3, 4, 0, 15),
                          _KF(0.7, 0, 0, 0)],
            "arm_right": [_KF(0.0, 0, 0, 0), _KF(0.1, 8, -4, 40), _KF(0.3, -4, 0, -15),
                          _KF(0.7, 0, 0, 0)],
            "leg_left":  [_KF(0.0, 0, 0, 0), _KF(0.1, -6, 0, -10), _KF(0.7, 0, 0, 0)],
            "leg_right": [_KF(0.0, 0, 0, 0), _KF(0.1, 6, 0, 10), _KF(0.7, 0, 0, 0)],
        },
    },
}

# ── Keyword → animation name mapping ────────────────────────

_KEYWORD_MAP: list[tuple[str, str]] = [
    (r"\b(run|sprint|dash|corriendo|correr)\b", "run"),
    (r"\b(walk|walking|caminar|caminando|march)\b", "walk"),
    (r"\b(attack|swing|slash|punch|hit|golpe|ataque)\b", "attack"),
    (r"\b(jump|leap|hop|salto|saltar)\b", "jump"),
    (r"\b(dance|dancing|baile|bailar|groove)\b", "dance"),
    (r"\b(wave|waving|greeting|saludar)\b", "wave"),
    (r"\b(hurt|pain|damage|hit|da[ñn]o|golpeado)\b", "hurt"),
    (r"\b(idle|stand|standing|breath|rest|esperar)\b", "idle"),
]


def _detect_animation(prompt: str) -> str:
    """Detect animation type from a text prompt. Falls back to 'idle'."""
    lower = prompt.lower()
    for pattern, anim_name in _KEYWORD_MAP:
        if re.search(pattern, lower):
            return anim_name
    return "idle"


# ── Easing ───────────────────────────────────────────────────

def _smoothstep(t: float) -> float:
    """Smoothstep ease-in-out: t must be in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _interpolate(kf_list: list[Keyframe], t: float, loop: bool, duration: float) -> tuple[float, float, float]:
    """Interpolate tx, ty, rot from keyframe list at time t."""
    if not kf_list:
        return 0.0, 0.0, 0.0

    if loop:
        t = t % duration

    # Find surrounding keyframes
    if t <= kf_list[0].time:
        kf = kf_list[0]
        return kf.tx, kf.ty, kf.rot
    if t >= kf_list[-1].time:
        kf = kf_list[-1]
        return kf.tx, kf.ty, kf.rot

    for i in range(len(kf_list) - 1):
        ka, kb = kf_list[i], kf_list[i + 1]
        if ka.time <= t <= kb.time:
            seg_len = kb.time - ka.time
            if seg_len < 1e-9:
                return kb.tx, kb.ty, kb.rot
            raw = (t - ka.time) / seg_len
            s = _smoothstep(raw)
            tx = ka.tx + s * (kb.tx - ka.tx)
            ty = ka.ty + s * (kb.ty - ka.ty)
            rot = ka.rot + s * (kb.rot - ka.rot)
            return tx, ty, rot

    kf = kf_list[-1]
    return kf.tx, kf.ty, kf.rot


# ── Main service ─────────────────────────────────────────────

class Animator2DService:
    """Generates 2D sprite sheet animations from a character 2D model."""

    def animate(
        self,
        model_json: dict,
        parts_dir: Path,
        prompt: str,
        output_dir: Path,
        frame_size: int | None = None,
    ) -> tuple[Path, Path]:
        """Generate a sprite sheet and metadata JSON for the given model + prompt.

        Args:
            model_json: The model dict loaded from model.json.
            parts_dir: Directory containing part PNGs.
            prompt: Text description of the desired animation.
            output_dir: Where to write sprite_sheet.png + animation.json.
            frame_size: Square frame size in pixels. Defaults to settings.SPRITE_SHEET_FRAME_SIZE.

        Returns:
            (sprite_sheet_path, metadata_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_size = frame_size or settings.SPRITE_SHEET_FRAME_SIZE
        fps = settings.SPRITE_SHEET_FPS

        anim_name = _detect_animation(prompt)
        logger.info(f"Animation type detected: {anim_name!r} from prompt: {prompt[:80]!r}")

        preset = ANIMATION_PRESETS.get(anim_name, ANIMATION_PRESETS["idle"])
        duration: float = preset["duration"]
        loop: bool = preset["loop"]
        part_keyframes: dict[str, list[Keyframe]] = preset["parts"]

        n_frames = max(1, int(round(duration * fps)))
        logger.info(f"Rendering {n_frames} frames at {fps}fps ({duration:.2f}s)")

        # Load part images
        parts_data = self._load_parts(model_json, parts_dir)

        # Determine character bounding box in the full image space
        canvas_w: int = model_json["bounds"]["width"]
        canvas_h: int = model_json["bounds"]["height"]

        # Generate frames
        frames: list[Image.Image] = []
        for frame_idx in range(n_frames):
            t = (frame_idx / n_frames) * duration
            frame_img = self._render_frame(
                parts_data=parts_data,
                part_keyframes=part_keyframes,
                t=t,
                loop=loop,
                duration=duration,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                frame_size=frame_size,
            )
            frames.append(frame_img)

        # Assemble sprite sheet
        from app.services.spritesheet_export import SpriteSheetExportService
        exporter = SpriteSheetExportService()
        sprite_path, meta_path = exporter.export(
            frames=frames,
            frame_size=frame_size,
            fps=fps,
            anim_name=anim_name,
            output_dir=output_dir,
        )
        return sprite_path, meta_path

    # ── Internals ──────────────────────────────────────────────

    def _load_parts(self, model_json: dict, parts_dir: Path) -> list[dict]:
        """Load PIL Images for each part, sorted by z_order ascending (draw back-to-front)."""
        parts = []
        for part_info in model_json.get("parts", []):
            img_path = parts_dir / part_info["image"]
            if not img_path.exists():
                logger.warning(f"Part image not found: {img_path}, skipping")
                continue
            img = Image.open(str(img_path)).convert("RGBA")
            parts.append({
                "name": part_info["name"],
                "image": img,
                "pivot": part_info["pivot"],          # [px, py] in part-image space
                "z_order": part_info.get("z_order", 0),
                "bounds": part_info["bounds"],         # {x, y, w, h} in canvas space
            })
        # Sort: lowest z_order drawn first (background)
        parts.sort(key=lambda p: p["z_order"])
        return parts

    def _render_frame(
        self,
        parts_data: list[dict],
        part_keyframes: dict[str, list[Keyframe]],
        t: float,
        loop: bool,
        duration: float,
        canvas_w: int,
        canvas_h: int,
        frame_size: int,
    ) -> Image.Image:
        """Compose one animation frame."""
        frame = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        for part in parts_data:
            name = part["name"]
            kfs = part_keyframes.get(name, [])
            tx, ty, rot = _interpolate(kfs, t, loop, duration)

            # Original part position in canvas
            bounds = part["bounds"]
            bx, by = bounds["x"], bounds["y"]

            # Pivot in canvas space
            pivot_px, pivot_py = part["pivot"]
            pivot_canvas_x = bx + pivot_px
            pivot_canvas_y = by + pivot_py

            # Apply rotation + translation
            part_img = part["image"].copy()
            if abs(rot) > 0.01:
                part_img = _rotate_around_pivot(
                    part_img, rot,
                    pivot_x=pivot_px, pivot_y=pivot_py,
                )

            # Paste with translation offset
            paste_x = int(round(bx + tx))
            paste_y = int(round(by + ty))
            frame.paste(part_img, (paste_x, paste_y), part_img)

        # Scale down to frame_size × frame_size (fit within, keep aspect)
        frame_scaled = _fit_to_square(frame, frame_size)
        return frame_scaled


def _rotate_around_pivot(
    image: Image.Image,
    degrees: float,
    pivot_x: float,
    pivot_y: float,
) -> Image.Image:
    """Rotate an image around a pivot point (in image space).

    PIL's rotate() takes center as (x, y) from top-left.
    expand=False keeps image size, so rotated content can be clipped —
    that's fine for sprite parts that have generous overlap margins.
    """
    w, h = image.size
    # PIL center is (col, row) from top-left
    rotated = image.rotate(
        degrees,                # positive = CCW in PIL
        resample=Image.BICUBIC,
        expand=False,
        center=(pivot_x, pivot_y),
    )
    return rotated


def _fit_to_square(image: Image.Image, size: int) -> Image.Image:
    """Fit image into a square of `size`×`size`, centered, transparent padding."""
    iw, ih = image.size
    scale = min(size / iw, size / ih)
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    out.paste(resized, (off_x, off_y), resized)
    return out
