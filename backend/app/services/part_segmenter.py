"""
Part Segmenter Service.

Segments a 2D RGBA character image into articulated body parts using
Y-zone slicing + X-zone splitting (no SAM required).

Output: a "2D model" dict saved as model.json in output_dir, alongside
individual part PNGs.

Model JSON schema:
    {
        "parts": [
            {
                "name": str,                   # e.g. "head", "hair", "torso"
                "image": str,                  # relative filename, e.g. "head.png"
                "pivot": [float, float],       # pivot x,y in pixels (in part image space)
                "z_order": int,                # higher = drawn on top
                "bounds": {                    # crop rect in the original image
                    "x": int, "y": int,
                    "w": int, "h": int
                },
                "z_depth": float,              # depth layer: >0 = closer to camera, <0 = farther
                "secondary_motion": bool,      # True = spring physics delay (hair, capes, etc.)
                "secondary_amplitude": float   # how much secondary motion affects this part (0–1)
            },
            ...
        ],
        "bounds": {"width": int, "height": int},
        "full_image": "character.png"
    }

z_depth conventions:
    head:       0.3   (slightly forward)
    hair:       0.5   (frontmost — reacts to motion most)
    torso:      0.0   (center reference)
    arm_left:   0.2
    arm_right:  0.2
    leg_left:  -0.1   (slightly behind)
    leg_right: -0.1
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Part definitions ──────────────────────────────────────────
#
# y_range and x_range are fractions of the bounding box of the
# non-transparent character pixels (not the full canvas).
# pivot is expressed as a fraction within the PART CROP rectangle:
#   (0, 0) = top-left of the crop
#
# z_order:         which part is drawn on top when compositing frames.
# z_depth:         simulated 3D depth (>0 = closer to camera, <0 = farther).
#                  Used by the animator for parallax offset.
# secondary_motion: if True, the animator applies spring physics delay.
# secondary_amplitude: how much secondary motion affects this part (0–1).

_PART_DEFINITIONS = [
    # ── Hair: top 35% of the head zone — frontmost, reacts to motion ──
    {
        "name": "hair",
        "y_range": (0.00, 0.15),      # top ~35% of head zone (0.00–0.22 → 0.00–0.15)
        "x_range": (0.15, 0.85),
        "pivot_frac": (0.50, 0.90),   # pivot near base of hair (connects to head)
        "z_order": 6,                  # drawn above head
        "overlap": 0.03,
        "z_depth": 0.5,
        "secondary_motion": True,
        "secondary_amplitude": 0.6,
    },
    # ── Face / head: bottom 65% of head zone ──────────────────────────
    {
        "name": "head",
        "y_range": (0.07, 0.22),      # overlaps slightly with hair for seamless compositing
        "x_range": (0.20, 0.80),
        "pivot_frac": (0.50, 0.90),   # near the neck, bottom of head crop
        "z_order": 5,
        "overlap": 0.05,
        "z_depth": 0.3,
        "secondary_motion": False,
        "secondary_amplitude": 0.0,
    },
    {
        "name": "torso",
        "y_range": (0.17, 0.55),
        "x_range": (0.25, 0.75),
        "pivot_frac": (0.50, 0.10),   # near the shoulders, top of torso
        "z_order": 3,
        "overlap": 0.08,
        "z_depth": 0.0,
        "secondary_motion": False,
        "secondary_amplitude": 0.0,
    },
    {
        "name": "arm_left",
        "y_range": (0.17, 0.60),
        "x_range": (0.00, 0.35),
        "pivot_frac": (0.85, 0.08),   # near shoulder attachment on the right edge of left arm
        "z_order": 4,
        "overlap": 0.08,
        "z_depth": 0.2,
        "secondary_motion": False,
        "secondary_amplitude": 0.0,
    },
    {
        "name": "arm_right",
        "y_range": (0.17, 0.60),
        "x_range": (0.65, 1.00),
        "pivot_frac": (0.15, 0.08),   # near shoulder attachment on the left edge of right arm
        "z_order": 4,
        "overlap": 0.08,
        "z_depth": 0.2,
        "secondary_motion": False,
        "secondary_amplitude": 0.0,
    },
    {
        "name": "leg_left",
        "y_range": (0.50, 1.00),
        "x_range": (0.20, 0.52),
        "pivot_frac": (0.50, 0.04),   # near hip, top of leg
        "z_order": 2,
        "overlap": 0.06,
        "z_depth": -0.1,
        "secondary_motion": False,
        "secondary_amplitude": 0.0,
    },
    {
        "name": "leg_right",
        "y_range": (0.50, 1.00),
        "x_range": (0.48, 0.80),
        "pivot_frac": (0.50, 0.04),   # near hip, top of leg
        "z_order": 2,
        "overlap": 0.06,
        "z_depth": -0.1,
        "secondary_motion": False,
        "secondary_amplitude": 0.0,
    },
]


class PartSegmenterService:
    """Segments a 2D RGBA character image into articulated body parts."""

    def segment(self, image: Image.Image, output_dir: Path) -> dict:
        """Segment the character into parts and write files to output_dir.

        Args:
            image: RGBA PIL Image of the character (transparent background).
            output_dir: Directory where part PNGs and model.json are written.

        Returns:
            The model dict (also written as model.json).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Work with numpy for speed
        rgba_arr = np.array(image.convert("RGBA"), dtype=np.uint8)
        h, w = rgba_arr.shape[:2]
        alpha = rgba_arr[:, :, 3]

        # Find bounding box of non-transparent pixels
        rows = np.any(alpha > 16, axis=1)
        cols = np.any(alpha > 16, axis=0)
        if not rows.any():
            raise ValueError("Character image has no non-transparent pixels")

        y_min, y_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
        x_min, x_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
        char_h = y_max - y_min + 1
        char_w = x_max - x_min + 1

        logger.info(
            f"Character bounding box: ({x_min},{y_min})–({x_max},{y_max}) "
            f"= {char_w}×{char_h} on {w}×{h} canvas"
        )

        # Save full character image
        full_path = output_dir / "character.png"
        image.save(str(full_path))
        logger.info(f"Full character saved: {full_path}")

        parts = []
        for part_def in _PART_DEFINITIONS:
            part_info = self._extract_part(
                rgba_arr=rgba_arr,
                part_def=part_def,
                char_bounds=(x_min, y_min, char_w, char_h),
                output_dir=output_dir,
            )
            if part_info is not None:
                parts.append(part_info)
                logger.info(f"  Part {part_info['name']}: saved as {part_info['image']}")

        model = {
            "parts": parts,
            "bounds": {"width": w, "height": h},
            "full_image": "character.png",
        }

        model_json_path = output_dir / "model.json"
        model_json_path.write_text(json.dumps(model, indent=2))
        logger.info(f"Model JSON written: {model_json_path}")

        return model

    # ── Internal helpers ──────────────────────────────────────

    def _extract_part(
        self,
        rgba_arr: np.ndarray,
        part_def: dict,
        char_bounds: tuple[int, int, int, int],
        output_dir: Path,
    ) -> dict | None:
        """Extract one body-part crop from the character array."""
        x_min, y_min, char_w, char_h = char_bounds
        canvas_h, canvas_w = rgba_arr.shape[:2]

        y_frac0, y_frac1 = part_def["y_range"]
        x_frac0, x_frac1 = part_def["x_range"]
        overlap = part_def.get("overlap", 0.05)

        # Map fractions to pixel coords in full canvas space
        # Add overlap to avoid harsh seams between parts
        y0 = max(0, y_min + int(y_frac0 * char_h) - int(overlap * char_h))
        y1 = min(canvas_h, y_min + int(y_frac1 * char_h) + int(overlap * char_h))
        x0 = max(0, x_min + int(x_frac0 * char_w))
        x1 = min(canvas_w, x_min + int(x_frac1 * char_w))

        if y1 <= y0 or x1 <= x0:
            logger.warning(f"Part {part_def['name']}: empty crop region, skipping")
            return None

        crop = rgba_arr[y0:y1, x0:x1].copy()
        crop_h, crop_w = crop.shape[:2]

        # Check if the crop has enough non-transparent pixels (at least 5 % of crop area)
        alpha_crop = crop[:, :, 3]
        non_transparent = int(np.sum(alpha_crop > 16))
        if non_transparent < crop_h * crop_w * 0.05:
            logger.warning(
                f"Part {part_def['name']}: too few non-transparent pixels "
                f"({non_transparent}/{crop_h * crop_w}), skipping"
            )
            return None

        # Save part PNG
        part_img = Image.fromarray(crop, "RGBA")
        filename = f"{part_def['name']}.png"
        part_path = output_dir / filename
        part_img.save(str(part_path))

        # Calculate pivot in part-image pixel space
        pf_x, pf_y = part_def["pivot_frac"]
        pivot_x = float(pf_x * crop_w)
        pivot_y = float(pf_y * crop_h)

        return {
            "name": part_def["name"],
            "image": filename,
            "pivot": [round(pivot_x, 2), round(pivot_y, 2)],
            "z_order": part_def["z_order"],
            "bounds": {"x": x0, "y": y0, "w": crop_w, "h": crop_h},
            # Depth / physics fields for the 3D-depth animation system
            "z_depth": part_def.get("z_depth", 0.0),
            "secondary_motion": part_def.get("secondary_motion", False),
            "secondary_amplitude": part_def.get("secondary_amplitude", 0.0),
        }
