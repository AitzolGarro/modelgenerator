"""
Sprite Sheet Export Service.

Combines a list of PIL Image frames into a single horizontal sprite-sheet PNG,
and writes a companion JSON metadata file understood by 2D game engines.

Metadata schema:
    {
        "name": str,
        "fps": int,
        "frame_count": int,
        "frame_width": int,
        "frame_height": int,
        "loop": bool,
        "frames": [
            {"index": 0, "x": 0, "y": 0, "w": 512, "h": 512},
            ...
        ]
    }

The sprite sheet is a single-row horizontal strip:
    [frame0][frame1][frame2]...
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


class SpriteSheetExportService:
    """Exports animation frames as a sprite sheet + JSON metadata."""

    def export(
        self,
        frames: list[Image.Image],
        frame_size: int,
        fps: int,
        anim_name: str,
        output_dir: Path,
        loop: bool = True,
    ) -> tuple[Path, Path]:
        """Create sprite sheet PNG + metadata JSON.

        Args:
            frames: List of RGBA PIL Images, each frame_size × frame_size.
            frame_size: Width and height of each frame square.
            fps: Frames per second for playback.
            anim_name: Animation name (written into metadata).
            output_dir: Directory where files are written.
            loop: Whether the animation loops.

        Returns:
            (spritesheet_path, metadata_path)
        """
        if not frames:
            raise ValueError("No frames to export")

        output_dir.mkdir(parents=True, exist_ok=True)

        n = len(frames)
        sheet_w = frame_size * n
        sheet_h = frame_size

        # Build sprite sheet
        sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            # Ensure frame is the right size
            if frame.size != (frame_size, frame_size):
                frame = frame.resize((frame_size, frame_size), Image.LANCZOS)
            sheet.paste(frame, (i * frame_size, 0), frame if frame.mode == "RGBA" else None)

        sprite_path = output_dir / "sprite_sheet.png"
        sheet.save(str(sprite_path), "PNG")
        logger.info(f"Sprite sheet saved: {sprite_path} ({sheet_w}×{sheet_h}, {n} frames)")

        # Build metadata
        frame_meta = [
            {
                "index": i,
                "x": i * frame_size,
                "y": 0,
                "w": frame_size,
                "h": frame_size,
            }
            for i in range(n)
        ]

        metadata = {
            "name": anim_name,
            "fps": fps,
            "frame_count": n,
            "frame_width": frame_size,
            "frame_height": frame_size,
            "loop": loop,
            "frames": frame_meta,
        }

        meta_path = output_dir / "animation.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Animation metadata saved: {meta_path}")

        return sprite_path, meta_path
