"""
2D Character Generation Service.

Generates a full-body 2D character from a text prompt using SDXL,
with style presets (anime, pixel_art, cartoon, realistic, chibi, comic).
Background is removed with rembg so the result is a transparent-background RGBA PNG.
"""

from __future__ import annotations

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import TextToImageService

logger = get_logger(__name__)
settings = get_settings()


class Character2DService:
    """Generates a 2D character from a text prompt with style presets.

    The returned image is RGBA with a transparent background, sized
    CHARACTER_2D_WIDTH × CHARACTER_2D_HEIGHT (default 1024 × 1536).
    """

    STYLE_PRESETS: dict[str, dict[str, str]] = {
        "anime": {
            "suffix": (
                "anime style, cel shaded, clean lines, vibrant colors, "
                "full body character, front view, white background, "
                "character design sheet, Honkai Star Rail style"
            ),
            "negative": "3d render, realistic, photo, blurry, deformed",
        },
        "pixel_art": {
            "suffix": (
                "pixel art style, 32-bit, retro game character, "
                "front view, transparent background"
            ),
            "negative": "3d, realistic, blurry, high resolution photo",
        },
        "cartoon": {
            "suffix": (
                "cartoon style, bold outlines, flat colors, character design, "
                "full body, front view, white background"
            ),
            "negative": "3d render, photo, realistic, anime",
        },
        "realistic": {
            "suffix": (
                "digital painting, detailed character art, full body, "
                "front view, studio lighting, white background"
            ),
            "negative": "3d render, anime, cartoon, pixel art",
        },
        "chibi": {
            "suffix": (
                "chibi style, cute, big head, small body, anime, "
                "full body, front view, white background"
            ),
            "negative": "realistic, 3d, scary, detailed anatomy",
        },
        "comic": {
            "suffix": (
                "comic book style, Marvel/DC style, bold inking, "
                "dynamic pose, full body, character design sheet"
            ),
            "negative": "3d, photo, anime, pixel art",
        },
    }

    def __init__(self, text_to_image: TextToImageService) -> None:
        self.text_to_image = text_to_image

    def generate(
        self,
        prompt: str,
        style: str = "anime",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        """Generate a full-body 2D character and remove its background.

        Returns an RGBA PIL Image with transparent background.
        """
        preset = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["anime"])

        full_prompt = f"{prompt}, {preset['suffix']}"
        combined_negative = preset["negative"]
        if negative_prompt:
            combined_negative = f"{negative_prompt}, {combined_negative}"

        logger.info(
            f"Generating 2D character: style={style!r} "
            f"({settings.CHARACTER_2D_WIDTH}×{settings.CHARACTER_2D_HEIGHT})"
        )

        image = self.text_to_image.generate(
            prompt=full_prompt,
            negative_prompt=combined_negative,
            width=settings.CHARACTER_2D_WIDTH,
            height=settings.CHARACTER_2D_HEIGHT,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # Remove background — produces RGBA with transparent background
        image_rgba = self._remove_background(image)
        logger.info("Background removed from 2D character image")
        return image_rgba

    # ── Internal helpers ──────────────────────────────────────

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background using rembg (U²-Net).

        Falls back to naive white-pixel alpha mask if rembg is unavailable.
        """
        try:
            from rembg import remove as rembg_remove  # type: ignore

            # rembg accepts and returns PIL Images
            rgba = rembg_remove(image)
            if isinstance(rgba, Image.Image):
                return rgba.convert("RGBA")
            # Sometimes returns bytes
            import io
            return Image.open(io.BytesIO(rgba)).convert("RGBA")

        except Exception as exc:
            logger.warning(
                f"rembg unavailable ({exc}), falling back to white-pixel alpha mask"
            )
            return self._naive_white_removal(image)

    @staticmethod
    def _naive_white_removal(image: Image.Image) -> Image.Image:
        """Naive background removal: make near-white pixels transparent."""
        import numpy as np

        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)

        # Pixels that are "mostly white" (each channel > 240)
        mask = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)

        rgba = np.concatenate(
            [arr, np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)],
            axis=2,
        )
        rgba[mask, 3] = 0  # fully transparent

        return Image.fromarray(rgba, "RGBA")
