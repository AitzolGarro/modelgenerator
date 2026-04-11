"""
Body Layer Service.

Generates an inpainted body base layer for 2D animation using SDXL inpainting.

The service builds a union mask from the alpha channels of moving parts
(arms, legs, hair), runs StableDiffusionXLInpaintPipeline.from_pipe() sharing
the already-loaded SDXL UNet, and saves body_base.png in the parts directory.

The animator draws this layer first on every frame, eliminating transparent gaps
when parts move.

Error handling:
    - OOM during inpainting: log warning, save original image as fallback body_base.png
    - Empty mask (no moving parts): skip generation entirely, do not write file
    - from_pipe() failure: log warning, use Approach 5 fallback (original image)
    - Generic exception from pipe: log warning, do NOT write file, do NOT set key
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PIL import Image, ImageFilter

from app.core.logging import get_logger
from app.services.base import TextToImageService
from app.services.character_2d import Character2DService

logger = get_logger(__name__)


class BodyLayerService:
    """Generates an inpainted body base layer for 2D animation."""

    # Moving parts whose alpha channels form the inpainting mask.
    # Head and torso are excluded (they are the stable background).
    MASK_PARTS: tuple[str, ...] = (
        "arm_left",
        "arm_right",
        "leg_left",
        "leg_right",
        "hair",
    )

    # MaxFilter kernel size: 2 * radius + 1 where radius ≈ 10px
    DILATION_KERNEL_SIZE: int = 21

    # Inpainting parameters
    INPAINT_STEPS: int = 20
    INPAINT_STRENGTH: float = 0.65

    def __init__(self, text_to_image: TextToImageService) -> None:
        """Initialise the service.

        Args:
            text_to_image: The SDXL text-to-image service whose internal _pipe
                will be used as the base for from_pipe() inpainting.
        """
        self._text_to_image = text_to_image
        self._inpaint_pipe = None  # lazy-created via _ensure_inpaint_pipe()

    # ── Public API ────────────────────────────────────────────

    def generate(
        self,
        image: Image.Image,
        model_dict: dict,
        parts_dir: Path,
        prompt: str,
        style: str = "anime",
    ) -> Path | None:
        """Build mask, inpaint, save body_base.png.

        Returns path to body_base.png, or None if generation was skipped.
        Updates model_dict in-place with "body_layer" key on success.

        Fallback behaviour:
            - Empty mask → skip entirely, return None.
            - from_pipe() failure → save original image as body_base.png (degraded).
            - OOM → log warning, save original image as body_base.png (degraded).
            - Any other exception from the pipe → log warning, return None (no key set).
        """
        canvas_size = (image.width, image.height)
        output_path = parts_dir / "body_base.png"

        # ── 1. Build mask ───────────────────────────────────────
        mask = self._build_mask(model_dict, parts_dir, canvas_size)
        if mask is None:
            logger.info("BodyLayerService: mask is empty, skipping body layer generation")
            return None

        # ── 2. Ensure inpaint pipe ─────────────────────────────
        self._ensure_inpaint_pipe()

        if self._inpaint_pipe is None:
            # from_pipe() failed — use Approach 5 fallback (original image)
            logger.warning(
                "BodyLayerService: inpaint pipe unavailable, saving original image as fallback"
            )
            self._save_fallback(image, output_path)
            model_dict["body_layer"] = "body_base.png"
            return output_path

        # ── 3. Run inpainting ──────────────────────────────────
        try:
            import torch

            # Get style-specific negative prompt for consistent inpainting quality
            style_preset = Character2DService.STYLE_PRESETS.get(
                style, Character2DService.STYLE_PRESETS.get("anime", {})
            )
            negative_prompt = style_preset.get("negative", "")

            result = self._inpaint_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image.convert("RGB"),
                mask_image=mask,
                num_inference_steps=self.INPAINT_STEPS,
                strength=self.INPAINT_STRENGTH,
            )
            inpainted: Image.Image = result.images[0]

            # Convert back to RGBA preserving original alpha channel
            original_rgba = image.convert("RGBA")
            inpainted_rgba = inpainted.convert("RGBA")
            # Use original alpha so transparency is preserved correctly
            r, g, b, _ = inpainted_rgba.split()
            _, _, _, a = original_rgba.split()
            body_image = Image.merge("RGBA", (r, g, b, a))

            body_image.save(str(output_path))
            logger.info(f"BodyLayerService: body_base.png saved to {output_path}")

            model_dict["body_layer"] = "body_base.png"
            return output_path

        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom:
            # RuntimeError may also be CUDA OOM in older PyTorch versions
            if isinstance(oom, RuntimeError) and "out of memory" not in str(oom).lower():
                raise  # Re-raise non-OOM RuntimeErrors
            logger.warning(
                f"BodyLayerService: OOM during inpainting ({oom}), "
                "saving original image as fallback"
            )
            self._save_fallback(image, output_path)
            model_dict["body_layer"] = "body_base.png"
            return output_path

        except Exception as exc:
            logger.warning(
                f"BodyLayerService: inpainting failed ({exc}), skipping body layer"
            )
            return None

    def unload_model(self) -> None:
        """Release inpainting pipeline and free VRAM.

        Note: this only unloads the inpainting-specific UNet weights.
        Shared components (VAE, text encoders) remain in the base SDXL pipe.
        """
        if self._inpaint_pipe is not None:
            del self._inpaint_pipe
            self._inpaint_pipe = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("BodyLayerService: inpaint pipeline unloaded")

    # ── Internal helpers ──────────────────────────────────────

    def _build_mask(
        self,
        model_dict: dict,
        parts_dir: Path,
        canvas_size: tuple[int, int],
    ) -> Image.Image | None:
        """Build the inpainting mask from the union of MASK_PARTS alpha channels.

        White (255) = inpaint, black (0) = preserve.
        The mask is dilated by DILATION_KERNEL_SIZE to extend slightly under
        part overlap zones, hiding seams.

        Returns:
            L-mode PIL Image (same size as canvas_size) or None if empty.
        """
        canvas_w, canvas_h = canvas_size
        # Start with an all-black (preserve) mask
        mask = Image.new("L", (canvas_w, canvas_h), 0)

        # Build a lookup: part name → file path from model_dict
        part_files: dict[str, str] = {}
        for part_info in model_dict.get("parts", []):
            name = part_info.get("name", "")
            image_file = part_info.get("image", "")
            if name and image_file:
                part_files[name] = image_file

        found_any = False
        for part_name in self.MASK_PARTS:
            # Look up the file from model_dict first, fallback to <name>.png
            image_file = part_files.get(part_name, f"{part_name}.png")
            part_path = parts_dir / image_file
            if not part_path.exists():
                logger.debug(f"BodyLayerService: mask part not found: {part_path}, skipping")
                continue

            try:
                part_img = Image.open(str(part_path)).convert("RGBA")
            except Exception as exc:
                logger.warning(f"BodyLayerService: could not open {part_path} ({exc}), skipping")
                continue

            # Extract alpha channel of this part in its own coordinate space
            _, _, _, part_alpha = part_img.split()

            # Find the bounding box / position of this part in the canvas
            bounds = self._find_part_bounds(part_name, model_dict)
            if bounds is not None:
                x, y, w, h = bounds
                # Resize alpha to the part's actual size if needed
                if part_alpha.size != (w, h):
                    part_alpha = part_alpha.resize((w, h), Image.Resampling.BICUBIC)
                # Paste the alpha into the canvas-sized mask at the correct position
                # Use the alpha itself as the paste data and no mask (direct overwrite)
                region = mask.crop((x, y, x + w, y + h))
                # Union: take max of existing mask and new part alpha
                region_arr = np.array(region, dtype=np.uint8)
                alpha_arr = np.array(part_alpha, dtype=np.uint8)
                # Ensure same shape
                if region_arr.shape == alpha_arr.shape:
                    union_arr = np.maximum(region_arr, alpha_arr)
                    union_region = Image.fromarray(union_arr, "L")
                    mask.paste(union_region, (x, y))
                    found_any = True
            else:
                # No bounds info — place the part alpha directly (top-left fallback)
                # This handles edge cases where model_dict lacks bounds
                part_alpha_resized = part_alpha.resize((canvas_w, canvas_h), Image.Resampling.BICUBIC)
                mask_arr = np.array(mask, dtype=np.uint8)
                alpha_arr = np.array(part_alpha_resized, dtype=np.uint8)
                union_arr = np.maximum(mask_arr, alpha_arr)
                mask = Image.fromarray(union_arr, "L")
                found_any = True

        if not found_any:
            return None

        # Check if mask has any non-zero pixels before dilation
        mask_arr = np.array(mask, dtype=np.uint8)
        if mask_arr.max() == 0:
            return None

        # Apply morphological dilation (MaxFilter) to extend mask edges
        mask = mask.filter(ImageFilter.MaxFilter(self.DILATION_KERNEL_SIZE))

        # Final check: still non-empty after dilation?
        mask_arr = np.array(mask, dtype=np.uint8)
        if mask_arr.max() == 0:
            return None

        return mask

    def _find_part_bounds(
        self,
        part_name: str,
        model_dict: dict,
    ) -> tuple[int, int, int, int] | None:
        """Find the canvas-space (x, y, w, h) for a named part in model_dict."""
        for part_info in model_dict.get("parts", []):
            if part_info.get("name") == part_name:
                b = part_info.get("bounds")
                if b:
                    return (
                        int(b.get("x", 0)),
                        int(b.get("y", 0)),
                        int(b.get("w", 0)),
                        int(b.get("h", 0)),
                    )
        return None

    def _ensure_inpaint_pipe(self) -> None:
        """Lazy-initialise the SDXL inpainting pipeline via from_pipe().

        Shares UNet/VAE/text-encoders with the base SDXL pipe so only the
        inpainting UNet weights are loaded additionally (~3-4 GB extra VRAM).

        On any failure, logs a warning and leaves self._inpaint_pipe = None.
        The caller must handle the None case.
        """
        if self._inpaint_pipe is not None:
            return  # Already initialised

        base_pipe = getattr(self._text_to_image, "_pipe", None)
        if base_pipe is None:
            logger.warning(
                "BodyLayerService: base SDXL pipe not loaded yet "
                "(text_to_image._pipe is None), cannot create inpaint pipe"
            )
            return

        try:
            from diffusers import StableDiffusionXLInpaintPipeline

            logger.info("BodyLayerService: creating inpaint pipeline via from_pipe()")
            self._inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pipe(base_pipe)
            # Move to same device as base pipe
            device = getattr(base_pipe, "device", None)
            if device is not None:
                self._inpaint_pipe.to(device)
            logger.info("BodyLayerService: inpaint pipeline ready")

        except Exception as exc:
            logger.warning(
                f"BodyLayerService: failed to create inpaint pipeline ({exc}), "
                "body layer will use original image as fallback"
            )
            self._inpaint_pipe = None

    @staticmethod
    def _save_fallback(image: Image.Image, output_path: Path) -> None:
        """Save the original character image as body_base.png (degraded fallback).

        The image is saved as RGBA to preserve transparency. This is used when
        inpainting fails so the animator still has a valid first layer.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGBA").save(str(output_path))
        logger.info(f"BodyLayerService: fallback body_base.png saved to {output_path}")
