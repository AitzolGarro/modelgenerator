"""
Text-to-Image service using Stable Diffusion XL.

Swap this for any other text-to-image model by implementing TextToImageService.
Examples: Flux, Kandinsky, DeepFloyd IF, etc.
"""

import torch
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import TextToImageService

logger = get_logger(__name__)
settings = get_settings()


class SDXLTextToImageService(TextToImageService):
    """Stable Diffusion XL text-to-image implementation."""

    def __init__(self) -> None:
        self._pipe = None

    def load_model(self) -> None:
        if self._pipe is not None:
            logger.info("SDXL model already loaded")
            return

        from diffusers import StableDiffusionXLPipeline

        logger.info(f"Loading SDXL model: {settings.TEXT_TO_IMAGE_MODEL}")

        dtype = getattr(torch, settings.TEXT_TO_IMAGE_DTYPE, torch.float16)

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            settings.TEXT_TO_IMAGE_MODEL,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
        )
        self._pipe.to(settings.TEXT_TO_IMAGE_DEVICE)

        # Optimizations for RTX 5090
        if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers attention enabled")
            except Exception:
                logger.info("xformers not available, using default attention")

        logger.info("SDXL model loaded successfully")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            self.load_model()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.TEXT_TO_IMAGE_DEVICE).manual_seed(seed)

        # Enhance prompt for 3D model generation
        enhanced_prompt = (
            f"{prompt}, high quality, detailed, sharp focus, "
            "studio lighting, product photography, centered, white background"
        )

        default_negative = (
            "blurry, low quality, distorted, deformed, noisy, "
            "watermark, text, multiple objects"
        )
        neg = negative_prompt or default_negative

        logger.info(f"Generating image: steps={num_steps}, guidance={guidance_scale}")

        result = self._pipe(
            prompt=enhanced_prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]
        logger.info(f"Image generated: {image.size}")
        return image

    def unload_model(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
            logger.info("SDXL model unloaded")


class MockTextToImageService(TextToImageService):
    """
    Mock service for testing without GPU.
    Generates a simple colored image with the prompt text.
    """

    def load_model(self) -> None:
        logger.info("Mock text-to-image service loaded")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        from PIL import ImageDraw, ImageFont

        logger.info(f"Mock generating image for: {prompt[:60]}")

        # Create a gradient-ish image
        img = Image.new("RGB", (width, height), color=(40, 40, 60))
        draw = ImageDraw.Draw(img)

        # Draw prompt text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()

        # Word wrap
        words = prompt.split()
        lines = []
        current = ""
        for w in words:
            test = f"{current} {w}".strip()
            if len(test) > 40:
                lines.append(current)
                current = w
            else:
                current = test
        if current:
            lines.append(current)

        y = height // 2 - len(lines) * 15
        for line in lines:
            draw.text((width // 4, y), line, fill=(200, 200, 220), font=font)
            y += 30

        draw.text((20, 20), "[MOCK IMAGE]", fill=(255, 100, 100), font=font)

        return img

    def unload_model(self) -> None:
        logger.info("Mock text-to-image service unloaded")
