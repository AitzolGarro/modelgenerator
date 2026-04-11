"""
Generative 2D Animator Service — Wan2.1 Image-to-Video.

Generates sprite-sheet animations using Wan2.1 I2V 14B (int4 quantized)
for true video generation from a single character image.

Architecture:
    - WanImageToVideoPipeline (14B parameter video diffusion transformer)
    - INT4 quantization via bitsandbytes (transformer + text_encoder)
    - CLIP Vision encoder preserves character identity from the input image
    - Generates 17 frames in ONE pass with native temporal coherence
    - enable_model_cpu_offload() keeps peak VRAM at ~15 GB

Why Wan2.1 instead of AnimateDiff / per-frame SDXL:
    - Native I2V: takes the EXACT input image and generates motion from it
    - 14B params = much better identity preservation and motion quality
    - True video generation model (not adapted image model)
    - INT4 quantization makes it fit in 32GB GPU + 32GB RAM

Models (auto-downloaded on first use, ~27 GB total cached):
    - Wan-AI/Wan2.1-I2V-14B-480P-Diffusers

Output:
    - sprite_sheet.png  — horizontal strip of N frames (background removed)
    - animation.json    — metadata for game engine / frontend
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.services.pose_sequences import VALID_ANIMATIONS

if TYPE_CHECKING:
    from app.services.base import TextToImageService

logger = get_logger(__name__)

# ── Model ────────────────────────────────────────────────────────────────────
_MODEL_ID: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# ── Generation parameters ────────────────────────────────────────────────────
_NUM_FRAMES: int = 33          # Wan2.1 4k+1: k=8 → 33 frames (~2s at 16fps)
_INFERENCE_STEPS: int = 30     # Balances quality vs generation time
_GUIDANCE_SCALE: float = 9.0   # High guidance for dramatic, visible motion
_MAX_AREA: int = 480 * 832     # 480p — safe for 61 frames on RTX 5090 32GB with int4
_DEFAULT_FPS: int = 16         # Wan2.1 native fps

# ── Prompt templates per animation type ──────────────────────────────────────
# Each prompt MUST contain ≥1 strong action verb and the phrase
# "clear pose changes between frames" to maximise Wan2.1 inter-frame motion.
_ANIMATION_PROMPTS: dict[str, str] = {
    "idle": (
        "character standing alive breathing deeply, chest rises and falls visibly, "
        "weight shifts from foot to foot, head tilts and looks around, "
        "hands fidget, large exaggerated body sway, dynamic alive motion"
    ),
    "walk": (
        "character walks forward with big dramatic steps, legs swing wide, "
        "arms pump back and forth with large motion, body bounces up and down, "
        "hips sway side to side, head bobs with each stride, full body movement"
    ),
    "run": (
        "character runs at full sprint speed, legs pump with huge strides, "
        "arms swing rapidly, body leans far forward, feet kick up behind, "
        "hair and clothes flow in the wind, intense physical exertion"
    ),
    "attack": (
        "character winds up and swings weapon with massive force, entire body rotates, "
        "arm fully extends in a wide sweeping arc, feet shift position, "
        "body twists dramatically, powerful follow-through motion, impact energy"
    ),
    "jump": (
        "character crouches deeply then launches upward with explosive force, "
        "legs fully extend, arms reach overhead, body rises high off the ground, "
        "reaches peak height suspended in air, then falls and lands with bent knees"
    ),
    "dance": (
        "character dances with energetic full-body movements, hips swing wide, "
        "arms wave and sweep through the air, body spins and bounces rhythmically, "
        "feet step and kick, head moves with the beat, joyful dynamic motion"
    ),
    "wave": (
        "character raises arm high overhead and waves hand back and forth widely, "
        "big sweeping arm movement, wrist rotates, elbow bends and straightens, "
        "body sways with the gesture, enthusiastic greeting motion"
    ),
    "hurt": (
        "character gets hit hard and staggers backward dramatically, "
        "body lurches and bends, head snaps back violently, arms flail, "
        "stumbles and nearly falls, pain reaction with full body recoil"
    ),
    # Rotational animation types (Phase 3 — multi-view angle routing)
    "rotate_left": (
        "character smoothly rotates body leftward, turns in place, "
        "showing side and back profile, clear pose changes between frames"
    ),
    "rotate_right": (
        "character smoothly rotates body rightward, turns in place, "
        "showing side and back profile, clear pose changes between frames"
    ),
    "turnaround": (
        "character turns fully around to face away and back again, "
        "complete 180 degree turn, clear pose changes between frames"
    ),
}

_DEFAULT_NEGATIVE_PROMPT: str = (
    "static image, still picture, frozen, motionless, no movement, statue, "
    "overall gray, blurry, low quality, worst quality, JPEG artifacts, "
    "deformed, disfigured, misshapen limbs, extra limbs, extra fingers, "
    "fused fingers, poorly drawn hands, poorly drawn faces, "
    "different character, morphing, melting, bright tones, overexposed, "
    "subtitles, text, watermark, logo, messy background, walking backwards"
)

# ── Rotational animation routing ─────────────────────────────────────────────
# Animation types that involve body rotation and therefore benefit from using
# a side/back orbital view as the Wan2.1 reference image instead of the
# default front-facing character image.
_ROTATION_TYPES: frozenset[str] = frozenset({
    "rotate_left",
    "rotate_right",
    "turnaround",
})

# View angles matching MultiViewExtractorService._VIEW_ANGLES (elev, azim).
# Used by _select_view() to pick the closest pre-generated orbital view.
_VIEW_ANGLES: list[tuple[float, float]] = [
    (30.0, 30.0),    # view_0: front-left elevated
    (30.0, 90.0),    # view_1: right elevated
    (30.0, 150.0),   # view_2: back-right elevated
    (-20.0, 210.0),  # view_3: back-left low
    (-20.0, 270.0),  # view_4: left low
    (-20.0, 330.0),  # view_5: front-right low
]

# Target azimuth angle (degrees) for each rotational animation type.
# These define which orbital direction is most relevant for each animation.
_ROTATION_TARGET_AZIMUTH: dict[str, float] = {
    "rotate_left":  270.0,   # left side view
    "rotate_right":  90.0,   # right side view
    "turnaround":   210.0,   # back-left (mid-turn view)
}


def _select_view(parts_dir: Path, animation_type: str) -> Path:
    """Select the closest pre-generated orbital view for a rotational animation.

    Picks the character_view_*.png file whose azimuth angle is closest to the
    target azimuth for the given animation_type.

    Falls back to character.png (front view) when:
    - The animation type is not in _ROTATION_TYPES
    - No character_view_*.png files exist in parts_dir

    Args:
        parts_dir: Directory containing character.png and optionally
                   character_view_0.png … character_view_5.png.
        animation_type: Animation type string.

    Returns:
        Path to the selected view image.
    """
    front_view = parts_dir / "character.png"

    if animation_type not in _ROTATION_TYPES:
        return front_view

    # Check if any pre-generated views exist
    view_files = sorted(parts_dir.glob("character_view_*.png"))
    if not view_files:
        logger.warning(
            f"_select_view: no character_view_*.png found in {parts_dir}; "
            "falling back to front view"
        )
        return front_view

    # Build index → path mapping for the views that actually exist
    existing_views: dict[int, Path] = {}
    for vf in view_files:
        stem = vf.stem  # "character_view_N"
        try:
            idx = int(stem.rsplit("_", 1)[-1])
            if 0 <= idx < len(_VIEW_ANGLES):
                existing_views[idx] = vf
        except (ValueError, IndexError):
            continue

    if not existing_views:
        logger.warning(
            f"_select_view: could not parse view indices in {parts_dir}; "
            "falling back to front view"
        )
        return front_view

    target_azimuth = _ROTATION_TARGET_AZIMUTH.get(animation_type, 90.0)

    # Find the existing view whose azimuth is closest to the target
    best_idx: int | None = None
    best_delta: float = float("inf")
    for idx, path in existing_views.items():
        _, azimuth = _VIEW_ANGLES[idx]
        # Circular angular distance (mod 360)
        delta = abs((azimuth - target_azimuth + 180) % 360 - 180)
        if delta < best_delta:
            best_delta = delta
            best_idx = idx

    selected = existing_views[best_idx]  # type: ignore[index]
    logger.info(
        f"_select_view: animation_type={animation_type!r} → "
        f"target_azimuth={target_azimuth}° → selected view_{best_idx} "
        f"(azimuth={_VIEW_ANGLES[best_idx][1]}°, delta={best_delta:.1f}°)"
    )
    return selected


class GenerativeAnimator2DService:
    """Generates 2D sprite sheet animations using Wan2.1 I2V (int4 quantized).

    Takes a single character image and generates a video animation of that
    character performing the requested action.  All frames are generated in
    one inference pass with native temporal coherence.
    """

    def __init__(
        self,
        text_to_image: "TextToImageService",
    ) -> None:
        self._text_to_image = text_to_image
        self._pipe = None  # lazy-init

    # ── Public API ────────────────────────────────────────────────────────────

    def animate(
        self,
        model_dir: Path,
        animation_type: str,
        num_frames: int | None = None,
        *,
        prompt: str = "",
        seed: int | None = None,
        enhance_animation: bool = False,
        enhance_personality: str = "calm",
        enhance_intensity: float = 0.7,
    ) -> Path:
        """Generate a sprite sheet animation using Wan2.1 I2V.

        Args:
            model_dir: Directory containing character.png.
            animation_type: One of the 8 valid animation types.
            num_frames: Number of frames (default 17, Wan2.1 native).
            prompt: Additional text prompt context.
            seed: Random seed for reproducibility.
            enhance_animation: If True, apply AnimationEnhancerService post-processing.
            enhance_personality: Motion personality profile (default "calm").
            enhance_intensity: Enhancement strength 0.0–1.0 (default 0.7).

        Returns:
            Path to output directory with sprite_sheet.png + animation.json.
        """
        if animation_type not in VALID_ANIMATIONS:
            valid = ", ".join(sorted(VALID_ANIMATIONS))
            raise ValueError(
                f"Unknown animation type: {animation_type!r}. Valid: {valid}"
            )

        if num_frames is None:
            num_frames = _NUM_FRAMES

        import random as _random
        if seed is None:
            seed = _random.randint(0, 2**32 - 1)

        logger.info(
            f"GenerativeAnimator2DService: animate type={animation_type!r} "
            f"n={num_frames} seed={seed}"
        )

        # Load reference character image
        char_png = model_dir / "character.png"
        if not char_png.exists():
            raise FileNotFoundError(f"character.png not found in {model_dir}")

        # ★ Angle routing: for rotational animations, use the closest orbital view
        # as the Wan2.1 reference image instead of the default front-facing image.
        if animation_type in _ROTATION_TYPES:
            ref_png = _select_view(model_dir, animation_type)
            logger.info(
                f"GenerativeAnimator2DService: rotational animation — "
                f"using {ref_png.name!r} as reference image"
            )
        else:
            ref_png = char_png

        ref_image = Image.open(str(ref_png)).convert("RGB")

        # Resize for Wan2.1 480p (must be compatible with VAE + patch size)
        ref_resized, anim_h, anim_w = self._resize_for_wan(ref_image)

        # Build pipeline (lazy)
        self._ensure_pipeline()

        # Build prompt
        anim_prompt = _ANIMATION_PROMPTS.get(
            animation_type, f"{animation_type} animation"
        )
        if prompt:
            full_prompt = f"{prompt}, {anim_prompt}, same character, same outfit"
        else:
            full_prompt = f"{anim_prompt}, same character, same outfit, high quality"

        logger.info(f"GenerativeAnimator2DService: prompt={full_prompt!r}")

        # Generate animation — ALL frames in one pass
        import torch
        generator = torch.Generator(device="cpu").manual_seed(seed)

        output = self._pipe(
            image=ref_resized,
            prompt=full_prompt,
            negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
            height=anim_h,
            width=anim_w,
            num_frames=num_frames,
            num_inference_steps=_INFERENCE_STEPS,
            guidance_scale=_GUIDANCE_SCALE,
            generator=generator,
        )

        # output.frames[0] is a list of numpy arrays (float32, 0-1 range)
        raw_frames = output.frames[0]
        logger.info(
            f"GenerativeAnimator2DService: got {len(raw_frames)} raw frames"
        )

        # ★ CRITICAL: Unload the Wan2.1 pipeline IMMEDIATELY after generation
        # to free ~15GB RAM before post-processing (rembg, enhancement, sprite sheet).
        # On 30GB systems, keeping the pipeline loaded during post-processing causes OOM.
        del output
        self.unload_model()
        logger.info("GenerativeAnimator2DService: pipeline unloaded after frame generation (OOM prevention)")

        # Convert numpy arrays to PIL Images
        frames = self._to_pil_frames(raw_frames)

        # Scale back to original character size
        orig_size = (ref_image.width, ref_image.height)
        if (anim_w, anim_h) != orig_size:
            frames = [
                f.resize(orig_size, Image.Resampling.LANCZOS) for f in frames
            ]

        # ★ Optional: AnimationEnhancerService post-processing
        # Sits between _to_pil_frames() and _remove_backgrounds() so that
        # loop blending and idle transforms operate on full RGB frames
        # (rembg alpha edges would create artefacts if transformed after).
        anim_metadata: dict | None = None
        if enhance_animation:
            try:
                from app.services.animation_enhancer import AnimationEnhancerService
                enhancer = AnimationEnhancerService()
                frames, anim_metadata = enhancer.enhance(
                    frames,
                    animation_type,
                    personality=enhance_personality,
                    intensity=enhance_intensity,
                )
                logger.info(
                    "GenerativeAnimator2DService: enhancement applied "
                    f"(personality={enhance_personality!r}, intensity={enhance_intensity})"
                )
            except Exception as exc:
                logger.warning(
                    f"GenerativeAnimator2DService: enhancer import/init failed ({exc}); "
                    "continuing without enhancement"
                )

        # Remove backgrounds (Wan2.1 generates with backgrounds)
        frames = self._remove_backgrounds(frames)

        # Save sprite sheet
        output_dir = model_dir.parent / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_sprite_sheet(
            frames=frames,
            canvas_size=orig_size,
            animation_type=animation_type,
            fps=_DEFAULT_FPS,
            output_dir=output_dir,
            anim_metadata=anim_metadata,
        )

        logger.info(
            f"GenerativeAnimator2DService: animation complete → {output_dir}"
        )
        return output_dir

    def unload_model(self) -> None:
        """Release the Wan2.1 pipeline and free all VRAM + RAM."""
        if self._pipe is not None:
            # With cpu_offload, components may be on CPU or GPU
            del self._pipe
            self._pipe = None
            try:
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("GenerativeAnimator2DService: Wan2.1 pipeline unloaded")

    # ── Pipeline construction ─────────────────────────────────────────────────

    def _ensure_pipeline(self) -> None:
        """Lazy-init the Wan2.1 I2V pipeline with int4 quantization.

        Quantizes transformer + text_encoder to int4 (nf4) via bitsandbytes.
        This reduces RAM from ~65 GB to ~15 GB, making it fit in 32 GB systems.
        enable_model_cpu_offload() keeps peak VRAM at ~15 GB.
        """
        if self._pipe is not None:
            return

        try:
            import torch
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from diffusers.quantizers import PipelineQuantizationConfig
            from transformers import CLIPVisionModel

            logger.info(
                f"GenerativeAnimator2DService: loading Wan2.1 I2V pipeline "
                f"from {_MODEL_ID!r} (int4 quantized)"
            )

            # Image encoder (CLIP) — fp32, small (~1.2 GB)
            logger.info("  Loading CLIP image encoder (fp32)...")
            image_encoder = CLIPVisionModel.from_pretrained(
                _MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32,
            )

            # VAE — fp32, small (~0.5 GB)
            logger.info("  Loading VAE (fp32)...")
            vae = AutoencoderKLWan.from_pretrained(
                _MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
            )

            # Full pipeline with int4 quantized transformer + text_encoder
            logger.info("  Loading pipeline (int4 transformer + text_encoder)...")
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )

            pipe = WanImageToVideoPipeline.from_pretrained(
                _MODEL_ID,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
            )

            # Memory optimizations
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()

            self._pipe = pipe
            logger.info("GenerativeAnimator2DService: Wan2.1 pipeline ready")

        except Exception as exc:
            logger.error(
                f"GenerativeAnimator2DService: pipeline build failed ({exc})"
            )
            raise RuntimeError(
                f"GenerativeAnimator2DService: failed to build pipeline: {exc}"
            ) from exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resize_for_wan(
        self, image: Image.Image,
    ) -> tuple[Image.Image, int, int]:
        """Resize image for Wan2.1 (must be compatible with VAE + patch size).

        Returns (resized_image, height, width).
        """
        if self._pipe is None:
            # Fallback if pipe not loaded yet — use standard mod value
            mod_value = 16
        else:
            mod_value = (
                self._pipe.vae_scale_factor_spatial
                * self._pipe.transformer.config.patch_size[1]
            )

        aspect_ratio = image.height / image.width
        height = round(np.sqrt(_MAX_AREA * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(_MAX_AREA / aspect_ratio)) // mod_value * mod_value

        # Ensure minimum dimensions
        height = max(mod_value, height)
        width = max(mod_value, width)

        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        return resized, height, width

    @staticmethod
    def _to_pil_frames(raw_frames: list) -> list[Image.Image]:
        """Convert raw output frames (numpy float32 or PIL) to PIL Images."""
        result = []
        for f in raw_frames:
            if isinstance(f, np.ndarray):
                # Wan2.1 outputs float32 arrays in [0, 1] range
                arr = (np.clip(f, 0, 1) * 255).astype(np.uint8)
                result.append(Image.fromarray(arr))
            elif isinstance(f, Image.Image):
                result.append(f)
            else:
                # Fallback: try converting
                result.append(Image.fromarray(np.array(f, dtype=np.uint8)))
        return result

    @staticmethod
    def _remove_backgrounds(frames: list[Image.Image]) -> list[Image.Image]:
        """Remove backgrounds from generated frames using rembg.

        Wan2.1 generates frames WITH backgrounds. We need transparent
        backgrounds for sprite sheets.

        Falls back to original frames if rembg is unavailable.
        """
        try:
            from rembg import remove as rembg_remove
        except ImportError:
            logger.warning(
                "rembg not available — sprite frames will have backgrounds"
            )
            return [f.convert("RGBA") for f in frames]

        result = []
        for i, frame in enumerate(frames):
            try:
                rgba = rembg_remove(frame.convert("RGB"))
                if isinstance(rgba, Image.Image):
                    result.append(rgba.convert("RGBA"))
                else:
                    import io
                    result.append(Image.open(io.BytesIO(rgba)).convert("RGBA"))
            except Exception as exc:
                logger.warning(f"rembg failed on frame {i}: {exc}")
                result.append(frame.convert("RGBA"))
        return result

    @staticmethod
    def _save_sprite_sheet(
        frames: list[Image.Image],
        canvas_size: tuple[int, int],
        animation_type: str,
        fps: int,
        output_dir: Path,
        anim_metadata: "dict | None" = None,
    ) -> tuple[Path, Path]:
        """Assemble frames into a horizontal sprite sheet + animation.json.

        Args:
            frames: RGBA PIL Image frames.
            canvas_size: (width, height) of each frame slot.
            animation_type: Animation type string (e.g. "walk").
            fps: Frames per second for the animation.
            output_dir: Directory to write sprite_sheet.png + animation.json.
            anim_metadata: Optional v2 enhancement metadata from
                AnimationEnhancerService.  When present, writes v2 schema
                (adds "version", "total_duration_ms", "layers", "enhancements",
                and per-frame duration_ms/easing/offset_x/offset_y).
                When None, writes v1 schema (backward-compatible).

        Returns:
            Tuple of (sprite_sheet_path, animation_json_path).
        """
        fw, fh = canvas_size
        n = len(frames)
        sheet = Image.new("RGBA", (fw * n, fh), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            f = frame
            if f.size != (fw, fh):
                f = f.resize((fw, fh), Image.Resampling.LANCZOS)
            if f.mode != "RGBA":
                f = f.convert("RGBA")
            sheet.paste(f, (i * fw, 0), f)

        sprite_path = output_dir / "sprite_sheet.png"
        sheet.save(str(sprite_path), "PNG")
        logger.info(
            f"GenerativeAnimator2DService: sprite sheet saved → {sprite_path}"
        )

        # Base v1 frame list (always present)
        loop_flag = True  # default; overridden by v2 metadata when available
        base_frames = [
            {"index": i, "x": i * fw, "y": 0, "w": fw, "h": fh}
            for i in range(n)
        ]

        if anim_metadata is not None and anim_metadata.get("version") == 2:
            # ── v2 schema ─────────────────────────────────────────────────────
            # Merge per-frame enhancement extras into each frame entry
            frame_extras: list[dict] = anim_metadata.get("frame_extras", [])
            v2_frames = []
            for i, base in enumerate(base_frames):
                entry = dict(base)
                if i < len(frame_extras):
                    entry.update(frame_extras[i])
                v2_frames.append(entry)

            loop_flag = anim_metadata.get("loop", True)
            metadata: dict = {
                "name": animation_type,
                "version": 2,
                "fps": fps,
                "frame_count": n,
                "frame_width": fw,
                "frame_height": fh,
                "loop": loop_flag,
                "total_duration_ms": anim_metadata.get("total_duration_ms", 0),
                "frames": v2_frames,
                "layers": anim_metadata.get("layers", {}),
                "enhancements": anim_metadata.get("enhancements", {}),
            }
        else:
            # ── v1 schema (default / backward-compatible) ─────────────────────
            metadata = {
                "name": animation_type,
                "fps": fps,
                "frame_count": n,
                "frame_width": fw,
                "frame_height": fh,
                "loop": loop_flag,
                "frames": base_frames,
            }

        meta_path = output_dir / "animation.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info(
            f"GenerativeAnimator2DService: animation.json saved → {meta_path} "
            f"(schema v{anim_metadata.get('version', 1) if anim_metadata else 1})"
        )

        return sprite_path, meta_path
