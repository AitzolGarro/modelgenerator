"""
MultiView Extractor Service — Zero123Plus Standalone.

Generates 6 orbital character views from a single front-facing character image
using Zero123Plus (standalone, no mesh reconstruction).

Output views are saved as character_view_0.png through character_view_5.png
in the specified output directory.

Zero123Plus view grid layout (960 × 640 output image):
  3 rows × 2 cols → 6 views at 320 × 320 px each

View angles (approximate, as used in InstantMesh / Zero123Plus default cameras):
  Index  Row  Col   Elevation  Azimuth     Description
  ─────  ───  ───   ─────────  ───────     ───────────
    0     0    0       +30°      30°        Front-left
    1     0    1       +30°      90°        Right
    2     0    2       +30°     150°        Back-right
    3     1    0       -20°     210°        Back-left
    4     1    1       -20°     270°        Left
    5     1    2       -20°     330°        Front-right

Note: Zero123Plus rearranges as (n=3 rows, m=2 cols) using einops:
  rearrange('c (n h) (m w) -> (n m) c h w', n=3, m=2)
  where h=w=320.  The output image is 960 H × 640 W (3 rows × 2 cols).

Usage:
    svc = MultiViewExtractorService()
    paths = svc.extract(character_image, output_dir)
    # Returns list of 6 Paths; on failure returns [char_path] * 6

VRAM: Zero123Plus ~4 GB fp16.  Caller MUST unload other models before calling
extract() and is responsible for VRAM budget.  The service unloads itself
at the end of extract() regardless of success/failure.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = get_logger(__name__)

# ── View angle catalogue ──────────────────────────────────────────────────────
# Each tuple: (elevation_deg, azimuth_deg)
# These correspond to the Zero123Plus fixed orbital camera positions as used
# by InstantMesh (run.py) when num_views=6.
_VIEW_ANGLES: list[tuple[float, float]] = [
    (30.0, 30.0),    # view_0: front-left elevated
    (30.0, 90.0),    # view_1: right elevated
    (30.0, 150.0),   # view_2: back-right elevated
    (-20.0, 210.0),  # view_3: back-left low
    (-20.0, 270.0),  # view_4: left low
    (-20.0, 330.0),  # view_5: front-right low
]

# Grid dimensions from Zero123Plus output (fixed)
_GRID_ROWS: int = 3
_GRID_COLS: int = 2
_VIEW_SIZE: int = 320          # each view is 320×320 px
_GRID_W: int = _VIEW_SIZE * _GRID_COLS   # 640
_GRID_H: int = _VIEW_SIZE * _GRID_ROWS   # 960

# Zero123Plus diffusion steps (75 default, reduced for speed)
_DIFFUSION_STEPS: int = 75

# Search paths for the InstantMesh repo (must match image_to_3d_instantmesh.py)
_REPO_SEARCH_PATHS = [
    Path("/app/instantmesh"),                                                    # Docker
    Path(__file__).resolve().parent.parent.parent.parent / "instantmesh",       # project root
]


def _find_instantmesh_repo() -> Path | None:
    for p in _REPO_SEARCH_PATHS:
        if (p / "run.py").exists():
            return p
    return None


class MultiViewExtractorService:
    """Generates 6 orbital views from a single character image via Zero123Plus.

    The model is loaded lazily on the first call to extract() and ALWAYS
    unloaded (GPU memory freed) before extract() returns — even on failure.
    """

    def __init__(self) -> None:
        self._pipeline = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        character_image: "PILImage.Image",
        output_dir: Path,
        *,
        character_path: Path | None = None,
        seed: int = 42,
    ) -> list[Path]:
        """Generate 6 orbital views from a single character image.

        Saves view images as character_view_0.png … character_view_5.png
        inside output_dir.

        Args:
            character_image: PIL Image of the character (RGB or RGBA).
            output_dir: Directory where view images will be saved.
            character_path: Optional path to the original character.png —
                used as fallback when extraction fails.  Defaults to
                output_dir / "character.png".
            seed: Random seed for reproducible results (default 42).

        Returns:
            List of 6 Paths to the generated view images.
            On any failure, returns [character_fallback_path] * 6.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if character_path is None:
            character_path = output_dir / "character.png"

        try:
            return self._extract_impl(character_image, output_dir, seed=seed)
        except Exception as exc:
            logger.warning(
                f"MultiViewExtractorService: Zero123Plus extraction failed ({exc}). "
                "Falling back to front-view for all 6 views."
            )
            return [character_path] * 6
        finally:
            self.unload_model()

    def unload_model(self) -> None:
        """Delete the Zero123Plus pipeline and free all VRAM + RAM."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            try:
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("MultiViewExtractorService: Zero123Plus pipeline unloaded")

    # ── Private implementation ─────────────────────────────────────────────────

    def _extract_impl(
        self,
        character_image: "PILImage.Image",
        output_dir: Path,
        *,
        seed: int,
    ) -> list[Path]:
        """Core extraction — loads pipeline, runs inference, crops grid."""
        import torch
        from PIL import Image

        self._load_pipeline()

        # Prepare input image (Zero123Plus expects white-background RGB)
        input_image = self._prepare_input(character_image)

        logger.info(
            f"MultiViewExtractorService: running Zero123Plus "
            f"({_DIFFUSION_STEPS} steps, seed={seed})"
        )

        # Seeded generator for reproducibility
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Run Zero123Plus inference
        output = self._pipeline(
            input_image,
            num_inference_steps=_DIFFUSION_STEPS,
            generator=generator,
        )
        grid_image: Image.Image = output.images[0]

        logger.info(
            f"MultiViewExtractorService: grid image size = {grid_image.size}"
        )

        # Crop the 3×2 grid into 6 individual view images
        view_paths = self._crop_and_save(grid_image, output_dir)

        logger.info(
            f"MultiViewExtractorService: saved {len(view_paths)} views → {output_dir}"
        )
        return view_paths

    def _load_pipeline(self) -> None:
        """Lazy-load the Zero123Plus DiffusionPipeline with custom UNet weights."""
        if self._pipeline is not None:
            return

        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        from huggingface_hub import hf_hub_download

        repo_dir = _find_instantmesh_repo()

        # Ensure the local zero123plus custom pipeline code is importable.
        # Zero123Plus uses `custom_pipeline=` which in older diffusers resolves to a
        # local directory when given an absolute path.  In newer diffusers, pass
        # the module path string directly.
        if repo_dir is not None:
            zero123plus_dir = str(repo_dir / "zero123plus")
            # Add parent to sys.path so diffusers can find 'zero123plus' as a module
            parent = str(repo_dir)
            if parent not in sys.path:
                sys.path.insert(0, parent)
        else:
            zero123plus_dir = "zero123plus"
            logger.warning(
                "MultiViewExtractorService: InstantMesh repo not found; "
                "assuming zero123plus is importable from sys.path"
            )

        logger.info("MultiViewExtractorService: loading Zero123Plus pipeline (fp16)...")

        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline=zero123plus_dir,
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
        )

        # Load custom white-background UNet weights (InstantMesh fine-tuned)
        unet_ckpt_path = self._resolve_unet_weights(repo_dir)
        if unet_ckpt_path is not None:
            logger.info(
                f"MultiViewExtractorService: loading custom UNet from {unet_ckpt_path}"
            )
            state_dict = torch.load(unet_ckpt_path, map_location="cpu")
            pipeline.unet.load_state_dict(state_dict, strict=True)
        else:
            logger.info(
                "MultiViewExtractorService: custom UNet weights not found; "
                "using base Zero123Plus weights (white-BG UNet will be auto-downloaded)"
            )
            unet_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="diffusion_pytorch_model.bin",
                repo_type="model",
            )
            state_dict = torch.load(unet_ckpt_path, map_location="cpu")
            pipeline.unet.load_state_dict(state_dict, strict=True)

        pipeline = pipeline.to("cuda")

        self._pipeline = pipeline
        logger.info("MultiViewExtractorService: Zero123Plus pipeline loaded")

    @staticmethod
    def _resolve_unet_weights(repo_dir: Path | None) -> Path | None:
        """Look for the InstantMesh custom UNet weights file."""
        if repo_dir is None:
            return None

        # Common config locations used by InstantMesh run.py
        candidate_configs = [
            repo_dir / "configs" / "instant-mesh-large.yaml",
            repo_dir / "configs" / "instant-mesh-base.yaml",
        ]
        for cfg_path in candidate_configs:
            if not cfg_path.exists():
                continue
            try:
                from omegaconf import OmegaConf
                cfg = OmegaConf.load(cfg_path)
                unet_path = Path(cfg.infer_config.unet_path)
                if unet_path.exists():
                    return unet_path
                # Try relative to repo_dir
                rel = repo_dir / cfg.infer_config.unet_path
                if rel.exists():
                    return rel
            except Exception:
                continue

        # Fallback: look for the binary directly in common locations
        for candidate in [
            repo_dir / "ckpts" / "diffusion_pytorch_model.bin",
            repo_dir / "checkpoints" / "diffusion_pytorch_model.bin",
        ]:
            if candidate.exists():
                return candidate

        return None

    @staticmethod
    def _prepare_input(image: "PILImage.Image") -> "PILImage.Image":
        """Prepare input image for Zero123Plus (RGB, white background for RGBA)."""
        from PIL import Image
        import numpy as np

        if image.mode == "RGBA":
            # Composite onto white background
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.getchannel("A"))
            return bg
        elif image.mode == "RGB":
            return image
        else:
            return image.convert("RGB")

    @staticmethod
    def _crop_and_save(grid_image: "PILImage.Image", output_dir: Path) -> list[Path]:
        """Crop 960×640 grid image into 6 individual 320×320 view images.

        Grid layout (3 rows × 2 cols, row-major order):
          [view_0] [view_1]
          [view_2] [view_3]
          [view_4] [view_5]

        Corresponds to einops rearrange order used in run.py:
          rearrange('c (n h) (m w) -> (n m) c h w', n=3, m=2)
          → indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5
        """
        from PIL import Image

        paths: list[Path] = []
        idx = 0
        for row in range(_GRID_ROWS):
            for col in range(_GRID_COLS):
                left = col * _VIEW_SIZE
                upper = row * _VIEW_SIZE
                right = left + _VIEW_SIZE
                lower = upper + _VIEW_SIZE
                view = grid_image.crop((left, upper, right, lower))
                view_path = output_dir / f"character_view_{idx}.png"
                view.save(str(view_path), "PNG")
                paths.append(view_path)
                idx += 1

        return paths
