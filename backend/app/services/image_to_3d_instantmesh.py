"""
InstantMesh image-to-3D service (subprocess wrapper).

Calls InstantMesh via subprocess — the repo is cloned to /app/instantmesh
during Docker build. Falls back gracefully if the repo is not present.

Pipeline:
  1. Save input image to a temp PNG
  2. Invoke: python run.py <config>.yaml <input.png> --output_path <dir> [--export_texmap]
  3. Parse the output directory for the generated .obj file
  4. Return path to the output directory

Timeout: 120 seconds (configurable via INSTANTMESH_TIMEOUT env var).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import ImageTo3DService

logger = get_logger(__name__)
settings = get_settings()

# Subprocess timeout in seconds.
SUBPROCESS_TIMEOUT = 300  # InstantMesh needs time to load models on first run

# Search paths for the InstantMesh repo (Docker path first, then local)
_REPO_SEARCH_PATHS = [
    Path("/app/instantmesh"),                                          # Docker
    Path(__file__).resolve().parent.parent.parent.parent / "instantmesh",  # project root / instantmesh
]

def _find_instantmesh_repo() -> Path | None:
    for p in _REPO_SEARCH_PATHS:
        if (p / "run.py").exists():
            return p
    return None

INSTANTMESH_REPO_DIR = _find_instantmesh_repo() or Path("/app/instantmesh")

def _has_nvdiffrast() -> bool:
    try:
        import nvdiffrast  # noqa: F401
        return True
    except ImportError:
        return False

def _instantmesh_available() -> bool:
    """Return True if InstantMesh repo + nvdiffrast are both available.
    InstantMesh fundamentally requires nvdiffrast for FlexiCubes mesh extraction."""
    if _find_instantmesh_repo() is None:
        return False
    if not _has_nvdiffrast():
        logger.info("InstantMesh repo found but nvdiffrast not available — requires Docker")
        return False
    return True


class InstantMeshImageTo3DService(ImageTo3DService):
    """
    InstantMesh-based image-to-3D service (subprocess approach).

    Calls the upstream InstantMesh run.py as a subprocess so we don't need
    to vendor or package the entire repo. The Docker build clones the repo
    to /app/instantmesh and installs its requirements.

    When the InstantMesh repo is not found (e.g. local dev without Docker),
    this service will raise RuntimeError from load_model() and the factory
    will fall back to TripoSR.
    """

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------
    # ImageTo3DService interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Verify InstantMesh is available. No weights loaded here."""
        global INSTANTMESH_REPO_DIR
        found = _find_instantmesh_repo()
        if found is None:
            raise RuntimeError(
                f"InstantMesh repo not found. Searched: {[str(p) for p in _REPO_SEARCH_PATHS]}. "
                "Clone it: git clone https://github.com/TencentARC/InstantMesh.git instantmesh"
            )
        INSTANTMESH_REPO_DIR = found
        self._ready = True
        logger.info(f"InstantMesh ready at {INSTANTMESH_REPO_DIR}")

    def generate(self, image: Image.Image, output_dir: Path) -> Path:
        """
        Run the full InstantMesh pipeline on the given image.

        Args:
            image: Input PIL Image (any mode / size).
            output_dir: Directory where output files will be written.

        Returns:
            Path to the output directory containing the .obj (and optionally
            .mtl + texture .png) files.

        Raises:
            ValueError: If image is None or not a PIL Image.
            RuntimeError: If the subprocess fails or times out.
        """
        if image is None or not isinstance(image, Image.Image):
            raise ValueError("image must be a non-None PIL.Image.Image instance")

        if not self._ready:
            self.load_model()

        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine config path
        config_name = getattr(settings, "INSTANTMESH_CONFIG", "instant-mesh-large")
        config_path = INSTANTMESH_REPO_DIR / "configs" / f"{config_name}.yaml"
        if not config_path.exists():
            raise RuntimeError(
                f"InstantMesh config not found: {config_path}. "
                f"Available configs: {list((INSTANTMESH_REPO_DIR / 'configs').glob('*.yaml'))}"
            )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save input as PNG (InstantMesh expects a plain image file)
            image_rgb = image.convert("RGB") if image.mode != "RGB" else image
            image_rgb.save(str(tmp_path), format="PNG")
            logger.info(f"Saved input image to {tmp_path}")

            # Build command
            cmd = [
                sys.executable,
                str(INSTANTMESH_REPO_DIR / "run.py"),
                str(config_path),
                str(tmp_path),
                "--output_path",
                str(output_dir),
            ]

            # Only use --export_texmap if nvdiffrast is available (needs CUDA compilation)
            use_texture_map = getattr(settings, "INSTANTMESH_USE_TEXTURE_MAP", True)
            if use_texture_map:
                try:
                    import nvdiffrast  # noqa: F401
                    cmd.append("--export_texmap")
                except ImportError:
                    logger.info("nvdiffrast not available, skipping --export_texmap (vertex colors mode)")

            logger.info(f"Running InstantMesh: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(INSTANTMESH_REPO_DIR),
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
            )

            if result.returncode != 0:
                logger.error(f"InstantMesh stdout:\n{result.stdout}")
                logger.error(f"InstantMesh stderr:\n{result.stderr}")
                raise RuntimeError(
                    f"InstantMesh subprocess failed with exit code {result.returncode}. "
                    f"stderr: {result.stderr[-500:]}"
                )

            if result.stdout:
                logger.debug(f"InstantMesh stdout:\n{result.stdout}")

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"InstantMesh subprocess timed out after {SUBPROCESS_TIMEOUT}s"
            ) from exc
        finally:
            # Clean up the temporary input file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        # Locate output mesh: InstantMesh writes to
        # <output_path>/<config_name>/meshes/<image_name>.obj
        obj_files = sorted(output_dir.rglob("*.obj"))
        if not obj_files:
            raise RuntimeError(
                f"InstantMesh finished but no .obj found in {output_dir}. "
                "Check the subprocess output above."
            )

        # Pick the first OBJ and copy it + its MTL/texture to the output dir root
        src_obj = obj_files[0]
        dst_obj = output_dir / "mesh.obj"
        if src_obj != dst_obj:
            import shutil
            shutil.copy2(str(src_obj), str(dst_obj))
            # Copy MTL and texture if present
            for ext in [".mtl", ".png"]:
                src_aux = src_obj.with_suffix(ext)
                if src_aux.exists():
                    shutil.copy2(str(src_aux), str(output_dir / src_aux.name))

        logger.info(f"InstantMesh output: {dst_obj} ({dst_obj.stat().st_size} bytes)")
        return dst_obj

    def unload_model(self) -> None:
        """No persistent models to unload (subprocess approach)."""
        self._ready = False
        logger.info("InstantMeshImageTo3DService unloaded")
