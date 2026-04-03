"""
Asset storage service: manages file storage for generated assets.
"""

import shutil
from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.base import AssetStorageService

logger = get_logger(__name__)
settings = get_settings()


class LocalAssetStorageService(AssetStorageService):
    """Filesystem-based asset storage."""

    def __init__(self) -> None:
        settings.ensure_dirs()

    def save_image(self, image: Image.Image, job_id: int, filename: str) -> str:
        job_dir = self.get_job_dir(job_id, "images")
        filepath = job_dir / filename
        image.save(str(filepath), quality=95)
        # Return relative path from storage root
        relative = filepath.relative_to(settings.STORAGE_ROOT)
        logger.info(f"Image saved: {relative}")
        return str(relative)

    def save_model(self, source_path: Path, job_id: int, filename: str) -> str:
        job_dir = self.get_job_dir(job_id, "models")
        dest = job_dir / filename

        if source_path != dest:
            shutil.copy2(str(source_path), str(dest))

        relative = dest.relative_to(settings.STORAGE_ROOT)
        logger.info(f"Model saved: {relative}")
        return str(relative)

    def save_export(self, source_path: Path, job_id: int, filename: str) -> str:
        job_dir = self.get_job_dir(job_id, "exports")
        dest = job_dir / filename

        if source_path != dest:
            shutil.copy2(str(source_path), str(dest))

        relative = dest.relative_to(settings.STORAGE_ROOT)
        logger.info(f"Export saved: {relative}")
        return str(relative)

    def get_absolute_path(self, relative_path: str) -> Path:
        return settings.STORAGE_ROOT / relative_path

    def get_job_dir(self, job_id: int, category: str) -> Path:
        """Get or create job-specific directory under a category."""
        path = settings.STORAGE_ROOT / category / str(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path
