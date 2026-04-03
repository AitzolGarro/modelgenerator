"""
Job orchestrator: runs the full generation pipeline for a job.

Pipeline:
1. pending → generating_image
2. generating_image → image_ready
3. image_ready → generating_model
4. generating_model → model_ready
5. model_ready → texturing
6. texturing → exporting
7. exporting → completed

On any failure: → failed
"""

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.job import Job, JobStatus
from app.services.base import (
    TextToImageService,
    ImageTo3DService,
    TexturingService,
    ExportService,
    AssetStorageService,
)

logger = get_logger(__name__)
settings = get_settings()


class JobOrchestrator:
    """
    Orchestrates the full generation pipeline for a single job.
    Each step updates the job status in the database.
    """

    def __init__(
        self,
        text_to_image: TextToImageService,
        image_to_3d: ImageTo3DService,
        texturing: TexturingService,
        export: ExportService,
        storage: AssetStorageService,
    ) -> None:
        self.text_to_image = text_to_image
        self.image_to_3d = image_to_3d
        self.texturing = texturing
        self.export = export
        self.storage = storage

    def process_job(self, db: Session, job_id: int) -> None:
        """Run the full pipeline for a job."""
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        if job.status != JobStatus.PENDING:
            logger.warning(f"Job {job_id} is not pending (status={job.status}), skipping")
            return

        logger.info(f"Starting pipeline for job {job_id}: '{job.prompt[:60]}'")

        try:
            # Step 1: Generate image
            self._update_status(db, job, JobStatus.GENERATING_IMAGE)
            image = self._generate_image(job)
            image_path = self.storage.save_image(image, job.id, "reference.png")
            job.image_path = image_path
            self._update_status(db, job, JobStatus.IMAGE_READY)

            # Step 2: Generate 3D model
            self._update_status(db, job, JobStatus.GENERATING_MODEL)
            model_dir = self.storage.get_job_dir(job.id, "models")
            raw_model_path = self.image_to_3d.generate(image, model_dir)
            model_rel = raw_model_path.relative_to(settings.STORAGE_ROOT)
            job.model_path = str(model_rel)
            self._update_status(db, job, JobStatus.MODEL_READY)

            # Step 3: Texturing
            self._update_status(db, job, JobStatus.TEXTURING)
            textured_path = model_dir / "textured.obj"
            textured_result = self.texturing.apply_texture(
                raw_model_path, image, textured_path
            )
            textured_rel = textured_result.relative_to(settings.STORAGE_ROOT)
            job.textured_model_path = str(textured_rel)
            db.commit()

            # Step 4: Export to GLB
            self._update_status(db, job, JobStatus.EXPORTING)
            export_dir = self.storage.get_job_dir(job.id, "exports")
            export_path = export_dir / f"model.{settings.EXPORT_FORMAT}"

            # Export from the best available source
            source = textured_result if textured_result.exists() else raw_model_path
            exported = self.export.export(source, export_path, settings.EXPORT_FORMAT)
            export_rel = exported.relative_to(settings.STORAGE_ROOT)
            job.export_path = str(export_rel)

            # Done!
            job.completed_at = datetime.now(timezone.utc)
            self._update_status(db, job, JobStatus.COMPLETED)
            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            job.error_message = str(e)
            job.retry_count += 1
            self._update_status(db, job, JobStatus.FAILED)

    def _generate_image(self, job: Job) -> Image.Image:
        """Generate reference image from prompt."""
        return self.text_to_image.generate(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            width=settings.IMAGE_WIDTH,
            height=settings.IMAGE_HEIGHT,
            num_steps=job.num_steps,
            guidance_scale=job.guidance_scale,
            seed=job.seed,
        )

    def _update_status(self, db: Session, job: Job, status: JobStatus) -> None:
        """Update job status and commit."""
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"Job {job.id} → {status.value}")
