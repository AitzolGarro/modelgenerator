"""
Job orchestrator: runs pipelines based on job type.

Pipelines:
- generate:     text → image → 3D model → texture → GLB
- animate:      GLB + prompt → animated GLB
- refine:       GLB → refined GLB (more detail)
- scene:        prompt → ground + backdrop + element → scene GLB
- skin:         GLB + prompt → textured GLB with UV-mapped texture
- generate_2d:  text → 2D character image (SDXL) → rembg → part segmentation → model.json
- animate_2d:   2D model.json + prompt → sprite sheet PNG + animation.json
"""

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.job import Job, JobType, JobStatus
from app.services.base import (
    TextToImageService,
    ImageTo3DService,
    TexturingService,
    ExportService,
    AssetStorageService,
    AnimationService,
    MeshRefinementService,
    SceneGenerationService,
    SkinGenerationService,
)

logger = get_logger(__name__)
settings = get_settings()


class JobOrchestrator:
    """Orchestrates all pipeline types."""

    def __init__(
        self,
        text_to_image: TextToImageService,
        image_to_3d: ImageTo3DService,
        texturing: TexturingService,
        export: ExportService,
        storage: AssetStorageService,
        animation: AnimationService,
        refinement: MeshRefinementService,
        scene: SceneGenerationService,
        skin: SkinGenerationService,
        character_2d=None,
        part_segmenter=None,
        animator_2d=None,
        spritesheet_export=None,
    ) -> None:
        self.text_to_image = text_to_image
        self.image_to_3d = image_to_3d
        self.texturing = texturing
        self.export = export
        self.storage = storage
        self.animation = animation
        self.refinement = refinement
        self.scene = scene
        self.skin = skin
        # 2D services
        self.character_2d = character_2d
        self.part_segmenter = part_segmenter
        self.animator_2d = animator_2d
        self.spritesheet_export = spritesheet_export

    def process_job(self, db: Session, job_id: int) -> None:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        if job.status != JobStatus.PENDING:
            logger.warning(f"Job {job_id} not pending ({job.status}), skipping")
            return

        job_type = job.job_type
        logger.info(f"Processing job {job_id} [{job_type}]: '{job.prompt[:60]}'")

        try:
            if job_type == JobType.GENERATE.value:
                self._pipeline_generate(db, job)
            elif job_type == JobType.ANIMATE.value:
                self._pipeline_animate(db, job)
            elif job_type == JobType.REFINE.value:
                self._pipeline_refine(db, job)
            elif job_type == JobType.SCENE.value:
                self._pipeline_scene(db, job)
            elif job_type == JobType.SKIN.value:
                self._pipeline_skin(db, job)
            elif job_type == JobType.GENERATE_2D.value:
                self._pipeline_generate_2d(db, job)
            elif job_type == JobType.ANIMATE_2D.value:
                self._pipeline_animate_2d(db, job)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            job.completed_at = datetime.now(timezone.utc)
            self._update_status(db, job, JobStatus.COMPLETED)
            logger.info(f"Job {job_id} completed")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            job.error_message = str(e)
            job.retry_count += 1
            self._update_status(db, job, JobStatus.FAILED)

    # ── Generate pipeline ────────────────────────────────────

    def _pipeline_generate(self, db: Session, job: Job) -> None:
        # Image
        self._update_status(db, job, JobStatus.GENERATING_IMAGE)
        image = self.text_to_image.generate(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            width=settings.IMAGE_WIDTH,
            height=settings.IMAGE_HEIGHT,
            num_steps=job.num_steps,
            guidance_scale=job.guidance_scale,
            seed=job.seed,
        )
        job.image_path = self.storage.save_image(image, job.id, "reference.png")
        self._update_status(db, job, JobStatus.IMAGE_READY)

        # Free GPU memory before 3D generation (InstantMesh needs ~10GB)
        self.text_to_image.unload_model()
        import torch
        torch.cuda.empty_cache()
        logger.info("Unloaded SDXL to free GPU for 3D generation")

        # 3D model
        self._update_status(db, job, JobStatus.GENERATING_MODEL)
        model_dir = self.storage.get_job_dir(job.id, "models")
        raw_path = self.image_to_3d.generate(image, model_dir)

        # SDXL will auto-reload on next generate() call — no need to reload now
        job.model_path = str(raw_path.relative_to(settings.STORAGE_ROOT))
        self._update_status(db, job, JobStatus.MODEL_READY)

        # Texture
        self._update_status(db, job, JobStatus.TEXTURING)
        textured_path = model_dir / "textured.obj"
        textured = self.texturing.apply_texture(raw_path, image, textured_path)
        job.textured_model_path = str(textured.relative_to(settings.STORAGE_ROOT))
        db.commit()

        # Export
        self._update_status(db, job, JobStatus.EXPORTING)
        export_dir = self.storage.get_job_dir(job.id, "exports")
        source = textured if textured.exists() else raw_path
        exported = self.export.export(source, export_dir / f"model.{settings.EXPORT_FORMAT}", settings.EXPORT_FORMAT)
        job.export_path = str(exported.relative_to(settings.STORAGE_ROOT))

    # ── Animate pipeline ─────────────────────────────────────

    def _pipeline_animate(self, db: Session, job: Job) -> None:
        # Resolve input GLB
        input_glb = self._resolve_input_file(db, job)
        if not input_glb or not input_glb.exists():
            raise FileNotFoundError(f"Input GLB not found: {job.input_file_path}")

        self._update_status(db, job, JobStatus.ANIMATING)
        export_dir = self.storage.get_job_dir(job.id, "exports")
        output_path = export_dir / "animated.glb"

        self.animation.animate(
            glb_path=input_glb,
            prompt=job.prompt,
            output_path=output_path,
            duration=3.0,
            fps=30,
        )

        job.export_path = str(output_path.relative_to(settings.STORAGE_ROOT))

    # ── Refine pipeline ──────────────────────────────────────

    def _pipeline_refine(self, db: Session, job: Job) -> None:
        input_glb = self._resolve_input_file(db, job)
        if not input_glb or not input_glb.exists():
            raise FileNotFoundError(f"Input GLB not found: {job.input_file_path}")

        self._update_status(db, job, JobStatus.REFINING)
        export_dir = self.storage.get_job_dir(job.id, "exports")
        output_path = export_dir / "refined.glb"

        self.refinement.refine(
            glb_path=input_glb,
            output_path=output_path,
            subdivisions=1,
            smooth_iterations=3,
            enhance_normals=True,
        )

        job.export_path = str(output_path.relative_to(settings.STORAGE_ROOT))

    # ── Scene pipeline ───────────────────────────────────────

    def _pipeline_scene(self, db: Session, job: Job) -> None:
        # Generate reference image for preview
        self._update_status(db, job, JobStatus.GENERATING_IMAGE)
        preview = self.text_to_image.generate(
            prompt=f"landscape environment: {job.prompt}, wide angle, cinematic",
            negative_prompt=job.negative_prompt,
            width=1024,
            height=512,
            num_steps=job.num_steps,
            guidance_scale=job.guidance_scale,
            seed=job.seed,
        )
        job.image_path = self.storage.save_image(preview, job.id, "preview.png")
        db.commit()

        # Generate scene
        self._update_status(db, job, JobStatus.GENERATING_SCENE)
        scene_dir = self.storage.get_job_dir(job.id, "models")

        scene_path = self.scene.generate(
            prompt=job.prompt,
            output_dir=scene_dir,
            negative_prompt=job.negative_prompt,
            seed=job.seed,
        )

        # Copy to exports
        self._update_status(db, job, JobStatus.EXPORTING)
        export_dir = self.storage.get_job_dir(job.id, "exports")
        import shutil
        export_path = export_dir / "scene.glb"
        shutil.copy2(str(scene_path), str(export_path))

        job.model_path = str(scene_path.relative_to(settings.STORAGE_ROOT))
        job.export_path = str(export_path.relative_to(settings.STORAGE_ROOT))

    # ── Skin pipeline ────────────────────────────────────────

    def _pipeline_skin(self, db: Session, job: Job) -> None:
        # Resolve input GLB
        input_glb = self._resolve_input_file(db, job)
        if not input_glb or not input_glb.exists():
            raise FileNotFoundError(f"Input GLB not found: {job.input_file_path}")

        # Load reference image if available (from a prior generate job)
        reference_image = None
        if job.image_path:
            try:
                ref_path = self.storage.get_absolute_path(job.image_path)
                if ref_path.exists():
                    reference_image = Image.open(ref_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not load reference image: {e}")

        self._update_status(db, job, JobStatus.GENERATING_SKIN)
        export_dir = self.storage.get_job_dir(job.id, "exports")
        output_path = export_dir / "textured.glb"

        self.skin.generate_skin(
            glb_path=input_glb,
            prompt=job.prompt,
            output_path=output_path,
            reference_image=reference_image,
        )

        job.export_path = str(output_path.relative_to(settings.STORAGE_ROOT))

    # ── Generate 2D pipeline ─────────────────────────────────

    def _pipeline_generate_2d(self, db: Session, job: Job) -> None:
        """text → 2D character → rembg → part segmentation → model.json"""
        if self.character_2d is None or self.part_segmenter is None:
            raise RuntimeError("2D services not initialized in orchestrator")

        style = job.style or settings.STYLE_2D_DEFAULT

        # 1. Generate 2D character image
        self._update_status(db, job, JobStatus.GENERATING_IMAGE)
        logger.info(f"Job {job.id}: generating 2D character (style={style})")
        image_rgba = self.character_2d.generate(
            prompt=job.prompt,
            style=style,
            num_steps=job.num_steps,
            guidance_scale=job.guidance_scale,
            seed=job.seed,
            negative_prompt=job.negative_prompt,
        )

        # Save full character preview (RGBA PNG)
        job.image_path = self.storage.save_image(image_rgba, job.id, "character.png")
        self._update_status(db, job, JobStatus.IMAGE_READY)

        # 2. Segment into body parts
        self._update_status(db, job, JobStatus.SEGMENTING)
        logger.info(f"Job {job.id}: segmenting into body parts")
        parts_dir = self.storage.get_job_dir(job.id, "models")
        model_dict = self.part_segmenter.segment(image_rgba, parts_dir)

        # Store model.json path
        import json
        model_json_path = parts_dir / "model.json"
        # Already written by part_segmenter; store relative path
        job.model_path = str(model_json_path.relative_to(settings.STORAGE_ROOT))
        job.model_json_path = str(model_json_path.relative_to(settings.STORAGE_ROOT))
        db.commit()

        # 3. Export full character PNG as the "export" (for preview/download)
        export_dir = self.storage.get_job_dir(job.id, "exports")
        import shutil
        char_export = export_dir / "character.png"
        shutil.copy2(str(parts_dir / "character.png"), str(char_export))
        job.export_path = str(char_export.relative_to(settings.STORAGE_ROOT))
        db.commit()

        logger.info(f"Job {job.id}: generate_2d complete — {len(model_dict.get('parts', []))} parts")

    # ── Animate 2D pipeline ──────────────────────────────────

    def _pipeline_animate_2d(self, db: Session, job: Job) -> None:
        """2D model.json + prompt → sprite sheet PNG + animation.json"""
        if self.animator_2d is None:
            raise RuntimeError("animator_2d service not initialized in orchestrator")

        # Resolve source generate_2d job
        source_job_id = self._parse_2d_source_job_id(job)
        source_job = db.query(Job).filter(Job.id == source_job_id).first()
        if not source_job:
            raise ValueError(f"Source 2D job {source_job_id} not found")
        if not source_job.model_json_path:
            raise ValueError(f"Source 2D job {source_job_id} has no model.json")

        # Load model.json
        import json
        model_json_abs = settings.STORAGE_ROOT / source_job.model_json_path
        with open(model_json_abs) as f:
            model_dict = json.load(f)

        parts_dir = model_json_abs.parent

        # Generate sprite sheet
        self._update_status(db, job, JobStatus.GENERATING_SPRITE_SHEET)
        logger.info(f"Job {job.id}: generating sprite sheet from job {source_job_id}")

        output_dir = self.storage.get_job_dir(job.id, "exports")
        sprite_path, meta_path = self.animator_2d.animate(
            model_json=model_dict,
            parts_dir=parts_dir,
            prompt=job.prompt,
            output_dir=output_dir,
        )

        job.sprite_sheet_path = str(sprite_path.relative_to(settings.STORAGE_ROOT))
        job.export_path = str(sprite_path.relative_to(settings.STORAGE_ROOT))
        job.model_json_path = str(meta_path.relative_to(settings.STORAGE_ROOT))
        # Image path: save a copy of the sprite sheet for preview
        job.image_path = str(sprite_path.relative_to(settings.STORAGE_ROOT))
        db.commit()

        logger.info(f"Job {job.id}: animate_2d complete — sprite sheet at {sprite_path}")

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_2d_source_job_id(job: Job) -> int:
        """Extract source job ID stored in input_file_path for animate_2d jobs."""
        if job.input_file_path and job.input_file_path.startswith("__2d_source_job:"):
            return int(job.input_file_path.split(":")[1])
        raise ValueError(
            f"animate_2d job {job.id} has no valid 2D source job reference "
            f"(input_file_path={job.input_file_path!r})"
        )

    def _resolve_input_file(self, db: Session, job: Job) -> Path | None:
        """Resolve the input GLB file for animate/refine jobs."""
        if job.input_file_path:
            return self.storage.get_absolute_path(job.input_file_path)
        return None

    def _update_status(self, db: Session, job: Job, status: JobStatus) -> None:
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"Job {job.id} → {status.value}")
