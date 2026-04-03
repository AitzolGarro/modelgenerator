"""
Background worker that runs inside the FastAPI process.
Spawns a daemon thread that polls for pending jobs.
"""

import threading

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.database import SessionLocal
from app.models.job import Job, JobStatus
from app.services.factory import (
    create_text_to_image_service,
    create_image_to_3d_service,
    create_texturing_service,
    create_export_service,
    create_storage_service,
    create_animation_service,
    create_refinement_service,
    create_scene_service,
    create_skin_service,
)
from app.workers.orchestrator import JobOrchestrator

logger = get_logger(__name__)
settings = get_settings()

_worker_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _get_next_pending_job(db) -> Job | None:
    return (
        db.query(Job)
        .filter(Job.status == JobStatus.PENDING)
        .order_by(Job.created_at.asc())
        .first()
    )


def _worker_loop() -> None:
    logger.info("Background worker: initializing services...")

    text_to_image = create_text_to_image_service()
    image_to_3d = create_image_to_3d_service()
    texturing = create_texturing_service()
    export = create_export_service()
    storage = create_storage_service()
    animation = create_animation_service()
    refinement = create_refinement_service()
    scene = create_scene_service(text_to_image, image_to_3d)
    skin = create_skin_service(text_to_image)

    logger.info("Background worker: preloading ML models...")
    text_to_image.load_model()
    image_to_3d.load_model()
    animation.load_model()
    scene.load_model()
    skin.load_model()
    logger.info("Background worker: ready, polling for jobs.")

    orchestrator = JobOrchestrator(
        text_to_image=text_to_image,
        image_to_3d=image_to_3d,
        texturing=texturing,
        export=export,
        storage=storage,
        animation=animation,
        refinement=refinement,
        scene=scene,
        skin=skin,
    )

    while not _stop_event.is_set():
        db = SessionLocal()
        try:
            job = _get_next_pending_job(db)
            if job:
                logger.info(f"Background worker: processing job {job.id} [{job.job_type}]")
                orchestrator.process_job(db, job.id)
            else:
                _stop_event.wait(timeout=settings.WORKER_POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Background worker error: {e}", exc_info=True)
            _stop_event.wait(timeout=settings.WORKER_POLL_INTERVAL)
        finally:
            db.close()

    logger.info("Background worker: shutting down...")
    text_to_image.unload_model()
    image_to_3d.unload_model()
    logger.info("Background worker: stopped.")


def start_worker() -> None:
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _stop_event.clear()
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="ml-worker")
    _worker_thread.start()
    logger.info("Background worker thread started.")


def stop_worker() -> None:
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        return
    logger.info("Stopping background worker...")
    _stop_event.set()
    _worker_thread.join(timeout=30)
    _worker_thread = None
    logger.info("Background worker stopped.")
