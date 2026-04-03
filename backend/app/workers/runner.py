"""
Worker runner: polls the database for pending jobs and processes them.

Run this as a separate process:
    python -m app.workers.runner
"""

import time
import signal
import sys

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.db.database import SessionLocal, init_db
from app.models.job import Job, JobStatus
from app.services.factory import (
    create_text_to_image_service,
    create_image_to_3d_service,
    create_texturing_service,
    create_export_service,
    create_storage_service,
)
from app.workers.orchestrator import JobOrchestrator

settings = get_settings()
logger = get_logger(__name__)

_running = True


def signal_handler(signum, frame):
    global _running
    logger.info("Received shutdown signal, finishing current job...")
    _running = False


def get_next_pending_job(db: Session) -> Job | None:
    """Get the oldest pending job."""
    return (
        db.query(Job)
        .filter(Job.status == JobStatus.PENDING)
        .order_by(Job.created_at.asc())
        .first()
    )


def run_worker() -> None:
    """Main worker loop."""
    setup_logging(settings.DEBUG)
    logger.info("=" * 60)
    logger.info("ModelGenerator Worker starting...")
    logger.info(f"Poll interval: {settings.WORKER_POLL_INTERVAL}s")
    logger.info("=" * 60)

    # Initialize database
    init_db()

    # Create services
    logger.info("Initializing services...")
    text_to_image = create_text_to_image_service()
    image_to_3d = create_image_to_3d_service()
    texturing = create_texturing_service()
    export = create_export_service()
    storage = create_storage_service()

    # Preload models
    logger.info("Preloading ML models...")
    text_to_image.load_model()
    image_to_3d.load_model()
    logger.info("Models loaded, worker ready!")

    # Create orchestrator
    orchestrator = JobOrchestrator(
        text_to_image=text_to_image,
        image_to_3d=image_to_3d,
        texturing=texturing,
        export=export,
        storage=storage,
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    while _running:
        db = SessionLocal()
        try:
            job = get_next_pending_job(db)
            if job:
                logger.info(f"Found pending job {job.id}")
                orchestrator.process_job(db, job.id)
            else:
                time.sleep(settings.WORKER_POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(settings.WORKER_POLL_INTERVAL)
        finally:
            db.close()

    # Cleanup
    logger.info("Shutting down worker...")
    text_to_image.unload_model()
    image_to_3d.unload_model()
    logger.info("Worker stopped.")


if __name__ == "__main__":
    run_worker()
