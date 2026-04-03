"""
SQLAlchemy model for generation jobs.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import String, Text, Integer, Float, DateTime, Enum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    GENERATING_IMAGE = "generating_image"
    IMAGE_READY = "image_ready"
    GENERATING_MODEL = "generating_model"
    MODEL_READY = "model_ready"
    TEXTURING = "texturing"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, nullable=False
    )

    # File paths (relative to storage root)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    textured_model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    export_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Metadata
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    num_steps: Mapped[int] = mapped_column(Integer, default=30)
    guidance_scale: Mapped[float] = mapped_column(Float, default=7.5)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Job id={self.id} status={self.status} prompt='{self.prompt[:40]}...'>"
