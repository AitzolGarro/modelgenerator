"""
SQLAlchemy model for generation jobs.
Supports multiple job types: generate, animate, refine, scene.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import String, Text, Integer, Float, DateTime, Enum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class JobType(str, enum.Enum):
    GENERATE = "generate"       # text → image → 3D → GLB
    ANIMATE = "animate"         # GLB + prompt → animated GLB
    REFINE = "refine"           # GLB → refined GLB (more detail)
    SCENE = "scene"             # prompt → full scene/environment
    SKIN = "skin"               # GLB + prompt → textured GLB with UV map
    GENERATE_2D = "generate_2d"  # text → 2D character image → parts
    ANIMATE_2D = "animate_2d"    # 2D model + prompt → sprite sheet


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    # Generate pipeline
    GENERATING_IMAGE = "generating_image"
    IMAGE_READY = "image_ready"
    GENERATING_MODEL = "generating_model"
    MODEL_READY = "model_ready"
    TEXTURING = "texturing"
    # Animate pipeline
    RIGGING = "rigging"
    ANIMATING = "animating"
    # Refine pipeline
    REFINING = "refining"
    # Scene pipeline
    GENERATING_SCENE = "generating_scene"
    COMPOSITING = "compositing"
    # Skin pipeline
    GENERATING_SKIN = "generating_skin"
    # 2D pipelines
    SEGMENTING = "segmenting"                          # separating character into parts
    GENERATING_SPRITE_SHEET = "generating_sprite_sheet"
    # Shared
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(
        String(32), default=JobType.GENERATE.value, nullable=False
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, nullable=False
    )

    # Input file (for animate/refine: the source GLB)
    input_file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Output file paths (relative to storage root)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    textured_model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    export_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 2D-specific fields
    style: Mapped[str | None] = mapped_column(String(32), nullable=True)   # anime, pixel_art, …
    sprite_sheet_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_json_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

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
        return f"<Job id={self.id} type={self.job_type} status={self.status}>"
