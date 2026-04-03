"""
Base service interfaces.
All ML services implement these ABCs so they can be swapped out easily.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class TextToImageService(ABC):
    """Generates a reference image from a text prompt."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory/GPU."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate an image and return as PIL Image."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Free GPU memory."""
        ...


class ImageTo3DService(ABC):
    """Converts a reference image to a 3D mesh."""

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def generate(self, image: Image.Image, output_dir: Path) -> Path:
        """Generate 3D model from image. Returns path to output file."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        ...


class TexturingService(ABC):
    """Applies texture to a 3D mesh."""

    @abstractmethod
    def apply_texture(
        self,
        mesh_path: Path,
        reference_image: Image.Image,
        output_path: Path,
    ) -> Path:
        """Apply texture and return path to textured model."""
        ...


class ExportService(ABC):
    """Exports 3D models to various formats."""

    @abstractmethod
    def export(
        self,
        input_path: Path,
        output_path: Path,
        format: str = "glb",
    ) -> Path:
        """Export model to target format. Returns output path."""
        ...


class AssetStorageService(ABC):
    """Manages file storage for generated assets."""

    @abstractmethod
    def save_image(self, image: Image.Image, job_id: int, filename: str) -> str:
        """Save image and return relative path."""
        ...

    @abstractmethod
    def save_model(self, source_path: Path, job_id: int, filename: str) -> str:
        """Save/move model file and return relative path."""
        ...

    @abstractmethod
    def get_absolute_path(self, relative_path: str) -> Path:
        """Resolve relative path to absolute."""
        ...

    @abstractmethod
    def get_job_dir(self, job_id: int, category: str) -> Path:
        """Get or create job-specific directory."""
        ...
