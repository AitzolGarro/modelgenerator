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
    def load_model(self) -> None: ...

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
    ) -> Image.Image: ...

    @abstractmethod
    def unload_model(self) -> None: ...


class ImageTo3DService(ABC):
    """Converts a reference image to a 3D mesh."""

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def generate(self, image: Image.Image, output_dir: Path) -> Path: ...

    @abstractmethod
    def unload_model(self) -> None: ...


class TexturingService(ABC):
    """Applies texture to a 3D mesh."""

    @abstractmethod
    def apply_texture(
        self, mesh_path: Path, reference_image: Image.Image, output_path: Path,
    ) -> Path: ...


class ExportService(ABC):
    """Exports 3D models to various formats."""

    @abstractmethod
    def export(self, input_path: Path, output_path: Path, format: str = "glb") -> Path: ...


class AssetStorageService(ABC):
    """Manages file storage for generated assets."""

    @abstractmethod
    def save_image(self, image: Image.Image, job_id: int, filename: str) -> str: ...

    @abstractmethod
    def save_model(self, source_path: Path, job_id: int, filename: str) -> str: ...

    @abstractmethod
    def get_absolute_path(self, relative_path: str) -> Path: ...

    @abstractmethod
    def get_job_dir(self, job_id: int, category: str) -> Path: ...


# ── New services ─────────────────────────────────────────────


class AnimationService(ABC):
    """Generates skeletal animation for a 3D model from a text prompt."""

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def animate(
        self,
        glb_path: Path,
        prompt: str,
        output_path: Path,
        duration: float = 3.0,
        fps: int = 30,
    ) -> Path:
        """
        Add animation to a GLB model based on a text description.
        Returns path to the animated GLB.
        """
        ...

    @abstractmethod
    def unload_model(self) -> None: ...


class MeshRefinementService(ABC):
    """Improves mesh detail: subdivision, smoothing, normal enhancement."""

    @abstractmethod
    def refine(
        self,
        glb_path: Path,
        output_path: Path,
        subdivisions: int = 1,
        smooth_iterations: int = 3,
        enhance_normals: bool = True,
    ) -> Path:
        """
        Refine a mesh to improve detail.
        Returns path to the refined GLB.
        """
        ...


class SceneGenerationService(ABC):
    """Generates a 3D scene/environment from a text prompt."""

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_dir: Path,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Path:
        """
        Generate a scene/environment as a 3D model.
        Returns path to the output GLB.
        """
        ...

    @abstractmethod
    def unload_model(self) -> None: ...
