"""
Export service: converts 3D models between formats.
Prioritizes GLB for web viewing.
"""

from pathlib import Path

import trimesh

from app.core.logging import get_logger
from app.services.base import ExportService

logger = get_logger(__name__)


class TrimeshExportService(ExportService):
    """Export 3D models using trimesh."""

    def export(
        self,
        input_path: Path,
        output_path: Path,
        format: str = "glb",
    ) -> Path:
        logger.info(f"Exporting {input_path} to {format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure correct extension
        ext = f".{format.lower()}"
        if output_path.suffix != ext:
            output_path = output_path.with_suffix(ext)

        mesh = trimesh.load(str(input_path))

        if format.lower() == "glb":
            # GLB export (binary glTF)
            if isinstance(mesh, trimesh.Scene):
                glb_data = mesh.export(file_type="glb")
            else:
                scene = trimesh.Scene(geometry={"model": mesh})
                glb_data = scene.export(file_type="glb")

            output_path.write_bytes(glb_data)

        elif format.lower() == "obj":
            mesh.export(str(output_path), file_type="obj")

        elif format.lower() == "stl":
            if isinstance(mesh, trimesh.Scene):
                combined = trimesh.util.concatenate(list(mesh.geometry.values()))
                combined.export(str(output_path), file_type="stl")
            else:
                mesh.export(str(output_path), file_type="stl")

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported to {output_path} ({output_path.stat().st_size} bytes)")
        return output_path
