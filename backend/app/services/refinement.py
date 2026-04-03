"""
Mesh refinement service: improves detail of 3D models.

Techniques applied:
1. Loop subdivision (adds geometry detail)
2. Laplacian smoothing (removes noise while preserving shape)
3. Normal recalculation (fixes lighting artifacts)
4. Vertex merging (cleans up duplicate vertices)
5. Optional decimation (reduce poly count while preserving detail)

For production, consider:
- Neural mesh refinement (DMTet, FlexiCubes)
- AI-based texture upscaling (Real-ESRGAN on UV maps)
"""

from pathlib import Path

import numpy as np
import trimesh

from app.core.logging import get_logger
from app.services.base import MeshRefinementService

logger = get_logger(__name__)


class TrimeshRefinementService(MeshRefinementService):
    """
    Mesh refinement using trimesh operations.
    Applies subdivision, smoothing, and normal enhancement.
    """

    def refine(
        self,
        glb_path: Path,
        output_path: Path,
        subdivisions: int = 1,
        smooth_iterations: int = 3,
        enhance_normals: bool = True,
    ) -> Path:
        logger.info(f"Refining mesh: {glb_path.name}")
        logger.info(f"  subdivisions={subdivisions}, smooth={smooth_iterations}, normals={enhance_normals}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.load(str(glb_path))

        if isinstance(scene, trimesh.Scene):
            refined_geometries = {}
            for name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    refined = self._refine_mesh(
                        geom, subdivisions, smooth_iterations, enhance_normals
                    )
                    refined_geometries[name] = refined
                else:
                    refined_geometries[name] = geom

            for name, geom in refined_geometries.items():
                scene.geometry[name] = geom

            glb_data = scene.export(file_type="glb")
        elif isinstance(scene, trimesh.Trimesh):
            refined = self._refine_mesh(
                scene, subdivisions, smooth_iterations, enhance_normals
            )
            export_scene = trimesh.Scene(geometry={"model": refined})
            glb_data = export_scene.export(file_type="glb")
        else:
            logger.warning(f"Unexpected type: {type(scene)}, copying as-is")
            import shutil
            shutil.copy2(str(glb_path), str(output_path))
            return output_path

        output_path.write_bytes(glb_data)

        logger.info(f"Refined mesh saved: {output_path}")
        return output_path

    def _refine_mesh(
        self,
        mesh: trimesh.Trimesh,
        subdivisions: int,
        smooth_iterations: int,
        enhance_normals: bool,
    ) -> trimesh.Trimesh:
        """Apply refinement operations to a single mesh."""
        original_faces = len(mesh.faces)
        original_verts = len(mesh.vertices)
        logger.info(f"  Original: {original_verts} verts, {original_faces} faces")

        # Step 1: Merge duplicate vertices
        mesh.merge_vertices()

        # Step 2: Remove degenerate faces
        mesh.remove_degenerate_faces()

        # Step 3: Subdivision (Loop subdivision for triangle meshes)
        for i in range(subdivisions):
            try:
                # trimesh subdivision
                mesh = mesh.subdivide()
                logger.info(f"  After subdivision {i+1}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

                # Cap at reasonable limit to avoid OOM
                if len(mesh.faces) > 500_000:
                    logger.warning(f"  Face count exceeds 500k, stopping subdivision")
                    break
            except Exception as e:
                logger.warning(f"  Subdivision {i+1} failed: {e}")
                break

        # Step 4: Laplacian smoothing
        if smooth_iterations > 0:
            try:
                smoothed = trimesh.smoothing.filter_laplacian(
                    mesh,
                    iterations=smooth_iterations,
                    lamb=0.5,  # smoothing factor
                )
                if smoothed is not None:
                    mesh = smoothed
                    logger.info(f"  Laplacian smoothing applied ({smooth_iterations} iterations)")
            except Exception as e:
                logger.warning(f"  Smoothing failed: {e}")

        # Step 5: Fix normals
        if enhance_normals:
            try:
                mesh.fix_normals()
                # Recompute vertex normals from face normals
                mesh.vertex_normals  # triggers recomputation
                logger.info("  Normals recalculated")
            except Exception as e:
                logger.warning(f"  Normal fix failed: {e}")

        # Step 6: Fill holes (small ones)
        try:
            if not mesh.is_watertight:
                mesh.fill_holes()
                logger.info(f"  Holes filled, watertight: {mesh.is_watertight}")
        except Exception:
            pass

        logger.info(f"  Final: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return mesh
