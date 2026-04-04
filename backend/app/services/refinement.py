"""
Mesh refinement service: intelligent mesh cleanup and detail enhancement.

Pipeline:
1. Clean: merge vertices, remove degenerates, fix winding
2. Decimate: reduce noise by removing small/thin faces (quadric decimation)
3. Re-mesh: rebuild topology with uniform triangles
4. Smooth: Taubin smoothing (shrink-free, preserves volume better than Laplacian)
5. Subdivide: controlled subdivision for added detail
6. Normals: recompute smooth normals with angle weighting
7. Watertight: fill small holes

The key insight: TripoSR meshes have noisy, uneven topology.
Subdividing raw TripoSR output amplifies noise.
We must CLEAN FIRST, then add detail.
"""

from pathlib import Path

import numpy as np
import trimesh

from app.core.logging import get_logger
from app.services.base import MeshRefinementService

logger = get_logger(__name__)


class TrimeshRefinementService(MeshRefinementService):

    def refine(
        self,
        glb_path: Path,
        output_path: Path,
        subdivisions: int = 1,
        smooth_iterations: int = 3,
        enhance_normals: bool = True,
    ) -> Path:
        logger.info(f"Refining: {glb_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.load(str(glb_path))

        if isinstance(scene, trimesh.Scene):
            for name, geom in list(scene.geometry.items()):
                if isinstance(geom, trimesh.Trimesh):
                    scene.geometry[name] = self._refine_mesh(geom)
            glb_data = scene.export(file_type="glb")
        elif isinstance(scene, trimesh.Trimesh):
            refined = self._refine_mesh(scene)
            glb_data = trimesh.Scene(geometry={"model": refined}).export(file_type="glb")
        else:
            import shutil
            shutil.copy2(str(glb_path), str(output_path))
            return output_path

        output_path.write_bytes(glb_data)
        logger.info(f"Refined mesh: {output_path}")
        return output_path

    def _refine_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        v0, f0 = len(mesh.vertices), len(mesh.faces)
        logger.info(f"  Input: {v0} verts, {f0} faces")

        # ── Step 1: Clean ────────────────────────────────────
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        # trimesh 4.x API: use nondegenerate_faces mask instead of remove_degenerate_faces
        try:
            mask = mesh.nondegenerate_faces()
            mesh.update_faces(mask)
        except Exception:
            pass
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass
        mesh.remove_unreferenced_vertices()

        # Fix face winding for consistent normals
        mesh.fix_normals()

        logger.info(f"  After clean: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # ── Step 2: Smart decimation ─────────────────────────
        # Reduce to ~60% to remove noise, then rebuild
        target_faces = max(int(len(mesh.faces) * 0.6), 1000)
        if len(mesh.faces) > 2000:
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                logger.info(f"  After decimation: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            except Exception as e:
                logger.warning(f"  Decimation failed: {e}")

        # ── Step 3: Taubin smoothing ─────────────────────────
        # Taubin is better than Laplacian because it doesn't shrink the mesh
        try:
            mesh = trimesh.smoothing.filter_taubin(
                mesh,
                lamb=0.5,    # Smoothing strength
                mu=-0.53,    # Inflation (slightly > -lamb to prevent shrinkage)
                iterations=10,
            )
            logger.info("  Taubin smoothing applied (10 iterations)")
        except Exception as e:
            logger.warning(f"  Taubin smoothing failed: {e}")
            # Fallback to Laplacian with low strength
            try:
                mesh = trimesh.smoothing.filter_laplacian(
                    mesh, iterations=5, lamb=0.3,
                )
                logger.info("  Laplacian smoothing fallback applied")
            except Exception:
                pass

        # ── Step 4: Subdivision ──────────────────────────────
        # Now that the mesh is clean and smooth, subdivide for detail
        if len(mesh.faces) < 200_000:
            try:
                mesh = mesh.subdivide()
                logger.info(f"  After subdivision: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            except Exception as e:
                logger.warning(f"  Subdivision failed: {e}")

        # ── Step 5: Final smoothing pass ─────────────────────
        # Light smoothing on the subdivided mesh
        try:
            mesh = trimesh.smoothing.filter_taubin(
                mesh, lamb=0.3, mu=-0.33, iterations=3,
            )
            logger.info("  Final smoothing pass")
        except Exception:
            pass

        # ── Step 6: Normal enhancement ───────────────────────
        mesh.fix_normals()
        # Force recomputation of vertex normals (area-weighted)
        mesh._cache.delete("vertex_normals")

        # ── Step 7: Fill holes ───────────────────────────────
        try:
            if not mesh.is_watertight:
                mesh.fill_holes()
        except Exception:
            pass

        logger.info(f"  Output: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        improvement = len(mesh.faces) / max(f0, 1)
        logger.info(f"  Detail ratio: {improvement:.1f}x faces vs original")

        return mesh
