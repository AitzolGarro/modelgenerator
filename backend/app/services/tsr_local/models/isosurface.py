"""
Isosurface extraction using marching cubes.
Patched to use scikit-image instead of torchmcubes (which doesn't build on Python 3.14).
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _skimage_marching_cubes(volume: torch.Tensor, isolevel: float):
    """
    Drop-in replacement for torchmcubes.marching_cubes using scikit-image.
    Returns (vertices, faces) as torch tensors matching torchmcubes interface.
    """
    from skimage.measure import marching_cubes

    vol_np = volume.detach().cpu().numpy()

    # Clamp isolevel to be within the volume data range
    vmin, vmax = vol_np.min(), vol_np.max()
    if isolevel <= vmin or isolevel >= vmax:
        # Use midpoint of the range as fallback
        isolevel = (vmin + vmax) / 2.0

    verts, faces, _, _ = marching_cubes(vol_np, level=isolevel)

    return (
        torch.from_numpy(verts.copy()).float(),
        torch.from_numpy(faces.copy()).long(),
    )


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = _skimage_marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
