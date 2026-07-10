"""FasterGS/DeferredShading.py

Deferred reflection shading (3DGS-DR, Ye et al. SIGGRAPH 2024). After the Gaussian
blend produces screen-space base color, normal, and reflection-strength maps, this
module queries a learned environment cubemap along the per-pixel reflection direction
and composes the final image::

    C' = (1 - R) * C + R * E(reflect(v, N))

The environment map is a small learnable cubemap texture optimized end-to-end from
the image loss; it is detached from the Gaussian blend weights (the gradient reaches it
only through the blended maps), matching the paper's deferred design.
"""

from __future__ import annotations

import torch

from Datasets.utils import View
from Methods.FasterGS.FasterGSCudaBackend import CubemapEncoder


def reflect(view_directions: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Reflect view directions about per-pixel normals: ``d - 2 (d . n) n``.

    ``view_directions`` and ``normals`` are (3, H, W); returns (3, H, W).
    """
    dot = (view_directions * normals).sum(dim=0, keepdim=True)
    return view_directions - 2.0 * dot * normals


class EnvironmentMap(torch.nn.Module):
    """Learnable environment map backed by a cubemap encoder (queried in [0, 1])."""

    def __init__(self, resolution: int = 256, interpolation: str = 'linear') -> None:
        super().__init__()
        self.encoder = CubemapEncoder(output_dim=3, resolution=resolution, interpolation=interpolation)

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Query the env map at unit ``directions`` (N, 3), returning colors in [0, 1] (N, 3)."""
        return torch.sigmoid(self.encoder(directions))

    def query_image(self, directions_chw: torch.Tensor) -> torch.Tensor:
        """Query along a (3, H, W) direction map, returning a (3, H, W) color map."""
        _, h, w = directions_chw.shape
        dirs = directions_chw.permute(1, 2, 0).reshape(-1, 3).contiguous()
        colors = self.forward(dirs)
        return colors.reshape(h, w, 3).permute(2, 0, 1).contiguous()


def world_ray_directions(view: View) -> torch.Tensor:
    """Per-pixel world-space ray directions from the camera through pixel centers (3, H, W)."""
    local = view.camera.compute_local_ray_directions().to(device=view.rotation.device, dtype=torch.float32)  # (H*W, 3)
    world = local @ view.rotation.T  # camera-to-world rotation
    world = torch.nn.functional.normalize(world, dim=-1)
    return world.reshape(view.camera.height, view.camera.width, 3).permute(2, 0, 1).contiguous()


def compose_deferred(
    base_color: torch.Tensor,
    feature_map: torch.Tensor,
    view: View,
    environment: EnvironmentMap,
) -> dict[str, torch.Tensor]:
    """Compose the final deferred-reflection image.

    Args:
        base_color: (3, H, W) blended SH base color.
        feature_map: (4, H, W) blended (normal.xyz, reflection strength).
        view: the rendering view (for per-pixel ray directions).
        environment: learnable environment map.

    Returns a dict with ``rgb`` and the decomposition maps for debugging.
    """
    normal_map = feature_map[:3]
    refl_strength = feature_map[3:4].clamp(0.0, 1.0)
    normals = torch.nn.functional.normalize(normal_map, dim=0, eps=1e-6)
    view_dirs = world_ray_directions(view)
    reflection_dirs = reflect(view_dirs, normals)
    reflection_dirs = torch.nn.functional.normalize(reflection_dirs, dim=0, eps=1e-6)
    reflection_color = environment.query_image(reflection_dirs)
    final = (1.0 - refl_strength) * base_color + refl_strength * reflection_color
    return {
        'rgb': final,
        'base_color': base_color,
        'reflection_color': reflection_color,
        'reflection_strength': refl_strength,
        'normal': normals,
    }
