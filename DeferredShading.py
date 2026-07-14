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

import numpy as np
import torch

import Framework
from Datasets.utils import View
from Methods.FasterGS.FasterGSCudaBackend import CubemapEncoder
from Methods.FasterGS.envmap_directions import prepare_envmap_directions


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


def turntable_reflection_rotation(view: View) -> torch.Tensor | None:
    """Per-view object rotation for turntable captures (3DGS-DR ``turntable_R`` parity)."""
    transform = view.exif.get('turntable_object_transform')
    if transform is None:
        return None
    rotation = np.asarray(transform, dtype=np.float32)[:3, :3]
    return torch.as_tensor(rotation, dtype=torch.float32, device=view.rotation.device)


def apply_turntable_to_directions(directions_chw: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate per-pixel directions (3, H, W) by ``rotation`` on the right (row-vector convention)."""
    _, height, width = directions_chw.shape
    flat = directions_chw.permute(1, 2, 0).reshape(-1, 3) @ rotation.T
    return flat.reshape(height, width, 3).permute(2, 0, 1).contiguous()


def camera_world_rotation(view: View) -> torch.Tensor:
    """C2W rotation for envmap rays; matches the rasterizer's turntable fast path."""
    original_c2w = view.exif.get('turntable_original_c2w')
    if original_c2w is not None:
        return torch.as_tensor(
            np.asarray(original_c2w, dtype=np.float32)[:3, :3],
            device=view.rotation.device,
            dtype=torch.float32,
        )
    return view.rotation


def world_ray_directions(view: View) -> torch.Tensor:
    """Per-pixel world-space ray directions from the camera through pixel centers (3, H, W)."""
    rotation = camera_world_rotation(view)
    local = view.camera.compute_local_ray_directions().to(device=rotation.device, dtype=torch.float32)  # (H*W, 3)
    world = local @ rotation.T  # camera-to-world rotation
    world = torch.nn.functional.normalize(world, dim=-1)
    return world.reshape(view.camera.height, view.camera.width, 3).permute(2, 0, 1).contiguous()


def compose_deferred(
    base_color: torch.Tensor,
    feature_map: torch.Tensor,
    view: View,
    environment: EnvironmentMap,
    *,
    mesh_normal_map: torch.Tensor | None = None,
    mesh_normal_mask: torch.Tensor | None = None,
    mesh_normal_hijack: bool = True,
) -> dict[str, torch.Tensor]:
    """Compose the final deferred-reflection image.

    Args:
        base_color: (3, H, W) blended SH base color.
        feature_map: (4, H, W) blended (normal.xyz, reflection strength).
        view: the rendering view (for per-pixel ray directions).
        environment: learnable environment map.
        mesh_normal_map: optional (3, H, W) mesh world normals in [-1, 1].
        mesh_normal_mask: optional (1, H, W) foreground mask for mesh normals.
        mesh_normal_hijack: when True and mesh normals are set, use them for cubemap lookup.

    Returns a dict with ``rgb`` and the decomposition maps for debugging / losses.
    """
    gaussian_normal_map = feature_map[:3]
    refl_strength = feature_map[3:4].clamp(0.0, 1.0)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normal_map, dim=0, eps=1e-6)
    if mesh_normal_map is not None and mesh_normal_hijack:
        # 3DGS-DR: GT mesh normals from normals/ fully drive cubemap lookup (detached).
        normals = torch.nn.functional.normalize(mesh_normal_map, dim=0, eps=1e-6).detach()
    else:
        normals = gaussian_normals
    view_dirs = world_ray_directions(view)
    reflection_dirs = reflect(view_dirs, normals)
    turntable_rotation = turntable_reflection_rotation(view)
    if turntable_rotation is not None:
        reflection_dirs = apply_turntable_to_directions(reflection_dirs, turntable_rotation)
    reflection_dirs = torch.nn.functional.normalize(reflection_dirs, dim=0, eps=1e-6)
    dr_cfg = Framework.config.MODEL.DEFERRED_REFLECTION
    envmap_dirs = prepare_envmap_directions(
        reflection_dirs,
        scene_alignment_transform=view.exif.get('scene_alignment_transform'),
        roll_deg=float(getattr(dr_cfg, 'ENVMAP_ROLL_DEG', 0.0)),
    )
    reflection_color = environment.query_image(envmap_dirs)
    final = (1.0 - refl_strength) * base_color + refl_strength * reflection_color
    return {
        'rgb': final,
        'base_color': base_color,
        'reflection_color': reflection_color,
        'reflection_strength': refl_strength,
        'normal': normals,
        'gaussian_normal': gaussian_normals,
        'mesh_normal': mesh_normal_map,
        'mesh_normal_mask': mesh_normal_mask,
    }
