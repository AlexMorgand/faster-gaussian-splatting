"""Direction-frame helpers for FasterGS-DR environment-map lookup."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def rotation_matrix_x_deg(deg: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """World-space rotation around +X (row-vector convention ``d' = d @ R``)."""
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        device=device,
        dtype=dtype,
    )


def directions_dataset_to_hdri_frame(
    directions_chw: torch.Tensor,
    scene_alignment_transform: Any,
) -> torch.Tensor:
    """Map unit directions from PCA-aligned dataset world back to pack/HDRI GL world."""
    rotation = torch.as_tensor(
        np.asarray(scene_alignment_transform, dtype=np.float32)[:3, :3],
        device=directions_chw.device,
        dtype=directions_chw.dtype,
    )
    _, height, width = directions_chw.shape
    flat = directions_chw.permute(1, 2, 0).reshape(-1, 3) @ rotation
    aligned = torch.nn.functional.normalize(flat, dim=-1)
    return aligned.reshape(height, width, 3).permute(2, 0, 1).contiguous()


def prepare_envmap_directions(
    reflection_dirs: torch.Tensor,
    *,
    scene_alignment_transform: Any | None,
    roll_deg: float = 0.0,
) -> torch.Tensor:
    """Convert reflection directions into the frame used when baking the cubemap."""
    dirs = reflection_dirs
    if scene_alignment_transform is not None:
        dirs = directions_dataset_to_hdri_frame(dirs, scene_alignment_transform)
    if abs(float(roll_deg)) >= 1e-6:
        rotation = rotation_matrix_x_deg(roll_deg, device=dirs.device, dtype=dirs.dtype)
        _, height, width = dirs.shape
        flat = dirs.permute(1, 2, 0).reshape(-1, 3) @ rotation
        dirs = torch.nn.functional.normalize(flat, dim=-1).reshape(height, width, 3).permute(2, 0, 1).contiguous()
    return dirs
