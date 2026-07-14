"""Bake a known HDR environment map into the FasterGS-DR learnable cubemap."""

from __future__ import annotations

import numpy as np
import torch

from Methods.FasterGS.DeferredShading import EnvironmentMap


def _load_hdri(path: str) -> np.ndarray:
    """Load an equirectangular HDRI as a linear ``(H, W, 3)`` float array."""
    lower = path.lower()
    if lower.endswith(('.exr', '.hdr')):
        try:
            import imageio.v3 as iio

            img = np.asarray(iio.imread(path), dtype=np.float32)
        except Exception:
            import os

            os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
            import cv2

            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        from PIL import Image

        img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3]


def _aces_tonemap(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _equirect_sample(hdri: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample an equirect HDRI (H,W,3) at unit directions (N,3)."""
    h, w, _ = hdri.shape
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    phi = torch.atan2(z, x) % (2.0 * np.pi)
    theta = torch.acos(torch.clamp(y, -1.0, 1.0))
    u = phi / (2.0 * np.pi) * (w - 1)
    v = theta / np.pi * (h - 1)
    x0 = torch.floor(u).long().clamp(0, w - 1)
    y0 = torch.floor(v).long().clamp(0, h - 1)
    x1 = (x0 + 1).clamp(0, w - 1)
    y1 = (y0 + 1).clamp(0, h - 1)
    fx = (u - x0.float()).unsqueeze(-1)
    fy = (v - y0.float()).unsqueeze(-1)
    c00 = hdri[y0, x0]
    c10 = hdri[y0, x1]
    c01 = hdri[y1, x0]
    c11 = hdri[y1, x1]
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


def bake_hdri_into_environment_map(
    environment: EnvironmentMap,
    hdri_path: str,
    *,
    exposure: float = 1.0,
    yaw_deg: float = 0.0,
    iters: int = 1500,
    batch: int = 100_000,
    lr: float = 0.05,
) -> None:
    """Fit ``environment`` so queried colors match the tonemapped pack HDRI."""
    from Logging import Logger

    hdri_np = _load_hdri(hdri_path) * float(exposure)
    if abs(float(yaw_deg)) >= 1e-6:
        shift = int(round(float(yaw_deg) / 360.0 * hdri_np.shape[1])) % hdri_np.shape[1]
        if shift:
            hdri_np = np.roll(hdri_np, shift, axis=1)
    hdri = torch.from_numpy(np.ascontiguousarray(hdri_np)).float().cuda()
    hdri = _aces_tonemap(hdri)
    hdri = torch.pow(torch.clamp(hdri, 0.0, 1.0), 1.0 / 2.2)

    opt = torch.optim.Adam(environment.parameters(), lr=lr)
    Logger.log_info(
        f'baking HDRI "{hdri_path}" (exposure={exposure}) into learnable cubemap: {iters} iters'
    )
    for it in range(iters):
        dirs = torch.randn(batch, 3, device='cuda')
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        target = _equirect_sample(hdri, dirs)
        pred = environment(dirs)
        loss = torch.abs(pred - target).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 300 == 0 or it == iters - 1:
            Logger.log_info(f'  envmap bake iter {it:4d}  fit L1={loss.item():.5f}')
    Logger.log_info('envmap HDRI bake complete (cubemap remains optimizable unless FREEZE_ENVMAP)')


def snapshot_envmap_parameters(environment: EnvironmentMap) -> dict[str, torch.Tensor]:
    """Clone cubemap parameters right after an HDRI bake for optional anchoring."""
    return {name: param.detach().clone() for name, param in environment.named_parameters()}


def envmap_anchor_loss(
    environment: EnvironmentMap,
    snapshot: dict[str, torch.Tensor],
) -> torch.Tensor:
    """L2 anchor keeping the learnable cubemap near its HDRI bake."""
    terms: list[torch.Tensor] = []
    for name, param in environment.named_parameters():
        terms.append((param - snapshot[name]).pow(2).mean())
    if not terms:
        return torch.zeros((), device=next(environment.parameters()).device)
    return torch.stack(terms).mean()
