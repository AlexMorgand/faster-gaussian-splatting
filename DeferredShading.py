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


def mirror_cue_maps(
    mesh_metallic_map: torch.Tensor | None,
    mesh_roughness_map: torch.Tensor | None,
    foreground_mask: torch.Tensor | None = None,
    mesh_albedo_map: torch.Tensor | None = None,
    hard_mirror_min_albedo: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Build gloss / chrome cue maps.

    Returns ``(raw_gloss, metallic_chrome, chrome_gloss, albedo_luma)``.

    - ``raw_gloss``: ``1 - roughness`` on foreground (not albedo-gated) — dark glossy leather.
    - ``metallic_chrome`` / ``chrome_gloss``: albedo-gated cues for bright pure mirrors.
    """
    fg = foreground_mask
    luma = None
    if mesh_albedo_map is not None:
        luma = mesh_albedo_map.mean(dim=0, keepdim=True)
        if fg is not None:
            luma = luma * fg.to(dtype=luma.dtype)

    chrome = None
    if luma is not None and float(hard_mirror_min_albedo) > 0.0:
        ref = mesh_metallic_map if mesh_metallic_map is not None else mesh_roughness_map
        chrome = (luma >= float(hard_mirror_min_albedo)).to(dtype=ref.dtype)
        if fg is not None:
            chrome = chrome * fg.to(dtype=chrome.dtype)

    raw_gloss = None
    if mesh_roughness_map is not None:
        raw_gloss = (1.0 - mesh_roughness_map).clamp(0.0, 1.0)
        if fg is not None:
            raw_gloss = raw_gloss * fg.to(dtype=raw_gloss.dtype)

    chrome_gloss = raw_gloss
    if chrome_gloss is not None and chrome is not None:
        chrome_gloss = chrome_gloss * chrome

    metallic_chrome = mesh_metallic_map
    if metallic_chrome is not None and chrome is not None:
        metallic_chrome = metallic_chrome * chrome

    return raw_gloss, metallic_chrome, chrome_gloss, luma


def compose_deferred(
    base_color: torch.Tensor,
    feature_map: torch.Tensor,
    view: View,
    environment: EnvironmentMap,
    *,
    mesh_normal_map: torch.Tensor | None = None,
    mesh_normal_mask: torch.Tensor | None = None,
    mesh_normal_hijack: bool = True,
    mesh_albedo_map: torch.Tensor | None = None,
    mesh_albedo_mask: torch.Tensor | None = None,
    force_albedo_base_color: bool = False,
    mesh_metallic_map: torch.Tensor | None = None,
    mesh_metallic_mask: torch.Tensor | None = None,
    mesh_roughness_map: torch.Tensor | None = None,
    hard_mirror_from_metallic: bool = False,
    hard_mirror_threshold: float = 0.5,
    hard_mirror_gloss_threshold: float = 0.9,
    hard_mirror_min_albedo: float = 0.0,
    soft_specular_use: bool = False,
    soft_specular_max_albedo: float = 0.28,
    soft_specular_gloss_threshold: float = 0.7,
    soft_specular_refl_scale: float = 0.55,
    soft_specular_refl_max: float = 0.55,
    soft_specular_use_albedo_base: bool = True,
    soft_specular_prior: float = 0.45,
) -> dict[str, torch.Tensor]:
    """Compose the final deferred-reflection image.

    Three material regimes when hard/soft hybrid flags are on:

    1. **Chrome / pure mirror** (bright albedo + high metal/gloss): ``R=1``, HDRI only.
    2. **Intermediate** (dark/mid albedo + high gloss, e.g. black patent leather): keep
       mesh albedo as base color, cap ``R < 1`` so object color shows under env highlights.
    3. **Diffuse**: learned Gaussian ``R`` / FasterGS base color.
    """
    gaussian_normal_map = feature_map[:3]
    gaussian_refl_strength = feature_map[3:4].clamp(0.0, 1.0)
    refl_strength = gaussian_refl_strength
    gaussian_normals = torch.nn.functional.normalize(gaussian_normal_map, dim=0, eps=1e-6)
    if force_albedo_base_color and mesh_albedo_map is not None and mesh_albedo_mask is not None:
        base_color = mesh_albedo_map * mesh_albedo_mask

    raw_gloss, metallic_chrome, chrome_gloss, albedo_luma = mirror_cue_maps(
        mesh_metallic_map,
        mesh_roughness_map,
        foreground_mask=mesh_metallic_mask,
        mesh_albedo_map=mesh_albedo_map,
        hard_mirror_min_albedo=hard_mirror_min_albedo,
    )

    mirror_mask = None
    soft_mask = None
    refl_target = None

    if hard_mirror_from_metallic and (metallic_chrome is not None or chrome_gloss is not None):
        metal_hit = (
            (metallic_chrome >= float(hard_mirror_threshold))
            if metallic_chrome is not None
            else torch.zeros_like(refl_strength, dtype=torch.bool)
        )
        gloss_hit = (
            (chrome_gloss >= float(hard_mirror_gloss_threshold))
            if chrome_gloss is not None
            else torch.zeros_like(refl_strength, dtype=torch.bool)
        )
        mirror_mask = (metal_hit | gloss_hit).to(dtype=refl_strength.dtype)
        if mesh_metallic_mask is not None:
            mirror_mask = mirror_mask * mesh_metallic_mask.to(dtype=refl_strength.dtype)
        refl_strength = torch.where(mirror_mask > 0.5, torch.ones_like(refl_strength), refl_strength)

    if (
        soft_specular_use
        and albedo_luma is not None
        and mesh_albedo_map is not None
    ):
        # Dark/mid surfaces: never allow R→1 (that turns black leather into chrome).
        # Cap learned R; replace base with mesh albedo on dark+glossy so fal-baked bright SH
        # highlights don't become a silver plate. Trust fal RGB loss to raise R on patent
        # leather highlights and leave matte fabric near R≈0.
        dark = (albedo_luma < float(soft_specular_max_albedo)).to(dtype=refl_strength.dtype)
        if mesh_metallic_mask is not None:
            dark = dark * mesh_metallic_mask.to(dtype=dark.dtype)
        if mirror_mask is not None:
            dark = dark * (1.0 - (mirror_mask > 0.5).to(dtype=dark.dtype))
        soft_mask = dark
        if raw_gloss is not None:
            soft_mask = soft_mask * (
                raw_gloss >= float(soft_specular_gloss_threshold)
            ).to(dtype=soft_mask.dtype)
        r_cap = refl_strength.new_full((), float(soft_specular_refl_max))
        refl_strength = torch.where(
            dark > 0.5, torch.minimum(refl_strength, r_cap.expand_as(refl_strength)), refl_strength
        )
        if soft_specular_use_albedo_base and soft_mask is not None:
            alb = mesh_albedo_map
            if mesh_albedo_mask is not None:
                alb = alb * mesh_albedo_mask
            soft3 = soft_mask.expand_as(base_color) if soft_mask.shape[0] == 1 else soft_mask
            base_color = torch.where(soft3 > 0.5, alb, base_color)

    # Prior target: chrome→1, else albedo-gated metal/gloss (dark → ~0; fal RGB drives leather R).
    if metallic_chrome is not None or chrome_gloss is not None:
        if metallic_chrome is None:
            refl_target = chrome_gloss
        elif chrome_gloss is None:
            refl_target = metallic_chrome
        else:
            refl_target = torch.maximum(metallic_chrome, chrome_gloss)
    elif raw_gloss is not None:
        refl_target = raw_gloss
    if refl_target is not None and mirror_mask is not None:
        refl_target = torch.where(mirror_mask > 0.5, torch.ones_like(refl_strength), refl_target)

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
        'gaussian_reflection_strength': gaussian_refl_strength,
        'mirror_mask': mirror_mask,
        'soft_specular_mask': soft_mask,
        'normal': normals,
        'gaussian_normal': gaussian_normals,
        'mesh_normal': mesh_normal_map,
        'mesh_normal_mask': mesh_normal_mask,
        'mesh_albedo': mesh_albedo_map,
        'mesh_albedo_mask': mesh_albedo_mask,
        'mesh_gloss': raw_gloss,
        'mesh_refl_target': refl_target,
    }
