"""Perspective-disk rendering and calibrated depth unprojection.

Adapted from MIT-licensed NeRF-ICG revision 57ddd49.
Copyright (c) 2026 Moritz Kappel, Florian Hahlbohm, Timon Scholz. See LICENSE.
"""
from copy import deepcopy
from enum import IntEnum
import math

import torch
import torch.nn.functional as F

import Framework
from Datasets.utils import View
from FasterGSCudaBackend import _C
from FasterGSCudaBackend.torch_bindings.disks import diff_rasterize_disks
from Methods.FasterGS.Renderer import FasterGSRenderer, extract_settings
from Methods.FasterGS.utils import rgb_to_sh0
from .Model import FasterGS2DModel


class RenderLayout(IntEnum):
    """Channel prefixes supported by the disk blend kernels."""
    RGB = 3
    EXPECTED_DEPTH_NORMALS = 8
    MEDIAN_DEPTH_NORMALS = 9
    DISTORTION = 10


def _world_differences(z, rx, ry, rotation, position):
    points = torch.stack((rx[None, :] * z, ry[:, None] * z, z), -1)
    points = points @ rotation.T + position
    return points[2:, 1:-1] - points[:-2, 1:-1], points[1:-1, 2:] - points[1:-1, :-2]


class _FixedCameraDifferences(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, rx, ry, rotation, position):
        ctx.save_for_backward(rx, ry, rotation)
        # Small matrix products can use a different FP32 reduction order in cuBLAS.
        if z.numel() >= 1024 and hasattr(_C, "depth_difference_forward"):
            return tuple(_C.depth_difference_forward(z.contiguous(), rx, ry, rotation.contiguous(), position.contiguous()))
        return _world_differences(z, rx, ry, rotation, position)

    @staticmethod
    def backward(ctx, vertical, horizontal):
        rx, ry, rotation = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve higher-order autograd through the linear difference adjoint.
            dv = F.pad(vertical, (0, 0, 1, 1, 2, 0)) - F.pad(vertical, (0, 0, 1, 1, 0, 2))
            dh = F.pad(horizontal, (0, 0, 2, 0, 1, 1)) - F.pad(horizontal, (0, 0, 0, 2, 1, 1))
            dx, dy, dz = ((dv + dh) @ rotation).unbind(-1)
            return dz + dy * ry[:, None] + dx * rx[None, :], None, None, None, None
        depth = _C.depth_difference_backward(vertical.contiguous(), horizontal.contiguous(),
                                             rx, ry, rotation.contiguous())
        return depth, None, None, None, None


class _NormalizeNormal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        length = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
        unit = x / length.clamp_min(1e-8)
        ctx.save_for_backward(x, length, unit)
        ctx.dim = dim
        return unit

    @staticmethod
    def backward(ctx, grad):
        x, length, unit = ctx.saved_tensors
        if torch.is_grad_enabled():
            unit = F.normalize(x, dim=ctx.dim, eps=1e-8)
            dx, = torch.autograd.grad(unit, x, grad, create_graph=True)
        elif hasattr(_C, "normalize_normal_backward_strided"):
            dx = _C.normalize_normal_backward_strided(unit, length, grad, ctx.dim == 0)
        else:
            dx = _C.normalize_normal_backward(unit, length, grad.contiguous(), ctx.dim == 0)
        return dx, None


def _normalize_normal(x, dim):
    if (torch.is_grad_enabled() and x.requires_grad and x.is_cuda and x.dtype == torch.float32
            and x.is_contiguous() and 0 < x.numel() <= 2**31 - 1 and not torch.is_autocast_enabled()
            and hasattr(_C, "normalize_normal_backward")):
        return _NormalizeNormal.apply(x, dim)
    return F.normalize(x, dim=dim, eps=1e-8)


def depth_to_normal(view, depth, pose=None):
    z = depth[0]
    if min(z.shape) < 3:
        raise Framework.RendererError("FasterGS2D depth normal requires at least a 3x3 raster")
    camera = view.camera
    rx = (torch.arange(z.shape[1], device=z.device, dtype=z.dtype) + .5 - camera.center_x) / camera.focal_x
    ry = (torch.arange(z.shape[0], device=z.device, dtype=z.dtype) + .5 - camera.center_y) / camera.focal_y
    rotation, position = (view.rotation, view.position) if pose is None else pose
    if (z.requires_grad and z.is_cuda and z.dtype == torch.float32 and hasattr(_C, "depth_difference_backward")
            and not any(t.requires_grad for t in (rx, ry, rotation, position)) and not torch.is_autocast_enabled()
            and not torch.backends.cuda.matmul.allow_tf32):
        vertical, horizontal = _FixedCameraDifferences.apply(z, rx, ry, rotation, position)
    else:
        vertical, horizontal = _world_differences(z, rx, ry, rotation, position)
    normals = _normalize_normal(torch.linalg.cross(vertical, horizontal), dim=-1)
    return F.pad(normals.permute(2, 0, 1), (1, 1, 1, 1))


class FasterGS2DRenderer(FasterGSRenderer):
    model_type = FasterGS2DModel
    _configuration = deepcopy(FasterGSRenderer.get_default_parameters())
    _configuration.update(DEPTH_RATIO=0.0)

    def __init__(self, model):
        super().__init__(model)
        if model.ppisp is not None:
            raise Framework.RendererError("FasterGS2D experimental path does not support PPISP")
        if self.PROPER_ANTIALIASING:
            raise Framework.RendererError("FasterGS2D supports its paper lowpass, not 3D proper_antialiasing")
        if not 0 <= self.DEPTH_RATIO <= 1:
            raise Framework.RendererError("FasterGS2D DEPTH_RATIO must be in [0,1]")

    def _rasterize(self, view, bg_color, collect=False, scale_modifier=1.0, output_channels=RenderLayout.DISTORTION, settings=None):
        g = self.model.gaussians
        if settings is None:
            settings = extract_settings(view, g.active_sh_bases, bg_color, self.PROPER_ANTIALIASING)
        shift = g.means.new_zeros((len(g.means), 2), requires_grad=True) if collect else None
        sh0, rest = g.sh_coefficients_0, g.sh_coefficients_rest
        if not self.model.training:
            if not self.RENDER_DIFFUSE:
                sh0 = torch.full_like(sh0, rgb_to_sh0(0.0))
            if not self.RENDER_SPECULAR:
                rest = torch.zeros_like(rest)
        options = {}
        if output_channels != RenderLayout.DISTORTION and hasattr(_C, "disk_forward_channels"):
            options["output_channels"] = output_channels
        scales = g.raw_scales
        if scale_modifier != 1.0:
            scales = scales + math.log(max(scale_modifier, 1e-6))
        rendered = diff_rasterize_disks(
            g.means, scales,
            g.raw_rotations, g.raw_opacities, sh0, rest,
            settings, image_shift=shift, return_metadata=collect, **options)
        output, metadata = rendered if collect else (rendered, None)
        return output, shift, metadata

    def _parse(self, view, output, compute_surf_normal, rgb_only=False, pose=None):
        rgb, z, alpha, raw_normal, median, distortion = output.split((3, 1, 1, 3, 1, 1), dim=0)
        if rgb_only:
            # The RGB kernel already zeroes every auxiliary plane.
            depth, normal = z, raw_normal
            unit_surface = surface = raw_normal.detach()
        else:
            depth = median
            if self.DEPTH_RATIO < 1:
                depth = (z / alpha.clamp_min(1e-8)).nan_to_num(0)
                if self.DEPTH_RATIO > 0:
                    depth = depth * (1 - self.DEPTH_RATIO) + median * self.DEPTH_RATIO
            normal = _normalize_normal(raw_normal, dim=0)
            unit_surface = depth_to_normal(view, depth, pose) if compute_surf_normal else torch.zeros_like(normal)
            surface = unit_surface * alpha.detach() if compute_surf_normal else unit_surface
        return {"rgb": rgb, "rend_alpha": alpha, "surf_depth": depth, "rend_normal": normal,
                "surf_normal": surface, "weighted_disk_normal": raw_normal,
                "unit_surface_normal": unit_surface, "rend_dist": distortion}

    def render_image_training(self, view, update_densification_info, bg_color, compute_surf_normal=True, output_channels=RenderLayout.DISTORTION):
        settings = extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, self.PROPER_ANTIALIASING)
        output, shift, metadata = self._rasterize(view, bg_color, update_densification_info,
                                                 output_channels=output_channels, settings=settings)
        # View constructs w2c with the transpose of its stored rotation. Reuse the
        # uploaded pose so normal supervision does not synchronize to upload it again.
        pose = (settings.w2c[:3, :3].T.contiguous(), settings.cam_position) if compute_surf_normal and type(view) is View else None
        rgb_only = output_channels == RenderLayout.RGB and not compute_surf_normal and hasattr(_C, "disk_forward_channels")
        return {**self._parse(view, output, compute_surf_normal, rgb_only, pose),
                "image_shift": shift, "metadata": metadata}

    @torch.no_grad()
    def render_image_inference(self, view, to_chw=False):
        layout = RenderLayout.MEDIAN_DEPTH_NORMALS if self.DEPTH_RATIO else RenderLayout.EXPECTED_DEPTH_NORMALS
        output, _, _ = self._rasterize(view, view.camera.background_color, scale_modifier=self.SCALE_MODIFIER,
                                       output_channels=layout)
        parsed = self._parse(view, output, False)
        rgb = parsed["rgb"].clamp(0, 1)
        normal = torch.where(parsed["rend_alpha"] > 1e-4, .5*parsed["rend_normal"]+.5,
                             torch.zeros_like(parsed["rend_normal"]))
        outputs = {"rgb": rgb, "alpha": parsed["rend_alpha"], "depth": parsed["surf_depth"], "normal": normal}
        return outputs if to_chw else {key: value.permute(1, 2, 0) for key, value in outputs.items()}

    def render_image_benchmark(self, *args, **kwargs):
        raise Framework.RendererError("FasterGS2D does not support optimized 3D inference")

    def compute_pruning_scores(self, *args, **kwargs):
        raise Framework.RendererError("FasterGS2D does not support the 3D pruning scorer")

    def ppisp_controller_distillation(self, view):
        raise Framework.RendererError("FasterGS2D experimental path does not support PPISP/distillation")

    def postprocess_outputs(self, outputs, *args):
        return {key: value for key, value in outputs.items() if key in ("rgb", "normal")}
