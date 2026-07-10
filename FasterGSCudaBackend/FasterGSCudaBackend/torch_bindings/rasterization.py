from typing import NamedTuple, Any
import torch
from torch.autograd.function import once_differentiable

from FasterGSCudaBackend import _C


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor  # affine transformation from model/world space to view space
    cam_position: torch.Tensor  # camera position in world space
    sh_rotation: torch.Tensor  # 3x3 matrix applied to SH query directions
    bg_color: torch.Tensor  # background color in RGB format
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int  # width of the image plane in pixels
    height: int  # height of the image plane in pixels
    focal_x: float  # focal length in x direction in pixels
    focal_y: float  # focal length in y direction in pixels
    center_x: float  # x coordinate of the image center in pixels (positive -> right)
    center_y: float  # y coordinate of the image center in pixels (positive -> down)
    near_plane: float  # near clipping plane distance
    far_plane: float  # far clipping plane distance
    proper_antialiasing: bool  # whether to use proper antialiasing

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.sh_rotation,
            self.bg_color,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.proper_antialiasing,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
    ) -> torch.Tensor:
        (
            image,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        return image

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
    ) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]':
        (
            grad_means, grad_scales, grad_rotations, grad_opacities,
            grad_sh_coefficients_0, grad_sh_coefficients_rest
        ) = _C.backward(
            ctx.densification_info,
            grad_image,
            *ctx.saved_tensors,
            *ctx.rasterizer_settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            None,  # densification_info
            None,  # rasterizer_settings
        )


def diff_rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> torch.Tensor:
    return _Rasterize.apply(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        densification_info,
        rasterizer_settings,
    )


class _RasterizeDR(torch.autograd.Function):
    """Deferred-reflection rasterization.

    In addition to the SH color image, blends a per-Gaussian feature vector
    ``features`` (N, 4) = (normal.xyz, reflection strength) with the same alpha
    weights and returns a 4-channel feature map. Gradients flow to the standard
    Gaussian parameters and to ``features``.
    """

    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        features: torch.Tensor,
        densification_info: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        (
            image,
            feature_map,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward_dr(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            features,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            feature_map,
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_rest,
            features,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        return image, feature_map

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
        grad_feature_map: torch.Tensor,
    ) -> tuple:
        (
            image, feature_map, means, scales, rotations, opacities,
            sh_coefficients_rest, features,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
        ) = ctx.saved_tensors
        (
            grad_means, grad_scales, grad_rotations, grad_opacities,
            grad_sh_coefficients_0, grad_sh_coefficients_rest, grad_features
        ) = _C.backward_dr(
            ctx.densification_info,
            grad_image,
            grad_feature_map,
            image,
            feature_map,
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_rest,
            features,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
            *ctx.rasterizer_settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            grad_features,
            None,  # densification_info
            None,  # rasterizer_settings
        )


def diff_rasterize_dr(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    features: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> 'tuple[torch.Tensor, torch.Tensor]':
    return _RasterizeDR.apply(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        features,
        densification_info,
        rasterizer_settings,
    )


def rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    to_chw: bool,
    clamp_output: bool = True,
) -> torch.Tensor:
    return _C.inference(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
        to_chw,
        clamp_output,
    )


def update_pruning_scores(
    scores: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> torch.Tensor:
    return _C.pruning_scores(
        scores,
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
    )
