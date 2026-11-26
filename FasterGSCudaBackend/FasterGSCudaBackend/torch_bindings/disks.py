"""Experimental true perspective disks with an explicit pixel-translation proxy.

Output CHW: RGB, weighted depth, alpha, weighted world-normal XYZ, median
depth, distortion. Raw log-scales are Nx2; SH0/rest remain separate.
Median is max accepted intersection depth with pre-hit transmittance > .5
(paper Appendix B Eq.19); exact ties select the first hit for backward.
Footprint is max(disk, sigma=sqrt(1/2)-pixel projected-centre Gaussian);
ties select disk. Depth always remains ray-plane intersection depth;
singular/clipped intersections are rejected, including screen-winning hits.
The separate FasterGS proper_antialiasing option remains unsupported.
"""
import torch
from torch.autograd.function import once_differentiable

from FasterGSCudaBackend import _C
from .rasterization import RasterizerSettings


class _DiskRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, scales, rotations, opacities, sh0, rest, image_shift, settings, collect_metadata, output_channels):
        if settings.proper_antialiasing:
            raise ValueError("disk primitive does not implement production antialiasing")
        args = (
            means, scales, rotations, opacities, sh0, rest,
            *settings.as_tuple()[:-1], image_shift, collect_metadata, False,
        )
        buffers = _C.disk_forward(*args) if output_channels == 10 else _C.disk_forward_channels(*args, output_channels)
        ctx.output_channels = output_channels
        image, geometry, primitive, tile, instance, _, metadata, state = buffers
        ctx.save_for_backward(image, geometry, means, scales, rotations, rest, primitive, tile, instance, image_shift)
        ctx.settings = settings
        ctx.state = state.tolist()
        ctx.mark_non_differentiable(metadata)
        return image[:10], metadata

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_image, _grad_metadata):
        s = ctx.settings
        n_instances, _, selector = ctx.state
        grads = _C.disk_backward(
            grad_image[:ctx.output_channels].contiguous(), *ctx.saved_tensors,
            s.w2c, s.cam_position, s.bg_color, s.active_sh_bases,
            n_instances, selector, s.focal_x, s.focal_y, s.center_x, s.center_y,
            s.near_plane, s.far_plane,
        )
        return *grads, None, None, None


def diff_rasterize_disks(means, scales, rotations, opacities, sh0, rest, settings: RasterizerSettings,
                         image_shift=None, return_metadata=False, output_channels=10):
    """Shift is Nx2 in pixels; empty input avoids an Nx2 allocation in inference.

    output_channels selects the retained prefix (3 RGB, 8 with depth/alpha/normals,
    9 with median depth, or all 10 with distortion). Omitted planes are zero and
    contribute no gradient. The returned image always has 10 channels.

    Metadata rows are conservative clipped support radius (pixels) and an
    accepted-contribution visibility flag, not a 3D score or a pruning policy.
    """
    if image_shift is None:
        image_shift = means.new_empty(0)
    image, metadata = _DiskRasterize.apply(
        means, scales, rotations, opacities, sh0, rest, image_shift, settings, return_metadata, output_channels)
    return (image, metadata) if return_metadata else image


@torch.no_grad()
def preflight_disks(means, scales, rotations, opacities, sh0, rest, settings):
    """Exact uint64 preprocess/count guard only; performs no sort, binning or render."""
    state, = _C.disk_forward(means, scales, rotations, opacities, sh0, rest,
                             *settings.as_tuple()[:-1], means.new_empty(0), False, True)
    return int(state[0])
