"""Differentiable cubemap encoder for deferred-reflection environment maps."""

from __future__ import annotations

from typing import Any

import torch

from FasterGSCudaBackend import _C

_INTERP_TO_ID = {
    'nearest': 0,
    'linear': 1,
}


class _CubemapEncode(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        cubemap_texture: torch.Tensor,
        fail_value: torch.Tensor,
        interpolation: int,
        seamless: int,
    ) -> torch.Tensor:
        cubemap_texture = cubemap_texture.contiguous()
        inputs = inputs.contiguous()
        _, channels, resolution, _ = cubemap_texture.shape
        batch_size = inputs.shape[0]
        outputs = torch.empty(channels, batch_size, dtype=cubemap_texture.dtype, device=cubemap_texture.device)
        _C.cubemap_encode_forward(
            inputs,
            cubemap_texture,
            fail_value,
            outputs,
            interpolation,
            seamless,
            batch_size,
            channels,
            resolution,
        )
        ctx.interpolation = interpolation
        ctx.seamless = seamless
        ctx.save_for_backward(inputs, cubemap_texture)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        inputs, cubemap_texture = ctx.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        _, channels, resolution, _ = cubemap_texture.shape
        batch_size = inputs.shape[0]
        grad_cubemap = torch.zeros_like(cubemap_texture)
        grad_inputs = torch.zeros_like(inputs)
        grad_fail = torch.zeros(channels, dtype=cubemap_texture.dtype, device=cubemap_texture.device)
        _C.cubemap_encode_backward(
            grad_outputs,
            inputs,
            cubemap_texture,
            grad_cubemap,
            grad_inputs,
            grad_fail,
            ctx.interpolation,
            ctx.seamless,
            batch_size,
            channels,
            resolution,
        )
        return grad_inputs, grad_cubemap, grad_fail, None, None


_cubemap_encode = _CubemapEncode.apply


class CubemapEncoder(torch.nn.Module):
    """Learnable cubemap texture queried by unit direction vectors.

    Matches the 3DGS-DR ``cubemapencoder`` API: stores pre-sigmoid logits in
    ``Cubemap_texture`` and returns raw channel values (apply ``sigmoid`` outside
    for display colors in ``[0, 1]``).
    """

    def __init__(self, output_dim: int = 3, resolution: int = 256, interpolation: str = 'linear') -> None:
        super().__init__()
        if interpolation not in _INTERP_TO_ID:
            raise ValueError(f'unsupported interpolation {interpolation!r}; expected nearest or linear')
        self.input_dim = 3
        self.resolution = resolution
        self.output_dim = output_dim
        self.interpolation = interpolation
        self.interp_id = _INTERP_TO_ID[interpolation]
        self.seamless = 1
        self.params = torch.nn.ParameterDict({
            'Cubemap_texture': torch.nn.Parameter(torch.rand(6, output_dim, resolution, resolution) * 10.0 - 5.0),
            'Cubemap_failv': torch.nn.Parameter(torch.zeros(output_dim)),
        })
        self.n_elems = 6 * output_dim * resolution * resolution + output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Query directions ``inputs`` (N, 3), returning logits (N, C)."""
        outputs = _cubemap_encode(
            inputs,
            self.params['Cubemap_texture'],
            self.params['Cubemap_failv'],
            self.interp_id,
            self.seamless,
        )
        return outputs.permute(1, 0)
