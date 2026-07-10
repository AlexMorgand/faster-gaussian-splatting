#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace faster_gs::cubemapencoder {

void cubemap_encode_forward(
    const at::Tensor inputs,
    const at::Tensor cubemap,
    const at::Tensor fail_value,
    at::Tensor outputs,
    const uint32_t interp,
    const uint32_t seamless,
    const uint32_t B,
    const uint32_t C,
    const uint32_t L);

void cubemap_encode_backward(
    const at::Tensor grad_outputs,
    const at::Tensor inputs,
    const at::Tensor cubemap,
    at::Tensor grad_cubemap,
    at::Tensor grad_inputs,
    at::Tensor grad_fail,
    const uint32_t interp,
    const uint32_t seamless,
    const uint32_t B,
    const uint32_t C,
    const uint32_t L);

}  // namespace faster_gs::cubemapencoder
