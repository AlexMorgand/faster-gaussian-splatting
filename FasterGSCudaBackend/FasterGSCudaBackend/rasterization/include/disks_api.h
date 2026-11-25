#pragma once
#include <torch/extension.h>
#include <vector>

namespace faster_gs::rasterization {

std::vector<torch::Tensor> disk_forward_wrapper(
    const torch::Tensor& means, const torch::Tensor& scales,
    const torch::Tensor& rotations, const torch::Tensor& opacities,
    const torch::Tensor& sh0, const torch::Tensor& sh_rest,
    const torch::Tensor& w2c, const torch::Tensor& cam_position,
    const torch::Tensor& bg_color, int active_sh, int width, int height,
    float focal_x, float focal_y, float center_x, float center_y,
    float near_plane, float far_plane, const torch::Tensor& image_shift, bool collect_metadata, bool preflight_only);

std::vector<torch::Tensor> disk_forward_channels_wrapper(
    const torch::Tensor& means, const torch::Tensor& scales,
    const torch::Tensor& rotations, const torch::Tensor& opacities,
    const torch::Tensor& sh0, const torch::Tensor& sh_rest,
    const torch::Tensor& w2c, const torch::Tensor& cam_position,
    const torch::Tensor& bg_color, int active_sh, int width, int height,
    float focal_x, float focal_y, float center_x, float center_y,
    float near_plane, float far_plane, const torch::Tensor& image_shift, bool collect_metadata, bool preflight_only, int channels);

std::vector<torch::Tensor> disk_backward_wrapper(
    const torch::Tensor& grad_image, const torch::Tensor& image,
    const torch::Tensor& geometry, const torch::Tensor& means,
    const torch::Tensor& scales, const torch::Tensor& rotations,
    const torch::Tensor& sh_rest, const torch::Tensor& primitive_blob,
    const torch::Tensor& tile_blob, const torch::Tensor& instance_blob,
    const torch::Tensor& image_shift,
    const torch::Tensor& w2c, const torch::Tensor& cam_position,
    const torch::Tensor& bg_color, int active_sh, int n_instances,
    int selector, float focal_x, float focal_y, float center_x, float center_y,
    float near_plane, float far_plane);

torch::Tensor depth_difference_backward_wrapper(const torch::Tensor& vertical, const torch::Tensor& horizontal,
    const torch::Tensor& rx, const torch::Tensor& ry, const torch::Tensor& rotation);
}
