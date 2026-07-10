#pragma once

#include <torch/extension.h>
#include <tuple>

namespace faster_gs::rasterization {

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
    forward_wrapper(
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_0,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& sh_rotation,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_wrapper(
        torch::Tensor& densification_info,
        const torch::Tensor& grad_image,
        const torch::Tensor& image,
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& primitive_buffers,
        const torch::Tensor& tile_buffers,
        const torch::Tensor& instance_buffers,
        const torch::Tensor& bucket_buffers,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& sh_rotation,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const int n_instances,
        const int n_buckets,
        const int instance_primitive_indices_selector);

    // deferred-reflection forward: also returns a 4-channel feature map
    // (blended normal.xyz + reflection strength) from a per-Gaussian features (N,4) input
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
    forward_wrapper_dr(
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_0,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& features,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& sh_rotation,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing);

    // deferred-reflection backward: also consumes grad_feature_map + feature_map and
    // returns grad_features (N,4) in addition to the standard gradients
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_wrapper_dr(
        torch::Tensor& densification_info,
        const torch::Tensor& grad_image,
        const torch::Tensor& grad_feature_map,
        const torch::Tensor& image,
        const torch::Tensor& feature_map,
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& features,
        const torch::Tensor& primitive_buffers,
        const torch::Tensor& tile_buffers,
        const torch::Tensor& instance_buffers,
        const torch::Tensor& bucket_buffers,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& sh_rotation,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const int n_instances,
        const int n_buckets,
        const int instance_primitive_indices_selector);

    torch::Tensor
    inference_wrapper(
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_0,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& sh_rotation,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const bool to_chw,
        const bool clamp_output);

    void
    pruning_scores_wrapper(
        torch::Tensor& scores,
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients_0,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const torch::Tensor& sh_rotation,
        const torch::Tensor& bg_color,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing);

}
