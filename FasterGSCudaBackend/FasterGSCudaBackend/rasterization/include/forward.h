#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include <functional>
#include <tuple>

namespace faster_gs::rasterization {

    std::tuple<int, int, int> forward(
        std::function<char* (size_t)> resize_primitive_buffers,
        std::function<char* (size_t)> resize_tile_buffers,
        std::function<char* (size_t)> resize_instance_buffers,
        std::function<char* (size_t)> resize_bucket_buffers,
        const float3* means,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        const float3* sh_rotation,
        const float3* bg_color,
        float* image,
        const int n_primitives,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        // deferred-reflection extension: when features != nullptr the blend also
        // composits a per-Gaussian feature vector (normal.xyz + refl) into feature_map
        const float4* features = nullptr,
        float* feature_map = nullptr);

    template <typename KeyT>
    void diff_rasterize(
        std::function<char* (size_t)>& resize_instance_buffers,
        std::function<char* (size_t)>& resize_bucket_buffers,
        PrimitiveBuffers& primitive_buffers,
        TileBuffers& tile_buffers,
        const dim3& grid,
        const dim3& block,
        const float3* bg_color,
        float* image,
        const cudaStream_t memset_stream,
        const int n_visible_primitives,
        const int n_instances,
        const int n_tiles,
        const int end_bit,
        const int width,
        const int height,
        int& n_buckets,
        int& instance_primitive_indices_selector);

    template <typename KeyT>
    void diff_rasterize_dr(
        std::function<char* (size_t)>& resize_instance_buffers,
        std::function<char* (size_t)>& resize_bucket_buffers,
        PrimitiveBuffers& primitive_buffers,
        TileBuffers& tile_buffers,
        const dim3& grid,
        const dim3& block,
        const float3* bg_color,
        const float4* features,
        float* image,
        float* feature_map,
        const cudaStream_t memset_stream,
        const int n_visible_primitives,
        const int n_instances,
        const int n_tiles,
        const int end_bit,
        const int width,
        const int height,
        int& n_buckets,
        int& instance_primitive_indices_selector);

}
