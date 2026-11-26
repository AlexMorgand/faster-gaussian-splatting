#pragma once

#include "buffer_utils.h"
#include <cuda_runtime.h>

namespace faster_gs::rasterization {

// Independent perspective-disk primitive slice; 3D remains the default path.
struct Disk {
    float3 center, tangent_u, tangent_v, normal;
    float opacity;
};

struct DiskRasterization {
    Disk* geometry;
    float* aux;
    float focal_x, focal_y, center_x, center_y, near_plane, far_plane;
    const float2* image_shift;
    float* metadata;  // [conservative clipped support radius, accepted-hit visibility]
    int n_primitives;
    unsigned long long* instance_counter;
    unsigned long long instance_limit;
    bool preflight_only;
    int output_channels = 10;
};

void preprocess_disks(
    const float3* means, const float2* scales, const float4* rotations,
    const float* opacities, const float3* sh0, const float3* sh_rest,
    const float4* w2c, const float3* cam_position, PrimitiveBuffers& primitives,
    const DiskRasterization& disks, int n, int active_sh, int total_sh,
    int width, int height, cudaStream_t stream);

void blend_disks(
    const PrimitiveBuffers& primitives, const TileBuffers& tiles,
    const uint* indices, uint* hit_masks, const float3* bg_color,
    float* image, const DiskRasterization& disks,
    int width, int height, cudaStream_t stream);

}
