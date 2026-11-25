#include "disks.h"
#include "disks_api.h"
#include "forward.h"
#include "kernel_utils.cuh"
#include "sh_utils.cuh"
#include "torch_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <type_traits>
#include <cuda/atomic>
#include <cooperative_groups/reduce.h>

namespace faster_gs::rasterization {
namespace {
using namespace kernels;

__device__ float3 camera_vector(const float4* w, float3 v) {
    return make_float3(dot(make_float3(w[0]), v), dot(make_float3(w[1]), v),
                       dot(make_float3(w[2]), v));
}

__device__ float3 world_adjoint(const float4* w, float3 v) {
    return make_float3(w[0].x*v.x + w[1].x*v.y + w[2].x*v.z,
                      w[0].y*v.x + w[1].y*v.y + w[2].y*v.z,
                      w[0].z*v.x + w[1].z*v.y + w[2].z*v.z);
}

__device__ bool finite3(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__device__ float3 shifted_ray(float3 ray, const DiskRasterization& d, int i) {
    if (d.image_shift) {
        ray.x -= d.image_shift[i].x/d.focal_x;
        ray.y -= d.image_shift[i].y/d.focal_y;
    }
    return ray;
}

__device__ bool outside_support(float3 c, float3 u, float3 v, float radius,
    float2 shift, const DiskRasterization& d, int width, int height) {
    // Certify BOTH supports; padding/error bounds only retain extra old-path work.
    const double sx = double(d.focal_x)*c.x/c.z, sy = double(d.focal_y)*c.y/c.z;
    const double px = sx+d.center_x+shift.x, py = sy+d.center_y+shift.y;
    const double sr = double(radius)*0.7071067811865476+1.0;
    const double ex = 1e-5*(fabs(sx)+fabs(double(d.center_x))+fabs(double(shift.x))+width+1);
    const double ey = 1e-5*(fabs(sy)+fabs(double(d.center_y))+fabs(double(shift.y))+height+1);
    if (!(px+sr+ex < 0 || px-sr-ex > width ||
          py+sr+ey < 0 || py-sr-ey > height)) return false;
    const double C = fabs(double(c.x))+fabs(double(c.y))+fabs(double(c.z));
    const double U = fabs(double(u.x))+fabs(double(u.y))+fabs(double(u.z));
    const double V = fabs(double(v.x))+fabs(double(v.y))+fabs(double(v.z));
    const double R = 1+(width+fabs(double(d.center_x))+fabs(double(shift.x)))/d.focal_x
                      +(height+fabs(double(d.center_y))+fabs(double(shift.y)))/d.focal_y;
    if (!(C < 1e8 && U < 1e8 && V < 1e8 && R < 1e8 &&
          d.focal_x < 1e8 && d.focal_y < 1e8 &&
          isfinite(radius) && radius >= 0 && d.far_plane < 1e8)) return false;
    const double nx = double(u.y)*v.z-double(u.z)*v.y;
    const double ny = double(u.z)*v.x-double(u.x)*v.z;
    const double nz = double(u.x)*v.y-double(u.y)*v.x;
    double lo = INFINITY, hi = -INFINITY;
    for (int x = 0; x <= width; x += width)
        for (int y = 0; y <= height; y += height) {
            const double det = -(nx*(x-double(d.center_x)-shift.x)/d.focal_x
                                +ny*(y-double(d.center_y)-shift.y)/d.focal_y+nz);
            lo = fmin(lo, det); hi = fmax(hi, det);
        }
    if (!(lo > 0 || hi < 0)) return false;  // Singular/near-parallel rays keep the old path.
    const double denom = fmin(fabs(lo), fabs(hi));
    const double ed = 1e-5*U*V*R+1e-30;
    if (!(denom > 2*ed && denom > 1e-20)) return false;
    const double r = double(radius)+1e-3;
    const double ru = r+(1e-5*C*V*R+1e-30+(r+1)*ed)/denom+1e-5*(r+1);
    const double rv = r+(1e-5*C*U*R+1e-30+(r+1)*ed)/denom+1e-5*(r+1);
    const double tn = fabs(double(c.x)*nx+double(c.y)*ny+double(c.z)*nz);
    const double tmax = fmin(double(d.far_plane), (tn+1e-5*C*U*V+1e-30)/(denom-ed));
    const double et = (1e-5*C*U*V+1e-30+(tmax+1)*ed)/denom+1e-5*(tmax+1);
    if (!(et < double(d.near_plane)*0.5)) return false;
    for (int plane = 0; plane < 4; ++plane) {
        const bool horizontal = plane < 2;
        const double f = horizontal ? d.focal_x : d.focal_y;
        const double center = horizontal ? double(d.center_x)+shift.x : double(d.center_y)+shift.y;
        const double size = horizontal ? width : height;
        const double a = (plane % 2 == 0) ? f : -f;
        const double b = (plane % 2 == 0) ? center+1 : size-center+1;
        const double lc = a*(horizontal ? c.x : c.y)+b*c.z;
        const double lu = a*(horizontal ? u.x : u.y)+b*u.z;
        const double lv = a*(horizontal ? v.x : v.y)+b*v.z;
        const double extent = ru*fabs(lu)+rv*fabs(lv);
        if (lc+extent < -1e-10*(fabs(lc)+extent+1)) return true;
    }
    return false;
}

__device__ double3 support_cross(float3 a, float3 b) {
    return make_double3(double(a.y)*b.z-double(a.z)*b.y, double(a.z)*b.x-double(a.x)*b.z,
                        double(a.x)*b.y-double(a.y)*b.x);
}

__device__ bool empty_near_fallback(float3 c, float3 u, float3 v, float2 shift,
    const DiskRasterization& d, int width, int height) {
    // Use a loose support envelope, not another rendering threshold.
    if (config::min_alpha_threshold < 1e-3f || d.focal_x < 1e-4f || d.focal_y < 1e-4f
        || c.z < 1e-4f || d.near_plane < 1e-4f || d.far_plane >= 1e6f) return false;
    const double C = fabs(double(c.x))+fabs(double(c.y))+fabs(double(c.z));
    const double U = fabs(double(u.x))+fabs(double(u.y))+fabs(double(u.z));
    const double V = fabs(double(v.x))+fabs(double(v.y))+fabs(double(v.z));
    const double px = double(d.center_x)+shift.x, py = double(d.center_y)+shift.y;
    const double X = width+fabs(double(d.center_x))+fabs(double(shift.x))+1;
    const double Y = height+fabs(double(d.center_y))+fabs(double(shift.y))+1;
    const double R = 1+X/d.focal_x+Y/d.focal_y;
    if (!(C < 1e6 && U < 1e6 && V < 1e6 && R < 1e6 &&
          d.focal_x < 1e6f && d.focal_y < 1e6f)) return false;
    constexpr double roundoff = 1e-5;
    const double sx = double(d.focal_x)*c.x/c.z, sy = double(d.focal_y)*c.y/c.z;
    const double rx = 3+roundoff*(fabs(sx)+X), ry = 3+roundoff*(fabs(sy)+Y);
    if (!(px+sx+rx < 0 || px+sx-rx > width || py+sy+ry < 0 || py+sy-ry > height)) return false;

    const double3 n = support_cross(u, v);
    const double k = c.x*n.x+c.y*n.y+c.z*n.z;
    const double ek = roundoff*C*U*V+1e-30;
    if (!(fabs(k) > ek+2e-30*d.far_plane)) return false;
    // Accepted positive depth fixes the determinant sign, even if it changes over the image.
    const double s = k > 0 ? 1.0 : -1.0;
    const double ed = roundoff*U*V*R+1e-30;
    double3 planes[4];
    double errors[4];
    int index = 0;
    for (int axis = 0; axis < 2; ++axis) {
        const double3 q = axis == 0 ? support_cross(v, c) : support_cross(c, u);
        const double eq = roundoff*C*(axis == 0 ? V : U)*R+1e-30;
        for (int sign = -1; sign <= 1; sign += 2) {
            const double ax = s*(4*n.x+sign*q.x), ay = s*(4*n.y+sign*q.y);
            const double az = s*(4*n.z+sign*q.z);
            planes[index] = make_double3(ax, ay, az);
            errors[index++] = 4*ed+eq;
            const double x = ((ax > 0 ? width : 0)-px)/d.focal_x;
            const double y = ((ay > 0 ? height : 0)-py)/d.focal_y;
            if (ax*x+ay*y+az < -(4*ed+eq)) return true;
        }
    }
    // A nonnegative combination preserves every accepted hit's old support inequalities.
    const double ray_bound = 1+fmax(fabs(px/d.focal_x), fabs((width-px)/d.focal_x))
                              +fmax(fabs(py/d.focal_y), fabs((height-py)/d.focal_y));
    for (int i = 0; i < 4; ++i) for (int j = i+1; j < 4; ++j)
        for (int axis = 0; axis < 2; ++axis) {
            const double a = axis == 0 ? planes[i].x : planes[i].y;
            const double b = axis == 0 ? planes[j].x : planes[j].y;
            if (!(a*b < 0)) continue;
            const double denominator = fabs(a)+fabs(b);
            const double wi = fabs(b)/denominator, wj = fabs(a)/denominator;
            const double ax = wi*planes[i].x+wj*planes[j].x;
            const double ay = wi*planes[i].y+wj*planes[j].y;
            const double az = wi*planes[i].z+wj*planes[j].z;
            const double error = wi*errors[i]+wj*errors[j];
            const double magnitude = wi*(fabs(planes[i].x)+fabs(planes[i].y)+fabs(planes[i].z))
                                    +wj*(fabs(planes[j].x)+fabs(planes[j].y)+fabs(planes[j].z));
            const double padded_error = error+1e-12*(ray_bound*magnitude+error+1e-30);
            const double x = ((ax > 0 ? width : 0)-px)/d.focal_x;
            const double y = ((ay > 0 ? height : 0)-py)/d.focal_y;
            if (ax*x+ay*y+az < -padded_error) return true;
        }
    return false;
}

struct Hit { float u, v, depth, gaussian, alpha, mapped_depth; bool screen; };
template<bool replay_hit = false>
__device__ bool intersect(const Disk&, float3, const DiskRasterization&, Hit&);

// A clipped empty rectangle can round out to a partial screen-edge tile.
// Check its remaining pixels once per disk, before building long tile lists.
__device__ bool empty_edge_tile(ushort4 bounds, uint4 tiles, const DiskRasterization& d,
    int i, int width, int height) {
    if ((bounds.x != bounds.y && bounds.z != bounds.w)
        || tiles.y != tiles.x+1 || tiles.w != tiles.z+1) return false;
    const int x0 = tiles.x*config::tile_width, y0 = tiles.z*config::tile_height;
    const int x1 = min(width, x0+config::tile_width), y1 = min(height, y0+config::tile_height);
    if ((x1-x0)*(y1-y0) > 64) return false;
    const Disk disk = d.geometry[i];
    for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
        const float3 ray = make_float3((x+0.5f-d.center_x)/d.focal_x,
            (y+0.5f-d.center_y)/d.focal_y, 1.0f);
        Hit h;
        if (intersect(disk, shifted_ray(ray, d, i), d, h)) return false;
    }
    return true;
}

__global__ void disk_preprocess(
    const float3* means, const float2* scales, const float4* rotations,
    const float* opacities, const float3* sh0, const float3* sh_rest,
    const float4* w, const float3* cam, PrimitiveBuffers p,
    DiskRasterization d, int n, int active_sh, int total_sh, int width, int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p.depth_keys.Current()[i] = __float_as_uint(INFINITY);
    p.primitive_indices.Current()[i] = i;
    p.n_touched_tiles[i] = 0;
    const float2 shift = d.image_shift ? d.image_shift[i] : make_float2(0);
    if (!isfinite(shift.x) || !isfinite(shift.y)) return;
    const float3 c = camera_vector(w, means[i]) + make_float3(w[0].w, w[1].w, w[2].w);
    const float opacity = sigmoid(opacities[i]);
    if (!finite3(c) || !isfinite(opacity) || c.z < d.near_plane || c.z > d.far_plane
        || opacity < config::min_alpha_threshold) return;
    const float4 q = rotations[i];
    if (!isfinite(dot(q,q)) || dot(q, q) < 1e-8f) return;
    float norm;
    const mat3x3 R = convert_quaternion_to_rotation_matrix(q, norm);
    const float3 normal = make_float3(R.m13, R.m23, R.m33);
    const float sign = dot(camera_vector(w, normal), c) > 0.0f ? -1.0f : 1.0f;
    const float3 u = expf(scales[i].x) * camera_vector(w, make_float3(R.m11, R.m21, R.m31));
    const float3 v = expf(scales[i].y) * camera_vector(w, make_float3(R.m12, R.m22, R.m32));
    if (!finite3(u) || !finite3(v)) return;
    d.geometry[i] = {c, u, v, sign * normal, opacity};

    // The projected local square contains the entire alpha-truncated disk.
    const float radius = sqrtf(2.0f * logf(opacity * config::min_alpha_threshold_rcp));
    float xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
    bool crosses_near = false;
    for (int a = -1; a <= 1; a += 2) for (int b = -1; b <= 1; b += 2) {
        const float3 corner = c + radius * (float(a) * u + float(b) * v);
        if (!finite3(corner)) return;
        crosses_near |= corner.z <= d.near_plane;
        const float x = d.focal_x * corner.x / corner.z + d.center_x + shift.x;
        const float y = d.focal_y * corner.y / corner.z + d.center_y + shift.y;
        xmin = fminf(xmin, x); xmax = fmaxf(xmax, x);
        ymin = fminf(ymin, y); ymax = fmaxf(ymax, y);
    }
    if (crosses_near) { xmin = ymin = 0; xmax = width; ymax = height; }
    // Eq.11-12: union disk support with sigma=sqrt(1/2)-pixel screen support.
    const float screen_radius = radius * 0.7071067811865475f;
    const float center_x = d.focal_x*c.x/c.z+d.center_x+shift.x;
    const float center_y = d.focal_y*c.y/c.z+d.center_y+shift.y;
    xmin = fminf(xmin, center_x-screen_radius); xmax = fmaxf(xmax, center_x+screen_radius);
    ymin = fminf(ymin, center_y-screen_radius); ymax = fmaxf(ymax, center_y+screen_radius);
    const ushort4 bounds = make_ushort4(
        int(fmaxf(0, fminf(width, floorf(xmin)))),
        int(fmaxf(0, fminf(width, ceilf(xmax)))),
        int(fmaxf(0, fminf(height, floorf(ymin)))),
        int(fmaxf(0, fminf(height, ceilf(ymax)))));
    const uint4 tiles = convert_screen_bounds_to_tile_bounds(bounds);
    const uint count = (tiles.y - tiles.x) * (tiles.w - tiles.z);
    // Empty tile ranges cannot contribute, so only certify the remaining disks.
    if (!count) return;
    if (outside_support(c, u, v, radius, shift, d, width, height)) return;
    if (d.metadata) d.metadata[i] = fmaxf(
        fmaxf(fabsf(bounds.x-center_x), fabsf(bounds.y-center_x)),
        fmaxf(fabsf(bounds.z-center_y), fabsf(bounds.w-center_y)));
    if (crosses_near && empty_near_fallback(c, u, v, shift, d, width, height)) return;
    if (empty_edge_tile(bounds, tiles, d, i, width, height)) return;
    p.n_touched_tiles[i] = count;
    p.screen_bounds[i] = bounds;
    p.mean2d[i] = make_float2(0.0f);
    // Conservative tile tests only; this conic is NEVER used to render a disk.
    p.conic_opacity[i] = make_float4(1e-20f, 0.0f, 1e-20f, 1.0f);
    p.color[i] = convert_sh_to_color(sh0, sh_rest, means[i], cam[0], i, active_sh, total_sh);
    // Stable depth ties inherit row IDs, never atomic arrival order.
    atomicAdd(&p.counters[0].x, 1);
    p.depth_keys.Current()[i] = __float_as_uint(c.z);
    atomicAdd(d.instance_counter, static_cast<unsigned long long>(count));
}

template<bool replay_hit>
__device__ bool intersect(const Disk& d, float3 ray, const DiskRasterization& settings, Hit& h) {
    // C + uU + vV = t*ray; Cramer's rule, no 3D covariance or third axis.
    const float3 a = cross(d.tangent_v, -ray);
    const float3 b = cross(-ray, d.tangent_u);
    const float3 c = cross(d.tangent_u, d.tangent_v);
    const float determinant = dot(d.tangent_u, a);
    if constexpr (!replay_hit) {
        const float determinant_scale = length(d.tangent_u)*length(d.tangent_v);
        if (!(determinant_scale > 0) || !isfinite(determinant_scale) || !isfinite(determinant)
            || determinant == 0 || fabsf(determinant)/determinant_scale < 1e-10f) return false;
    }
    h.u = -dot(d.center, a) / determinant;
    h.v = -dot(d.center, b) / determinant;
    h.depth = -dot(d.center, c) / determinant;
    if (!replay_hit && (h.depth < settings.near_plane || h.depth > settings.far_plane)) return false;
    const float disk_gaussian = expf(-0.5f * (h.u*h.u + h.v*h.v));
    const float2 delta = make_float2(settings.focal_x*(ray.x-d.center.x/d.center.z),
                                    settings.focal_y*(ray.y-d.center.y/d.center.z));
    const float screen_gaussian = expf(-dot(delta, delta));  // sigma^2 = 1/2
    h.screen = screen_gaussian > disk_gaussian;  // exact ties retain disk adjoints
    h.gaussian = h.screen ? screen_gaussian : disk_gaussian;
    h.alpha = fminf(0.99f, d.opacity * h.gaussian);
    // Pairwise distortion is translation-invariant; omit the common near-1 offset.
    h.mapped_depth = -settings.far_plane*settings.near_plane
        / ((settings.far_plane-settings.near_plane)*h.depth);
    return replay_hit || h.alpha >= config::min_alpha_threshold;
}

// Each tile owns ceil(entries/32) words per pixel. The tile index provides
// enough padding for an unaligned start without a separate prefix scan.
__device__ size_t disk_mask_index(uint range_start, uint tile) {
    const uint local = threadIdx.y * config::tile_width + threadIdx.x;
    return size_t(range_start / 32 + tile) * config::block_size_blend + local;
}

// Specify the rounding order of the ray solve. A change in load order can otherwise
// change which product the compiler contracts and flip a hit at the alpha cutoff.
__device__ __forceinline__ float disk_dot_xy(float3 a, float3 b) {
    return __fmaf_rn(a.z, b.z, __fmaf_rn(a.y, b.y, __fmul_rn(a.x, b.x)));
}

__device__ __forceinline__ float disk_dot_yx(float3 a, float3 b) {
    return __fmaf_rn(a.z, b.z, __fmaf_rn(a.x, b.x, __fmul_rn(a.y, b.y)));
}

__device__ bool intersect_staged(const Disk& d, float2 fixed, float3 ray, const DiskRasterization& settings, Hit& h) {
    const float3 u = d.tangent_u, v = d.tangent_v;
    const float3 a = make_float3(__fmaf_rn(v.z, ray.y, -v.y), __fmaf_rn(-v.z, ray.x, v.x),
                                 __fmaf_rn(v.y, ray.x, -__fmul_rn(v.x, ray.y)));
    const float3 b = make_float3(__fmaf_rn(-u.z, ray.y, u.y), __fmaf_rn(u.z, ray.x, -u.x),
                                 __fmaf_rn(u.x, ray.y, -__fmul_rn(u.y, ray.x)));
    const float det = disk_dot_xy(u, a);
    const float scale = fixed.x;
    if (!(scale > 0) || !isfinite(scale) || !isfinite(det) || det == 0 || fabsf(det)/scale < 1e-10f) return false;
    h.depth = fixed.y/det;
    if (h.depth < settings.near_plane || h.depth > settings.far_plane) return false;
    h.u = -disk_dot_xy(d.center,a)/det;
    h.v = -disk_dot_xy(d.center,b)/det;
    const float disk_log = -0.5f * __fmaf_rn(h.u,h.u,__fmul_rn(h.v,h.v));
    const float2 delta = make_float2(settings.focal_x*(ray.x-d.center.x/d.center.z),
                                     settings.focal_y*(ray.y-d.center.y/d.center.z));
    const float screen_log = -__fmaf_rn(delta.x,delta.x,__fmul_rn(delta.y,delta.y));
    h.gaussian = expf(screen_log > disk_log ? screen_log : disk_log);
    h.alpha = fminf(0.99f, d.opacity*h.gaussian);
    return h.alpha >= config::min_alpha_threshold;
}

template<int channels, bool growth>
__global__ void disk_blend_staged(
    PrimitiveBuffers p, TileBuffers tiles, const uint* indices, uint* hit_masks,
    const float3* bg, float* image, DiskRasterization d, int width, int height)
{
    // Each warp covers an 8x4 subtile. Keep the saved hit masks in raster order for backward.
    static_assert(config::tile_width == 16 && config::tile_height == 12);
    const uint thread = threadIdx.y * config::tile_width + threadIdx.x;
    const int lx = ((thread >> 5) % 2) * 8 + (thread % 8);
    const int ly = ((thread >> 5) / 2) * 4 + ((thread % 32) / 8);
    const int x = blockIdx.x * 16 + lx, y = blockIdx.y * 12 + ly;
    const int grid_width = div_round_up(width, config::tile_width);
    const int tile = blockIdx.y * grid_width + blockIdx.x;
    const uint2 range = tiles.instance_ranges[tile];
    constexpr uint batch_size = 64;
    // The determinant scale and plane numerator are common to every pixel in the tile.
    __shared__ float2 fixed[batch_size];
    static_assert(batch_size % 32 == 0, "staged batches must contain whole hit-mask groups");
    __shared__ ushort4 screen[batch_size];
    __shared__ Disk geometry[batch_size];
    __shared__ float3 colors[batch_size];
    __shared__ int ids[batch_size];
    __shared__ float2 shifts[batch_size];
    const uint lane = threadIdx.y * blockDim.x + threadIdx.x;
    const bool inside = x < width && y < height;
    bool done = !inside;
    const int pixel = y * width + x, pixels = width * height;
    const float3 base_ray = make_float3((x+0.5f-d.center_x)/d.focal_x,
                                  (y+0.5f-d.center_y)/d.focal_y, 1.0f);
    float T = 1, depth = 0, alpha = 0, median = 0, mean_depth = 0, central_moment = 0;
    float3 rgb = make_float3(0), normal = make_float3(0);
    uint processed = 0;
    const size_t mask_base = size_t(range.x / 32 + tile) * config::block_size_blend + ly * config::tile_width + lx;
    for (uint start = range.x; start < range.y; start += batch_size) {
        if (__syncthreads_and(done)) break;
        const uint count = min(batch_size, range.y - start);
        if (lane < count) {
            const int i = indices[start + lane];
            ids[lane] = i;
            screen[lane] = p.screen_bounds[i];
            const Disk disk = d.geometry[i];
            geometry[lane] = disk;
            const float3 u = disk.tangent_u, v = disk.tangent_v;
            const float3 c = make_float3(__fmaf_rn(u.y,v.z,-__fmul_rn(u.z,v.y)),
                                         __fmaf_rn(u.z,v.x,-__fmul_rn(u.x,v.z)),
                                         __fmaf_rn(u.x,v.y,-__fmul_rn(u.y,v.x)));
            fixed[lane] = make_float2(sqrtf(disk_dot_yx(u,u)) * sqrtf(disk_dot_yx(v,v)), -disk_dot_yx(disk.center,c));
            colors[lane] = fmaxf(p.color[i], 0.0f);
            if (growth && d.image_shift) shifts[lane] = d.image_shift[i];
        }
        __syncthreads();
        const int local_x = blockIdx.x * 16 + ((lane >> 5) % 2) * 8;
        const int local_y = blockIdx.y * 12 + ((lane >> 5) / 2) * 4;
        // Expand the projected disk and screen-space low-pass support before rejection.
        constexpr int padding = 2;
        uint keep[2];
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int j = group * 32 + (lane & 31);
            bool overlap = false;
            if (j < count) {
                const ushort4 box = screen[j];
                overlap = int(box.x) <= local_x + 7 + padding && int(box.y) >= local_x - padding
                       && int(box.z) <= local_y + 3 + padding && int(box.w) >= local_y - padding;
            }
            keep[group] = __ballot_sync(0xffffffffu, overlap);
        }
        // Backward reads one 32-bit hit mask per pixel and group of disks.
        for (uint first = 0; !done && first < count; first += 32) {
            uint hits = 0;
            const uint candidates = keep[first / 32];
            #pragma unroll 4
            for (uint bit = 0; bit < 32; ++bit) {
                if (done || bit + first >= count) break;
                if (!(candidates & (1u << bit))) continue;
                const uint j = first + bit;
                const int i = ids[j];
                const Disk disk = geometry[j];
                float3 ray = base_ray;
                if (growth && d.image_shift) {
                    ray.x -= shifts[j].x/d.focal_x;
                    ray.y -= shifts[j].y/d.focal_y;
                }
                Hit h;
                if (!intersect_staged(disk, fixed[j], ray, d, h)) continue;
                hits |= 1u << (j - first);
                if (growth && d.metadata) {
                    // Visibility is monotonic over this render; all accepted hits publish one.
                    cuda::atomic_ref<float, cuda::thread_scope_device> visible(d.metadata[d.n_primitives+i]);
                    if (visible.load(cuda::memory_order_relaxed) == 0.0f)
                        visible.store(1.0f, cuda::memory_order_relaxed);
                }
                const float weight = T * h.alpha;
                rgb += weight * colors[j];
                if constexpr (channels >= 8) {
                    depth += weight * h.depth;
                    normal += weight * disk.normal;
                }
                if constexpr (channels >= 9) {
                    if (T > 0.5f && h.depth > median) median = h.depth;
                }
                if constexpr (channels == 10) {
                    const float mapped_depth = -d.far_plane*d.near_plane / ((d.far_plane-d.near_plane)*h.depth);
                    const float next_alpha = alpha + weight, delta = mapped_depth-mean_depth;
                    central_moment += weight*alpha/next_alpha*delta*delta;
                    mean_depth = alpha == 0 ? mapped_depth : mean_depth+weight/next_alpha*delta;
                    alpha = next_alpha;
                } else if constexpr (channels >= 8) {
                    alpha += weight;
                }
                T *= 1.0f-h.alpha;
                processed = start - range.x + j + 1;
                done = T < config::transmittance_threshold;
            }
            const uint group = (start - range.x + first) >> 5;
            hit_masks[mask_base + size_t(group) * config::block_size_blend] = hits;
        }
    }
    if (!inside) return;
    rgb += T * bg[0];
    image[pixel] = rgb.x; image[pixels+pixel] = rgb.y; image[2*pixels+pixel] = rgb.z;
    d.aux[pixel] = depth; d.aux[pixels+pixel] = alpha;
    d.aux[2*pixels+pixel] = normal.x; d.aux[3*pixels+pixel] = normal.y;
    d.aux[4*pixels+pixel] = normal.z; d.aux[5*pixels+pixel] = median;
    d.aux[6*pixels+pixel] = alpha*central_moment;
    d.aux[7*pixels+pixel] = mean_depth; d.aux[8*pixels+pixel] = central_moment;
    tiles.final_transmittances[pixel] = T;
    tiles.n_processed[pixel] = processed;
}

// Reduce dense four-lane groups that write the same disk. Sparse and edge
// groups retain the scalar atomic path.
__device__ __forceinline__ void warp_gradient_add(float* address, float value, uint peers) {
    const uint lane = (threadIdx.x + blockDim.x * threadIdx.y) & 31u;
    const uint active = peers & __activemask();
    const uint octet = 255u << (lane & ~7u);
    const uint quad = 15u << (lane & ~3u);
    if ((active & octet) == octet) {
        value += __shfl_xor_sync(octet, value, 1);
        value += __shfl_xor_sync(octet, value, 2);
        value += __shfl_xor_sync(octet, value, 4);
        if ((lane & 7u) != 0) return;
    } else if ((active & quad) == quad) {
        value += __shfl_xor_sync(quad, value, 1);
        value += __shfl_xor_sync(quad, value, 2);
        if ((lane & 3u) != 0) return;
    }
    atomicAdd(address, value);
}

// Pack the RGB adjoints into the fourth component of the disk vectors.
struct alignas(16) PackedDiskGradient {
    float4 center_opacity, tangent_u_red, tangent_v_green, normal_blue;
};
static_assert(sizeof(PackedDiskGradient) == 16 * sizeof(float));

struct TileGradient {
    PackedDiskGradient grad;
    float2 shift;
    int id;
};

#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12020
constexpr bool use_tile_gradient_cache = false;
#else
constexpr bool use_tile_gradient_cache = true;
#endif

__device__ __forceinline__ void atomic_add_gradient(float4* address, float4 value, bool shared) {
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12020
    // Vector atomicAdd is available for global memory only.
    if (!shared) {
        atomicAdd(address, value);
        return;
    }
#endif
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
    atomicAdd(&address->w, value.w);
}

template<int width>
__device__ __forceinline__ void reduce_gradient(float4* address, float4 value, uint lane, uint mask, bool shared) {
    #pragma unroll
    for (int step = 1; step < width; step *= 2) {
        value.x += __shfl_xor_sync(mask, value.x, step);
        value.y += __shfl_xor_sync(mask, value.y, step);
        value.z += __shfl_xor_sync(mask, value.z, step);
        value.w += __shfl_xor_sync(mask, value.w, step);
    }
    if ((lane & (width - 1)) == 0) atomic_add_gradient(address, value, shared);
}

__device__ __forceinline__ void accumulate_gradient(float4* address, float4 value, uint peers, bool shared) {
    const uint lane = (threadIdx.x + blockDim.x * threadIdx.y) & 31u;
    const uint active = peers & __activemask();
    const uint octet = 255u << (lane & ~7u);
    const uint quad = 15u << (lane & ~3u);
    if ((active & octet) == octet) reduce_gradient<8>(address, value, lane, octet, shared);
    else if ((active & quad) == quad) reduce_gradient<4>(address, value, lane, quad, shared);
    else atomic_add_gradient(address, value, shared);
}

constexpr unsigned tile_gradient_slots = 64;

__device__ void replay_pixel(TileGradient* cache,
    PrimitiveBuffers p, TileBuffers tiles, const uint* indices, const uint* hit_masks, const Disk* geometry,
    const float3* bg, const float* image, const float* grad, PackedDiskGradient* geometry_grad,
    float2* shift_grad, DiskRasterization d, int width, int height)
{
    const int x = blockIdx.x * config::tile_width + threadIdx.x;
    const int y = blockIdx.y * config::tile_height + threadIdx.y;
    if (x >= width || y >= height) return;
    const int pixel = y*width+x, pixels = width*height;
    const int tile = blockIdx.y * div_round_up(width, config::tile_width) + blockIdx.x;
    const uint2 range = tiles.instance_ranges[tile];
    const float3 base_ray = make_float3((x+0.5f-d.center_x)/d.focal_x,
                                  (y+0.5f-d.center_y)/d.focal_y, 1.0f);
    const float3 g_rgb = make_float3(grad[pixel], grad[pixels+pixel], grad[2*pixels+pixel]);
    const float g_depth = d.output_channels >= 8 ? grad[3*pixels+pixel] : 0.0f;
    const float g_alpha = d.output_channels >= 8 ? grad[4*pixels+pixel] : 0.0f;
    const float3 g_normal = d.output_channels >= 8
        ? make_float3(grad[5*pixels+pixel], grad[6*pixels+pixel], grad[7*pixels+pixel]) : make_float3(0.0f);
    const float g_median = d.output_channels >= 9 ? grad[8*pixels+pixel] : 0.0f;
    const float g_dist = d.output_channels == 10 ? grad[9*pixels+pixel] : 0.0f;
    const float M0 = image[4*pixels+pixel], mean_depth = image[10*pixels+pixel], central_moment = image[11*pixels+pixel];
    const float final_T = tiles.final_transmittances[pixel];
    float T = final_T, suffix = final_T * dot(bg[0], g_rgb);
    // Eq.19 selects max intersection z, not the last centre-sorted hit; ties keep the first.
    int median_winner = -1;
    const int processed = tiles.n_processed[pixel];
    float median_T = 1, median_depth = 0;
    // Only the median-depth adjoint needs the identity of the winning hit.
    for (int k = 0; g_median != 0.0f && k < processed && median_T > 0.5f; ++k) {
        Hit h;
        const int i = indices[range.x+k];
        if (!intersect(geometry[i], shifted_ray(base_ray, d, i), d, h)) continue;
        if (h.depth > median_depth) { median_depth = h.depth; median_winner = k; }
        median_T *= 1.0f-h.alpha;
    }
    const size_t mask_base = disk_mask_index(range.x, tile);
    const auto warp = cg::coalesced_threads();
    const int pixel_groups = (processed + 31) / 32;
    int group = cg::reduce(warp, pixel_groups, cg::greater<int>());
    uint remaining = 0, hits = 0;
    const uint lane = (threadIdx.x + blockDim.x*threadIdx.y) & 31u;
    int staged_index = 0;
    const bool full_warp = warp.size() == 32;
    while (true) {
        while (!remaining && group > 0) {
            --group;
            hits = group < pixel_groups ? hit_masks[mask_base + size_t(group) * config::block_size_blend] : 0u;
            remaining = cg::reduce(warp, hits, cg::bit_or<uint>());
            // One lane looks up each referenced disk for this 32-entry group.
            // Edge warps can lack the owner lane, so they retain direct indexed loads.
            if (full_warp && (remaining & (1u << lane))) {
                staged_index = indices[range.x + 32*group + lane];
                const char* record = reinterpret_cast<const char*>(geometry + staged_index);
                asm volatile("prefetch.global.L2 [%0];" :: "l"(record));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(record + sizeof(Disk) - 1));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(p.color + staged_index));
            }
        }
        if (!remaining) break;
        const int bit = 31 - __clz(remaining);
        remaining ^= 1u << bit;
        const int k = 32 * group + bit;
        const int i = full_warp ? __shfl_sync(0xffffffffu, staged_index, bit) : indices[range.x+k];
        if (!(hits & (1u << bit))) continue;
        const Disk disk = geometry[i];
        const float3 ray = shifted_ray(base_ray, d, i);
        Hit h;
        // The saved forward bit certifies this exact pixel, ray and disk.
        intersect<true>(disk, ray, d, h);
        T /= 1.0f-h.alpha;
        const float weight = T*h.alpha, centered = h.mapped_depth-mean_depth;
        const float3 color = fmaxf(p.color[i], 0.0f);
        const float gw = dot(color, g_rgb) + g_depth*h.depth + g_alpha
            + dot(disk.normal, g_normal) + g_dist*(centered*centered*M0+central_moment);
        const float ga = T*gw-suffix/(1.0f-h.alpha);
        suffix += weight*gw;
        float gt = weight*g_depth + g_dist*2*weight*M0*centered
            * d.far_plane*d.near_plane/((d.far_plane-d.near_plane)*h.depth*h.depth);
        if (k == median_winner) gt += g_median;
        float gu = 0, gv = 0;
        float2 screen_shift_grad = make_float2(0);
        float3 screen_center_grad = make_float3(0);
        // Large footprints send many pixel gradients to the same global row.
        // Accumulate these rows per tile, falling back on a hash collision.
        PackedDiskGradient* dg = geometry_grad + i;
        bool shared = false;
        float2* ds = shift_grad ? shift_grad + i : nullptr;
        const ushort4 bounds = p.screen_bounds[i];
        if (use_tile_gradient_cache && unsigned(bounds.y - bounds.x) * unsigned(bounds.w - bounds.z) >= 4096u) {
            TileGradient& slot = cache[unsigned(i) % tile_gradient_slots];
            cuda::atomic_ref<int, cuda::thread_scope_block> key(slot.id);
            int owner = key.load(cuda::memory_order_relaxed);
            if (owner == -1 && key.compare_exchange_strong(owner, i, cuda::memory_order_relaxed))
                owner = i;
            if (owner == i) {
                dg = &slot.grad;
                shared = true;
                ds = &slot.shift;
            }
        }
#if __CUDA_ARCH__ >= 700
        const uint peers = __match_any_sync(__activemask(), reinterpret_cast<unsigned long long>(dg));
#else
        const uint peers = 0;
#endif
        float opacity_grad = 0.0f;
        if (disk.opacity*h.gaussian < 0.99f) {
            if (h.screen) {
                const float jx = d.focal_x/disk.center.z, jy = d.focal_y/disk.center.z;
                const float gx = 2*h.alpha*ga*d.focal_x*(ray.x-disk.center.x/disk.center.z);
                const float gy = 2*h.alpha*ga*d.focal_y*(ray.y-disk.center.y/disk.center.z);
                screen_shift_grad = make_float2(gx, gy);
                screen_center_grad = make_float3(gx*jx, gy*jy,
                    -(gx*jx*disk.center.x+gy*jy*disk.center.y)/disk.center.z);
            } else {
                gu = -h.alpha*ga*h.u;
                gv = -h.alpha*ga*h.v;
            }
            opacity_grad = ga*h.gaussian;
        }
        const float3 a = cross(disk.tangent_v, -1.0f*ray), b = cross(-1.0f*ray, disk.tangent_u);
        const float3 c = cross(disk.tangent_u, disk.tangent_v);
        const float3 lambda = (gu*a+gv*b+gt*c)/dot(disk.tangent_u, a);
        if (shift_grad) {
            warp_gradient_add(&ds->x, -h.depth*lambda.x/d.focal_x+screen_shift_grad.x, peers);
            warp_gradient_add(&ds->y, -h.depth*lambda.y/d.focal_y+screen_shift_grad.y, peers);
        }
        const bool plane = gu != 0.0f || gv != 0.0f || gt != 0.0f;
        const float3 center = plane ? -1.0f*lambda + screen_center_grad : screen_center_grad;
        const float3 tangent_u = plane ? -h.u*lambda : make_float3(0);
        const float3 tangent_v = plane ? -h.v*lambda : make_float3(0);
        const float3 normal = weight*g_normal;
        const float3 raw = p.color[i];
        const float3 rgb = weight*g_rgb*make_float3(raw.x >= 0, raw.y >= 0, raw.z >= 0);
        accumulate_gradient(&dg->center_opacity, make_float4(center, opacity_grad), peers, shared);
        accumulate_gradient(&dg->tangent_u_red, make_float4(tangent_u, rgb.x), peers, shared);
        accumulate_gradient(&dg->tangent_v_green, make_float4(tangent_v, rgb.y), peers, shared);
        accumulate_gradient(&dg->normal_blue, make_float4(normal, rgb.z), peers, shared);
    }
}

__global__ void disk_blend_backward(
    PrimitiveBuffers p, TileBuffers tiles, const uint* indices, const uint* hit_masks, const Disk* geometry,
    const float3* bg, const float* image, const float* grad, PackedDiskGradient* geometry_grad,
    float2* shift_grad, DiskRasterization d, int width, int height)
{
    if constexpr (!use_tile_gradient_cache) {
        // A direct global float4 atomic costs less here than the tile hash and flush.
        replay_pixel(nullptr, p, tiles, indices, hit_masks, geometry, bg, image, grad,
            geometry_grad, shift_grad, d, width, height);
    } else {
        __shared__ TileGradient cache[tile_gradient_slots];
        const unsigned lane = threadIdx.y * blockDim.x + threadIdx.x;
        if (lane < tile_gradient_slots) {
            cache[lane] = {};
            cache[lane].id = -1;
        }
        __syncthreads();
        replay_pixel(cache, p, tiles, indices, hit_masks, geometry, bg, image, grad,
            geometry_grad, shift_grad, d, width, height);
        __syncthreads();
        if (lane >= tile_gradient_slots) return;
        const TileGradient& slot = cache[lane];
        if (slot.id < 0) return;
        PackedDiskGradient& dg = geometry_grad[slot.id];
        atomic_add_gradient(&dg.center_opacity, slot.grad.center_opacity, false);
        atomic_add_gradient(&dg.tangent_u_red, slot.grad.tangent_u_red, false);
        atomic_add_gradient(&dg.tangent_v_green, slot.grad.tangent_v_green, false);
        atomic_add_gradient(&dg.normal_blue, slot.grad.normal_blue, false);
        if (shift_grad) {
            atomicAdd(&shift_grad[slot.id].x, slot.shift.x);
            atomicAdd(&shift_grad[slot.id].y, slot.shift.y);
        }
    }
}

template<bool collect_sh>
__global__ void disk_preprocess_backward(
    const float3* means, const float2* scales, const float4* rotations,
    const float3* rest, const float4* w, const float3* cam,
    const uint* touched, const Disk* geometry, const PackedDiskGradient* grad_geometry,
    float3* grad_means, float2* grad_scales, float4* grad_rotations,
    float* grad_opacities, float3* grad_sh0, float3* grad_rest,
    int n, int active_sh, int total_sh)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    constexpr int degree_three_rest = 15;
    constexpr int row_width = 3 * degree_three_rest;
    extern __shared__ __align__(16) float sh_storage[];
    const bool active = i < n && touched[i];
    if constexpr (!collect_sh) {
        if (!active) return;
    }
    float3* row = collect_sh ? reinterpret_cast<float3*>(sh_storage) + threadIdx.x*degree_three_rest
                             : grad_rest + size_t(i)*total_sh;
    if (active) {
        const Disk d = geometry[i];
        const PackedDiskGradient v = grad_geometry[i];
        const Disk g = {make_float3(v.center_opacity), make_float3(v.tangent_u_red), make_float3(v.tangent_v_green),
            make_float3(v.normal_blue), v.center_opacity.w};
        grad_sh0[i] = make_float3(v.tangent_u_red.w, v.tangent_v_green.w, v.normal_blue.w);
        const float3 gc = world_adjoint(w, g.center);
        const float3 gu = expf(scales[i].x)*world_adjoint(w, g.tangent_u);
        const float3 gv = expf(scales[i].y)*world_adjoint(w, g.tangent_v);
        float norm;
        const mat3x3 R = convert_quaternion_to_rotation_matrix(rotations[i], norm);
        const float sign = dot(d.normal, make_float3(R.m13, R.m23, R.m33)) > 0 ? 1.0f : -1.0f;
        const float3 gn = sign*g.normal;
        const mat3x3 gR = {gu.x, gv.x, gn.x, gu.y, gv.y, gn.y, gu.z, gv.z, gn.z};
        grad_rotations[i] = convert_quaternion_to_rotation_matrix_backward(rotations[i], gR);
        grad_scales[i] = make_float2(dot(g.tangent_u, d.tangent_u), dot(g.tangent_v, d.tangent_v));
        grad_opacities[i] = g.opacity*d.opacity*(1.0f-d.opacity);
        grad_means[i] = gc + convert_sh_to_color_backward(
            rest, grad_sh0, row, means[i], cam[0], i, active_sh, total_sh);
    } else if (i < n) {
        grad_means[i] = make_float3(0);
        grad_scales[i] = make_float2(0);
        grad_rotations[i] = make_float4(0);
        grad_opacities[i] = 0;
    }
    if constexpr (collect_sh) {
        for (int k = active ? active_sh-1 : 0; k < degree_three_rest; ++k) row[k] = make_float3(0);
        __syncwarp();
        // A warp starts on a 16-byte boundary, including the last partial warp.
        const int lane = threadIdx.x % 32;
        const int first = i - lane;
        if (first >= n) return;
        const int count = min(32, n-first) * row_width;
        const float* local = sh_storage + (threadIdx.x-lane)*row_width;
        float* dest = reinterpret_cast<float*>(grad_rest) + size_t(first)*row_width;
        for (int k = lane; k < count/4; k += 32)
            reinterpret_cast<float4*>(dest)[k] = reinterpret_cast<const float4*>(local)[k];
        for (int k = (count/4)*4 + lane; k < count; k += 32) dest[k] = local[k];
    }
}

void check_tensor(const torch::Tensor& t, const torch::Tensor& means, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == means.device() && t.scalar_type() == torch::kFloat
        && t.is_contiguous(), name, " must be contiguous float32 on the means CUDA device");
}

void check_camera(int width, int height, float fx, float fy, float cx, float cy, float near, float far) {
    TORCH_CHECK(width > 0 && height > 0 && width <= 65520 && height <= 65520
        && int64_t(width)*height <= INT_MAX,
        "disk image dimensions must be in [1,65520] with at most INT_MAX pixels");
    TORCH_CHECK(std::isfinite(fx) && std::isfinite(fy) && fx > 0 && fy > 0
        && std::isfinite(cx) && std::isfinite(cy) && std::isfinite(near)
        && std::isfinite(far) && near > 0 && far > near, "invalid disk camera");
}
}

void preprocess_disks(
    const float3* means, const float2* scales, const float4* rotations,
    const float* opacities, const float3* sh0, const float3* rest,
    const float4* w, const float3* cam, PrimitiveBuffers& p,
    const DiskRasterization& d, int n, int active_sh, int total_sh,
    int width, int height, cudaStream_t stream)
{
    disk_preprocess<<<div_round_up(n, 256), 256, 0, stream>>>(
        means, scales, rotations, opacities, sh0, rest, w, cam, p, d,
        n, active_sh, total_sh, width, height);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void blend_disks(
    const PrimitiveBuffers& p, const TileBuffers& tiles, const uint* indices,
    uint* hit_masks, const float3* bg, float* image,
    const DiskRasterization& d, int width, int height, cudaStream_t stream)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height));
    const dim3 block(config::tile_width, config::tile_height);
    auto launch = [&](auto channels) {
        if (d.metadata || d.image_shift) {
            disk_blend_staged<decltype(channels)::value, true><<<grid, block, 0, stream>>>(p, tiles, indices, hit_masks, bg, image, d, width, height);
        } else {
            disk_blend_staged<decltype(channels)::value, false><<<grid, block, 0, stream>>>(p, tiles, indices, hit_masks, bg, image, d, width, height);
        }
    };
    switch (d.output_channels) {
        case 3: launch(std::integral_constant<int,3>{}); break;
        case 8: launch(std::integral_constant<int,8>{}); break;
        case 9: launch(std::integral_constant<int,9>{}); break;
        default: launch(std::integral_constant<int,10>{}); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> disk_forward_wrapper(
    const torch::Tensor& means, const torch::Tensor& scales,
    const torch::Tensor& rotations, const torch::Tensor& opacities,
    const torch::Tensor& sh0, const torch::Tensor& rest,
    const torch::Tensor& w, const torch::Tensor& cam, const torch::Tensor& bg,
    int active_sh, int width, int height, float fx, float fy, float cx, float cy, float near, float far,
    const torch::Tensor& image_shift, bool collect_metadata, bool preflight_only)
{
    return disk_forward_channels_wrapper(means, scales, rotations, opacities, sh0, rest, w, cam, bg,
        active_sh, width, height, fx, fy, cx, cy, near, far, image_shift, collect_metadata, preflight_only, 10);
}

std::vector<torch::Tensor> disk_forward_channels_wrapper(
    const torch::Tensor& means, const torch::Tensor& scales,
    const torch::Tensor& rotations, const torch::Tensor& opacities,
    const torch::Tensor& sh0, const torch::Tensor& rest,
    const torch::Tensor& w, const torch::Tensor& cam, const torch::Tensor& bg,
    int active_sh, int width, int height, float fx, float fy, float cx, float cy, float near, float far,
    const torch::Tensor& image_shift, bool collect_metadata, bool preflight_only, int channels)
{
    TORCH_CHECK(channels == 3 || channels == 8 || channels == 9 || channels == 10, "invalid disk output channel count");
    for (const auto& t : {means, scales, rotations, opacities, sh0, rest, w, cam, bg})
        check_tensor(t, means, "disk input");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3 && means.size(0) > 0
        && means.size(0) <= INT_MAX, "disk means must be nonempty Nx3");
    const int n = means.size(0);
    check_tensor(image_shift, means, "disk image_shift");
    TORCH_CHECK(image_shift.numel() == 0 || image_shift.sizes() == torch::IntArrayRef({n,2}),
        "disk image_shift must be empty or Nx2 pixel displacement");
    TORCH_CHECK(scales.sizes() == torch::IntArrayRef({n,2}), "disk scales must be Nx2");
    TORCH_CHECK(rotations.sizes() == torch::IntArrayRef({n,4}), "disk rotations must be Nx4");
    TORCH_CHECK(opacities.sizes() == torch::IntArrayRef({n,1}), "disk opacity must be Nx1");
    TORCH_CHECK(sh0.sizes() == torch::IntArrayRef({n,1,3}), "disk SH0 must be Nx1x3");
    TORCH_CHECK(rest.dim() == 3 && rest.size(0) == n && rest.size(2) == 3, "disk SHrest must be NxKx3");
    TORCH_CHECK((active_sh == 1 || active_sh == 4 || active_sh == 9 || active_sh == 16)
        && rest.size(1) >= active_sh-1, "invalid disk SH basis count");
    TORCH_CHECK(w.sizes() == torch::IntArrayRef({4,4}) && cam.numel() == 3 && bg.numel() == 3,
        "disk camera requires 4x4 w2c and three-element position/background");
    check_camera(width, height, fx, fy, cx, cy, near, far);
    const at::cuda::CUDAGuard guard(means.device());
    size_t free_bytes, total_bytes;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t reserve = std::max(size_t(2ull<<30), total_bytes/10);
    const size_t budget = free_bytes > reserve ? std::min(size_t(8ull<<30), free_bytes-reserve) : 0;
    const size_t n_tiles = size_t(div_round_up(width, config::tile_width))*div_round_up(height, config::tile_height);
    const size_t fixed = size_t(2048)*n+size_t(256)*width*height+size_t(4096)*n_tiles;
    TORCH_CHECK(fixed <= budget, "FasterGS2D resource guard: insufficient headroom before preflight");
    const auto limit = static_cast<unsigned long long>(std::min(size_t(1<<24), (budget-fixed)/256));
    static_assert(sizeof(Disk) == 13*sizeof(float));
    auto image = preflight_only ? torch::empty({0}, means.options()) : torch::empty({12,height,width}, means.options());
    auto geometry = torch::zeros({n,13}, means.options());
    auto counter = torch::zeros({1}, means.options().dtype(torch::kInt64));
    auto metadata = torch::zeros({collect_metadata ? 2 : 0, n}, means.options());
    auto primitive = torch::empty({0}, means.options().dtype(torch::kByte));
    auto tile = torch::empty_like(primitive), instance = torch::empty_like(primitive), bucket = torch::empty_like(primitive);
    DiskRasterization d{reinterpret_cast<Disk*>(geometry.data_ptr<float>()),
        preflight_only ? nullptr : image.data_ptr<float>()+3*width*height, fx, fy, cx, cy, near, far,
        image_shift.numel() ? reinterpret_cast<const float2*>(image_shift.data_ptr<float>()) : nullptr,
        collect_metadata ? metadata.data_ptr<float>() : nullptr, n,
        reinterpret_cast<unsigned long long*>(counter.data_ptr<int64_t>()), limit, preflight_only, channels};
    auto counts = forward(
        resize_function_wrapper(primitive), resize_function_wrapper(tile),
        resize_function_wrapper(instance), resize_function_wrapper(bucket),
        reinterpret_cast<const float3*>(means.data_ptr<float>()),
        reinterpret_cast<const float3*>(scales.data_ptr<float>()),
        reinterpret_cast<const float4*>(rotations.data_ptr<float>()), opacities.data_ptr<float>(),
        reinterpret_cast<const float3*>(sh0.data_ptr<float>()), reinterpret_cast<const float3*>(rest.data_ptr<float>()),
        reinterpret_cast<const float4*>(w.data_ptr<float>()), reinterpret_cast<const float3*>(cam.data_ptr<float>()),
        reinterpret_cast<const float3*>(bg.data_ptr<float>()), image.data_ptr<float>(), n, active_sh,
        rest.size(1), width, height, fx, fy, cx, cy, near, far, false, at::cuda::getCurrentCUDAStream(), &d);
    auto state = torch::tensor({std::get<0>(counts),std::get<1>(counts),std::get<2>(counts)}, torch::kInt32);
    if (preflight_only) return {state};  // Explicit count-only result, never an uninitialized image.
    return {image, geometry, primitive, tile, instance, bucket, metadata, state};
}

std::vector<torch::Tensor> disk_backward_wrapper(
    const torch::Tensor& grad_image, const torch::Tensor& image,
    const torch::Tensor& geometry, const torch::Tensor& means,
    const torch::Tensor& scales, const torch::Tensor& rotations,
    const torch::Tensor& rest, const torch::Tensor& primitive_blob,
    const torch::Tensor& tile_blob, const torch::Tensor& instance_blob,
    const torch::Tensor& image_shift,
    const torch::Tensor& w, const torch::Tensor& cam, const torch::Tensor& bg,
    int active_sh, int n_instances, int selector, float fx, float fy, float cx, float cy, float near, float far)
{
    const at::cuda::CUDAGuard guard(means.device());
    check_tensor(grad_image, means, "disk grad_image");
    const int n = means.size(0), height = image.size(1), width = image.size(2);
    TORCH_CHECK(grad_image.dim() == 3 && grad_image.size(1) == height && grad_image.size(2) == width,
        "disk grad_image must have shape CxHxW");
    const int channels = grad_image.size(0);
    TORCH_CHECK(channels == 3 || channels == 8 || channels == 9 || channels == 10, "invalid disk gradient channel count");
    const bool collect_sh = rest.size(1) == 15;
    auto output = [&](const torch::Tensor& like) {
        return collect_sh ? torch::empty_like(like) : torch::zeros_like(like);
    };
    auto gm = output(means), gs = output(scales), gq = output(rotations);
    auto go = collect_sh ? torch::empty({n,1}, means.options()) : torch::zeros({n,1}, means.options());
    auto g0 = torch::zeros({n,1,3}, means.options());
    auto gr = output(rest), gg = torch::zeros({n,16}, means.options());
    auto gshift = torch::zeros_like(image_shift);
    char* pb = reinterpret_cast<char*>(primitive_blob.data_ptr());
    char* tb = reinterpret_cast<char*>(tile_blob.data_ptr());
    char* ib = reinterpret_cast<char*>(instance_blob.data_ptr());
    auto p = PrimitiveBuffers::from_blob(pb, n);
    const int grid_width = div_round_up(width, config::tile_width);
    const int grid_height = div_round_up(height, config::tile_height);
    const int n_tiles = grid_width*grid_height, end_bit = extract_end_bit(n_tiles-1);
    auto tiles = TileBuffers::from_blob(tb, n_tiles);
    DiskRasterization d{reinterpret_cast<Disk*>(geometry.data_ptr<float>()), nullptr, fx, fy, cx, cy, near, far,
        image_shift.numel() ? reinterpret_cast<const float2*>(image_shift.data_ptr<float>()) : nullptr, nullptr, n};
    d.output_channels = channels;
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto blend = [&](const uint* indices) {
        // from_blob advanced ib past the sorted instance data.
        const size_t offset = ib - reinterpret_cast<char*>(instance_blob.data_ptr());
        const size_t mask_offset = (offset + alignof(uint) - 1) & ~(size_t(alignof(uint)) - 1);
        const size_t words = size_t(n_instances / 32 + n_tiles) * config::block_size_blend;
        TORCH_CHECK(mask_offset + words * sizeof(uint) <= size_t(instance_blob.numel()), "missing disk forward hit masks");
        const uint* hit_masks = reinterpret_cast<const uint*>(instance_blob.data_ptr<uint8_t>() + mask_offset);
        disk_blend_backward<<<dim3(grid_width,grid_height), dim3(config::tile_width,config::tile_height),0,stream>>>(
            p, tiles, indices, hit_masks, d.geometry, reinterpret_cast<const float3*>(bg.data_ptr<float>()),
            image.data_ptr<float>(), grad_image.data_ptr<float>(), reinterpret_cast<PackedDiskGradient*>(gg.data_ptr<float>()),
            image_shift.numel() ? reinterpret_cast<float2*>(gshift.data_ptr<float>()) : nullptr, d, width, height);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (end_bit <= 16) {
        auto instances = InstanceBuffers<ushort>::from_blob(ib, n_instances, end_bit);
        instances.primitive_indices.selector = selector;
        blend(instances.primitive_indices.Current());
    } else {
        auto instances = InstanceBuffers<uint>::from_blob(ib, n_instances, end_bit);
        instances.primitive_indices.selector = selector;
        blend(instances.primitive_indices.Current());
    }
    auto convert = [&](auto mode) {
        constexpr int threads = 128;
        constexpr int shared_bytes = decltype(mode)::value ? threads*15*sizeof(float3) : 0;
        disk_preprocess_backward<decltype(mode)::value><<<div_round_up(n,threads),threads,shared_bytes,stream>>>(
            reinterpret_cast<const float3*>(means.data_ptr<float>()), reinterpret_cast<const float2*>(scales.data_ptr<float>()),
            reinterpret_cast<const float4*>(rotations.data_ptr<float>()), reinterpret_cast<const float3*>(rest.data_ptr<float>()),
            reinterpret_cast<const float4*>(w.data_ptr<float>()), reinterpret_cast<const float3*>(cam.data_ptr<float>()),
            p.n_touched_tiles, d.geometry, reinterpret_cast<const PackedDiskGradient*>(gg.data_ptr<float>()),
            reinterpret_cast<float3*>(gm.data_ptr<float>()), reinterpret_cast<float2*>(gs.data_ptr<float>()),
            reinterpret_cast<float4*>(gq.data_ptr<float>()), go.data_ptr<float>(),
            reinterpret_cast<float3*>(g0.data_ptr<float>()), reinterpret_cast<float3*>(gr.data_ptr<float>()),
            n, active_sh, rest.size(1));
    };
    if (collect_sh) convert(std::true_type{});
    else convert(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gm,gs,gq,go,g0,gr,gshift};
}
namespace {
// Match the FP32 matrix product's accumulation order and its separate translation.
// Near-coincident points can lose the surface normal if these operations are reassociated.
__device__ float3 unproject_pixel(const float* depth, const float* rx, const float* ry,
    const float* rotation, const float* position, int x, int y, int w) {
    const float z = depth[y * w + x];
    const float px = rx[x] * z, py = ry[y] * z;
    float xyz[3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float* row = rotation + 3 * i;
        const float xy = __fmaf_rn(py, row[1], __fmul_rn(px, row[0]));
        xyz[i] = __fadd_rn(__fmaf_rn(z, row[2], xy), position[i]);
    }
    return make_float3(xyz[0], xyz[1], xyz[2]);
}

__global__ void depth_difference_forward(const float* depth, const float* rx, const float* ry,
    const float* rotation, const float* position, float3* vertical, float3* horizontal, int h, int w) {
    const int x = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w - 1 || y >= h - 1) return;
    const int i = (y - 1) * (w - 2) + x - 1;
    vertical[i] = unproject_pixel(depth, rx, ry, rotation, position, x, y + 1, w)
                - unproject_pixel(depth, rx, ry, rotation, position, x, y - 1, w);
    horizontal[i] = unproject_pixel(depth, rx, ry, rotation, position, x + 1, y, w)
                  - unproject_pixel(depth, rx, ry, rotation, position, x - 1, y, w);
}

// Each depth contributes to at most four interior difference vectors. Gather
// their adjoints at that pixel instead of scattering into four full images.
__global__ void depth_difference_backward(const float3* vertical, const float3* horizontal,
    const float* rx, const float* ry, const float* rotation, float* depth, int h, int w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int stride = w - 2;
    float3 dv = make_float3(0), dh = make_float3(0);
    if (x > 0 && x < w - 1) {
        if (y >= 2) dv = vertical[(y - 2) * stride + x - 1];
        if (y < h - 2) dv -= vertical[y * stride + x - 1];
    }
    if (y > 0 && y < h - 1) {
        if (x >= 2) dh = horizontal[(y - 1) * stride + x - 2];
        if (x < w - 2) dh -= horizontal[(y - 1) * stride + x];
    }
    const float3 g = dv + dh;
    const float dx = g.x * rotation[0] + g.y * rotation[3] + g.z * rotation[6];
    const float dy = g.x * rotation[1] + g.y * rotation[4] + g.z * rotation[7];
    const float dz = g.x * rotation[2] + g.y * rotation[5] + g.z * rotation[8];
    depth[y * w + x] = dz + dy * ry[y] + dx * rx[x];
}
}

std::vector<torch::Tensor> depth_difference_forward_wrapper(const torch::Tensor& depth,
    const torch::Tensor& rx, const torch::Tensor& ry, const torch::Tensor& rotation, const torch::Tensor& position) {
    for (const auto& t : {depth, rx, ry, rotation, position}) check_tensor(t, depth, "depth difference input");
    const auto h = ry.numel(), w = rx.numel();
    TORCH_CHECK(rx.dim() == 1 && ry.dim() == 1 && w >= 3 && h >= 3 && w <= 65520 && h <= 65520
        && w*h <= INT_MAX && depth.sizes() == torch::IntArrayRef({h,w})
        && rotation.sizes() == torch::IntArrayRef({3,3}) && position.sizes() == torch::IntArrayRef({3}),
        "invalid depth difference tensors");
    const at::cuda::CUDAGuard guard(depth.device());
    auto vertical = torch::empty({h-2,w-2,3}, depth.options()), horizontal = torch::empty_like(vertical);
    depth_difference_forward<<<dim3(div_round_up(int(w)-2,32),div_round_up(int(h)-2,8)), dim3(32,8), 0,
        at::cuda::getCurrentCUDAStream()>>>(depth.data_ptr<float>(), rx.data_ptr<float>(), ry.data_ptr<float>(),
        rotation.data_ptr<float>(), position.data_ptr<float>(), reinterpret_cast<float3*>(vertical.data_ptr<float>()),
        reinterpret_cast<float3*>(horizontal.data_ptr<float>()), h, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {vertical,horizontal};
}

torch::Tensor depth_difference_backward_wrapper(const torch::Tensor& vertical, const torch::Tensor& horizontal,
    const torch::Tensor& rx, const torch::Tensor& ry, const torch::Tensor& rotation) {
    for (const auto& t : {vertical, horizontal, rx, ry, rotation}) check_tensor(t, vertical, "depth difference gradient");
    TORCH_CHECK(rx.dim() == 1 && ry.dim() == 1 && rx.numel() >= 3 && ry.numel() >= 3
        && rx.numel() <= 65520 && ry.numel() <= 65520 && rx.numel()*ry.numel() <= INT_MAX,
        "invalid depth difference dimensions");
    const int h = ry.numel(), w = rx.numel();
    TORCH_CHECK(vertical.sizes() == torch::IntArrayRef({h-2,w-2,3}) && horizontal.sizes() == vertical.sizes()
        && rotation.sizes() == torch::IntArrayRef({3,3}), "invalid depth difference tensors");
    const at::cuda::CUDAGuard guard(vertical.device());
    auto depth = torch::empty({h,w}, vertical.options());
    depth_difference_backward<<<dim3(div_round_up(w,32),div_round_up(h,8)), dim3(32,8), 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(vertical.data_ptr<float>()), reinterpret_cast<const float3*>(horizontal.data_ptr<float>()),
        rx.data_ptr<float>(), ry.data_ptr<float>(), rotation.data_ptr<float>(), depth.data_ptr<float>(), h, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return depth;
}

}
