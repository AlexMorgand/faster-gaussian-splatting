# Perspective disk rasterization

The CUDA backend shares FasterGS’s depth sort and tile binning. Each pixel ray
intersects an oriented disk with two tangent scales. The intersection, blend
and their gradients are implemented independently in FasterGS.

## From the paper to FasterGS

| Part of 2DGS | Implementation in FasterGS |
| --- | --- |
| Perspective splatting (§4) | `Gaussians2D` stores two tangent scales. CUDA intersects each pixel ray with the oriented plane and applies the screen-space low-pass filter. Rotation, opacity, appearance, sorting and tile binning come from FasterGS. |
| Depth distortion (§5, Appendix A) | The blend kernel evaluates Eq. 17 using weighted moments of mapped depth. Garden sets the distortion weight to zero. |
| Normal consistency (§5, Eqs. 14–15) | Finite differences of unprojected depth give the surface normal. The default loss compares this with the normalized weighted disk normal and scales the dot product by detached opacity. With `TRAINING.LOSS.NORMAL_LOSS_MODE: per_disk`, Eq. 14 uses accumulated opacity minus the dot product of the weighted disk normals with the unit surface normal. |
| Adaptive density control (§6.1) | The renderer differentiates a pixel displacement, converts the gradient to normalized device coordinates and averages over views with projected support. This score feeds FasterGS's clone, split and prune operations. Split offsets are sampled along the two tangent axes. |

The stock FasterGS trainer owns sampling, photometric loss evaluation, backward
and Adam. The disk trainer selects the required render outputs and updates disk
density statistics between backward and Adam. The hit masks and channel
specializations below implement those choices in CUDA.

Equation and section numbers refer to the [2DGS paper](https://arxiv.org/html/2403.17888).

## Math and API

For camera-space centre C, scaled tangents U,V and pinhole ray r with r.z=1,
solve `C+uU+vV=t*r`. The disk Gaussian is `exp(-(u²+v²)/2)`. Screen lowpass
is the maximum with `exp(-||pixel-project(C)||²)`, with disk-winning exact ties.
All hits retain ray/plane intersection depth, including screen winners.
The intersection test rejects clipped depths and nearly singular systems using
a relative determinant threshold.

The ten CHW output planes are RGB, weighted camera-depth, alpha, weighted
world-normal XYZ, median camera-depth and distortion. Median is the maximum
eligible intersection depth while pre-hit transmittance exceeds 0.5. Exact
maximum ties select the first centre-depth/row-ID-sorted hit. Distortion is
`sum(i<j) wi*wj*(mi-mj)²` in mapped inverse-depth coordinates. Empty auxiliaries
are zero. Background contributes only to RGB.

RGB and every auxiliary have independent adjoints. Implicit differentiation
solves `[U V -r]^T lambda=[g_u,g_v,g_t]`, yielding `g_C=-lambda`,
`g_U=-u*lambda`, `g_V=-v*lambda`, plus screen-centre/normal/SH derivatives.
Hard visibility, order, cap and median choices are piecewise constant.
Second derivatives are not supported.

Optional N×2 `image_shift` evaluates a primitive at pixel-delta, holding its
world geometry, normal and SH direction fixed. Its gradient measures sensitivity
to a screen-space displacement. Optional 2×N metadata
contains conservative clipped support radius and actual accepted-hit visibility.

`preflight_disks` performs preprocessing/counting only. Actual uint64 tile counts
are checked before signed casts, sorting and instance/bucket allocation.
Tiles are 16×12 pixels. The guard caps tile instances at 2²⁴ and estimated
incremental allocations at 8 GiB. It reserves the greater of 2 GiB and 10%
of device memory. A disk crossing the near plane can cover the full viewport.

## Training performance

| Change | Effect on training |
| --- | --- |
| Select outputs from active losses | Both passes use 3 channels for RGB, 8 for expected-depth normals, 9 for median-depth normals and 10 for distortion. The public output layout stays stable. |
| Bound the blend work | Each warp rejects disks outside its 8×4 subtile. Pixels reuse 32-bit forward hit masks in backward and skip rejected intersections. |
| Accumulate growth statistics in place | Visibility masks replace per-step gathers. Projected support still counts toward the denominator for occluded disks. |
| Supervise normals | For fixed FP32 cameras, CUDA unprojects depth neighbors directly into world-space differences. The normalization backward kernel reads the incoming surface gradient in its native planar layout. |

On an RTX 5090, subtile rejection reduced mean forward blend time from 1.494
to 1.393 ms on Garden, 2.077 to 1.916 ms on Bicycle and 0.518 to 0.507 ms on
Bonsai (nine training views per checkpoint). On the 27k Garden checkpoint,
fusing fixed-camera differences reduced a normal-supervised training step
from 13.41 to 12.90 ms. Disk count changes during densification also affect
end-to-end training time.

In two 30,000-step Garden runs with the optimized normal backward pass, training
took 327 and 342 seconds on the same RTX 5090 after image prefetch, with
checkpoint writes disabled. Mean PSNR was 26.983 and 26.995 dB on 24 held-out views.
The matching runs before the backward change took 335 and 341 seconds.

An earlier run of this Garden recipe also measured SSIM and VGG-LPIPS.
It scored 27.000 dB PSNR, 0.8554 SSIM and 0.1244 VGG-LPIPS on the same 24 views.

## Build and numerical checks

Use the normal documented NeRFICG/FasterGS prerequisites and matching CUDA
toolchain. No additional dependency, GLM copy, research renderer or prebuilt
disk library is required. For a no-install build, from `FasterGSCudaBackend`:

```sh
MAX_JOBS=1 python setup.py build --build-base /absolute/build \
    --build-lib /absolute/native
```

Use `PYTHONPATH=/absolute/native:$NERFICG/src`, run from NeRFICG root:

```sh
python src/Methods/FasterGS/tests/check_disks.py --output /absolute/cpu.json
python src/Methods/FasterGS/tests/check_disks.py --gpu --output /absolute/native.json
python src/Methods/FasterGS/tests/check_disk_shift.py --gpu --output /absolute/shift.json
```

An optional `check_disks.py --stock-reference /absolute/official/_C...so`
compares actual 3D RGB/gradients against a separately built official-main
library; without it, the 3D control is same-build repeatability, not independent
baseline equivalence. Tests otherwise need no external binary or private import.
The double oracle uses linear solves and explicit pairwise distortion; native
code uses Cramer's rule and weighted moments. Tests cover scalar finite
differences, individual auxiliaries, SH separation, median crossing/ties,
lowpass, saturation, clipping, streams and input validation.

The training recipe in `disks/` is adapted from MIT-licensed NeRF-ICG
revision 57ddd49. See [its license](../disks/LICENSE). The CUDA disk path
uses FasterGS sorting, buffers, quaternion and spherical-harmonic helpers.
