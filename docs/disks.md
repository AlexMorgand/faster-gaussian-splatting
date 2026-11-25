# Experimental perspective disks

The disk entry point, `diff_rasterize_disks`, renders primitives with two
log-scales and a quaternion defining the tangent plane.

## From the paper to FasterGS

| Part of 2DGS | Implementation in FasterGS |
| --- | --- |
| Oriented disk and perspective splatting (§4) | `Gaussians2D` stores two tangent scales and inherits the FasterGS rotation, opacity and appearance parameters. The CUDA renderer solves for the pixel ray's intersection in disk coordinates and applies the screen-space lowpass. FasterGS supplies depth sorting and tile binning. |
| Depth distortion (§5, Appendix A) | The blend kernel accumulates the squared mapped-depth form of Eq. 17 using weighted moments. The Garden recipe sets its loss weight to zero. |
| Normal consistency (§5, Eqs. 14–15) | `disks/Renderer.py` unprojects depth and takes finite differences to obtain a surface normal. The default loss compares it with the normalized sum of weighted disk normals and scales the dot product by detached accumulated opacity. Set `TRAINING.LOSS.NORMAL_LOSS_MODE: per_disk` to use Eq. 14. It computes accumulated opacity minus the dot product of the unnormalized weighted disk normals with the unit surface normal, preserving gradients through the blending weights. These objectives can produce different gradients where disks disagree. Garden uses expected depth and enables this term after iteration 7000. |
| Adaptive density control (§6.1) | The paper projects the 3D center gradient onto the screen. Our renderer differentiates a 2D pixel shift, converts its gradient to normalized device coordinates and averages over views with projected support. The calibrated score feeds FasterGS's shared clone, split and prune operations. |

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
world geometry, normal and SH direction fixed. Its gradient measures sensitivity to a screen-space displacement. Optional 2×N metadata
contains conservative clipped support radius and actual accepted-hit visibility.

`preflight_disks` performs preprocessing/counting only. Actual uint64 tile counts
are checked before signed casts, sorting and instance/bucket allocation.
Tiles are 16×12 pixels. The guard caps tile instances at 2²⁴ and estimated
incremental allocations at 8 GiB. It reserves the greater of 2 GiB and 10%
of device memory. A disk crossing the near plane can cover the full viewport.

## How the training path was optimized

The disk renderer shares FasterGS's depth sort, tile binning and instance
ranges. The disk kernels specialize the parts that depend on a ray intersecting
an oriented plane. Three changes account for how the training path works today.

1. **Select outputs from the active losses.** RGB-only training needs three
   channels. Expected-depth normal supervision needs eight, median-depth normals
   need nine, and distortion needs all ten. Forward and backward specialize on
   that choice while the Python result keeps its ten-plane layout.
2. **Reuse the forward visibility decisions in backward.** Each pixel records
   a 32-bit mask per group of 32 candidate disks. Backward skips rejected hits
   and replays accepted intersections. This preserves the forward decisions
   without saving full per-hit geometry or changing the optimizer.
3. **Reduce repeated work in forward.** Neighboring pixels share batches of
   64 disk records. The training blend kernel processes hit-mask groups with
   four-way loop unrolling and specializes away growth bookkeeping after
   densification ends. Preprocessing skips the support test for disks whose
   tile range is already empty.

Each warp covers an 8×4 subtile and tests the projected disk bounds, including
the screen-space low-pass footprint and two pixels of padding. A warp ballot
lets all 32 pixels skip a disk that misses their subtile. Surviving disks
retain their sorted order, and backward reads the same per-pixel hit masks.

On nine training views per scene (RTX 5090), subtile rejection reduced mean
forward blend time from 1.494 to 1.393 ms on Garden, 2.077 to 1.916 ms on
Bicycle and 0.518 to 0.507 ms on Bonsai. These measurements use the 27k
Garden checkpoint and the final 30k Bicycle and Bonsai checkpoints.

On a Garden checkpoint at 27k iterations, the earlier preprocessing and blending
changes reduced GPU kernel time as follows (milliseconds per profiled view).

| Forward kernel | Previous disks | Revised disks | Stock 3D |
| --- | ---: | ---: | ---: |
| Blending | 1.879 | 1.506 | 0.459 |
| Preprocessing | 1.018 | 0.874 | 0.368 |

The paired 30k Garden training runs on one RTX 5090 took 362.5 and 351.8
seconds after image prefetch (10.7 seconds saved, 2.9%). The revised run
finished with 2.96 million disks and 27.000 dB PSNR on 24 held-out views.
A stock run took 254.9 seconds with 4.15 million 3D Gaussians. The single
checkpoint timings describe that training stage only. Wall time also includes
backward, optimization, topology changes and checkpointing.

The disk trainer also avoids per-step gathers of visible disks. It accumulates
scores and maximum radii with a visibility mask and still counts occluded
disks with projected support in the denominator. At Garden iteration 12k
(3.02 million disks), this update averaged 0.473 ms versus 0.812 ms across
54 paired measurements on nine views. All three statistics matched exactly
on those views. RGB-only training also reuses the zero auxiliary planes, and
unit scale and unmixed depth use their source tensors directly.

With these changes, a separate pair of 30k Garden runs took 333.0 and 327.7
seconds after prefetch, with checkpoint writes disabled. Mean PSNR across
24 held-out views was 26.990 and 26.999 dB respectively. This comparison
includes backward, density control and Adam. Each timing is one run.

For fixed FP32 training cameras, a CUDA kernel unprojects the four neighbors
of each interior pixel and writes the two world-space differences directly.
The renderer reuses the pose uploaded for rasterization. The normal loss and
the depth-difference adjoint are shared with the original path. On nine Garden
views at 27k, normalized supervision increased a full step from 11.80 to
13.41 ms. Fusing the differences and reusing the pose reduced it to 12.90 ms.
These timings use a fixed checkpoint on an RTX 5090 with distortion disabled.

## Build and checks

Use the normal documented NeRFICG/FasterGS prerequisites and matching CUDA
toolchain. No additional dependency, GLM copy, research renderer or prebuilt
disk library is required. For a no-install build, from `FasterGSCudaBackend`:

```sh
MAX_JOBS=1 python setup.py build --build-base /absolute/build \
    --build-lib /absolute/native
```

Use `PYTHONPATH=/absolute/native:$NERFICG/src`, run from NeRFICG root:

```sh
python -m unittest discover -s src/Methods/FasterGS/tests
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

## Source provenance and limits

Disk intersection mathematics and adjoints were independently implemented on
FasterGS. The implementation reuses the project's Apache-2.0 sorting, buffers,
quaternion and spherical-harmonic helpers. The disk path has no dependency on
the Inria CUDA rasterizer.
The `disks/` package supplies the training schedule and converts camera-space
depth to world-space normals. NeRF-ICG exposes this package as `FasterGS2D`.
