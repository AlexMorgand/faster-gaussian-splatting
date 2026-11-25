# Experimental perspective disks

The disk entry point, `diff_rasterize_disks`, renders primitives with two
log-scales and a quaternion defining the tangent plane.

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

On a Garden checkpoint at 27k iterations, the last preprocessing and blending
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
