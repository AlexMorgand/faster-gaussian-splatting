"""Independent double-precision ray/plane oracle and staged native controls.

No research rasterizer imports. The oracle solves a linear system and computes
distortion explicitly over pairs; CUDA uses Cramer's rule and moment sums.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

from FasterGSCudaBackend import _C
from FasterGSCudaBackend.torch_bindings.disks import diff_rasterize_disks
from FasterGSCudaBackend.torch_bindings.rasterization import RasterizerSettings, diff_rasterize


def rotate(q, axes):
    q = q / q.norm()
    xyz = q[1:].expand_as(axes)
    uv = torch.linalg.cross(xyz, axes)
    return axes + 2 * (q[0] * uv + torch.linalg.cross(xyz, uv))


def oracle(params, s, image_shift=None):
    means, scales, quats, opacity, sh0, rest = params
    w, camera, bg = (x.to(means) for x in (s.w2c, s.cam_position, s.bg_color))
    yy, xx = torch.meshgrid(torch.arange(s.height, dtype=means.dtype),
                            torch.arange(s.width, dtype=means.dtype), indexing="ij")
    rays = torch.stack(((xx + .5 - s.center_x) / s.focal_x,
                        (yy + .5 - s.center_y) / s.focal_y, torch.ones_like(xx)), -1)
    shape = (s.height, s.width)
    T = torch.ones(shape, dtype=means.dtype)
    rgb = torch.zeros((*shape, 3), dtype=means.dtype)
    depth, alpha, median = (torch.zeros_like(T) for _ in range(3))
    normal = torch.zeros_like(rgb)
    weights, mapped = [], []
    centers = means @ w[:3, :3].T + w[:3, 3]
    for i in centers[:, 2].argsort(stable=True).tolist():
        center = centers[i]
        if not s.near_plane <= center[2] <= s.far_plane or quats[i].square().sum() < 1e-8:
            continue
        axes = rotate(quats[i], torch.eye(3, dtype=means.dtype))
        u, v = (axes[:2] * scales[i].exp()[:, None]) @ w[:3, :3].T
        shift = means.new_zeros(2) if image_shift is None else image_shift[i]
        shifted_rays = rays-torch.stack((shift[0]/s.focal_x, shift[1]/s.focal_y, shift[0]*0))
        n = axes[2]
        n = n * torch.where((n @ w[:3, :3].T * center).sum() > 0, -1., 1.)
        A = torch.stack((u.expand_as(rays), v.expand_as(rays), -shifted_rays), -1)
        determinant = torch.linalg.det(A)
        determinant_scale = u.norm()*v.norm()
        good = ((determinant.abs()/determinant_scale >= 1e-10) & (determinant_scale > 0)
                & torch.isfinite(determinant_scale) & torch.isfinite(determinant))
        safe_A = torch.where(good[..., None, None], A, torch.eye(3, dtype=A.dtype))
        uvz = torch.linalg.solve(safe_A, -center.expand_as(rays))
        z = uvz[..., 2]
        disk = torch.exp(-.5 * uvz[..., :2].square().sum(-1))
        screen = torch.exp(-((xx+.5-s.center_x-s.focal_x*center[0]/center[2]-shift[0]).square()
                             +(yy+.5-s.center_y-s.focal_y*center[1]/center[2]-shift[1]).square()))
        gaussian = torch.where(screen > disk, screen, disk)
        a = (opacity[i].sigmoid() * gaussian).clamp(max=.99)
        hit = good & (z >= s.near_plane) & (z <= s.far_plane) & (a >= 1/255) & (T >= 1e-4)
        a = torch.where(hit, a, 0.)
        weight = T * a
        # Real spherical harmonics, independently written scalar polynomial basis.
        direction = means[i] - camera
        x, y, zdir = direction / direction.norm()
        basis = [
            x.new_tensor(1 / math.sqrt(4 * math.pi)),
            -math.sqrt(3/(4*math.pi))*y, math.sqrt(3/(4*math.pi))*zdir,
            -math.sqrt(3/(4*math.pi))*x,
            math.sqrt(15/(4*math.pi))*x*y, -math.sqrt(15/(4*math.pi))*y*zdir,
            math.sqrt(5/(16*math.pi))*(3*zdir*zdir-1),
            -math.sqrt(15/(4*math.pi))*x*zdir, math.sqrt(15/(16*math.pi))*(x*x-y*y),
            -math.sqrt(35/(32*math.pi))*y*(3*x*x-y*y),
            math.sqrt(105/(4*math.pi))*x*y*zdir,
            -math.sqrt(21/(32*math.pi))*y*(5*zdir*zdir-1),
            math.sqrt(7/(16*math.pi))*zdir*(5*zdir*zdir-3),
            -math.sqrt(21/(32*math.pi))*x*(5*zdir*zdir-1),
            math.sqrt(105/(16*math.pi))*zdir*(x*x-y*y),
            -math.sqrt(35/(32*math.pi))*x*(x*x-3*y*y),
        ][:s.active_sh_bases]
        color = .5 + basis[0] * sh0[i, 0]
        for k in range(1, s.active_sh_bases):
            color = color + basis[k] * rest[i, k-1]
        rgb = rgb + weight[..., None] * color.clamp(min=0)
        depth = depth + weight * z
        normal = normal + weight[..., None] * n
        # Appendix B Eq.19: maximum eligible hit depth, not last centre-sorted hit.
        median = torch.where(hit & (T > .5) & (z > median), z, median)
        alpha = alpha + weight
        weights.append(weight)
        mapped.append(s.far_plane/(s.far_plane-s.near_plane)*(1-s.near_plane/z))
        T = T * (1-a)
    distortion = torch.zeros_like(T)
    for i in range(len(weights)):
        for j in range(i):
            distortion = distortion + weights[i]*weights[j]*(mapped[i]-mapped[j]).square()
    rgb = rgb + T[..., None]*bg
    return torch.cat((rgb.permute(2, 0, 1), depth[None], alpha[None],
                      normal.permute(2, 0, 1), median[None], distortion[None]))


def fixture():
    torch.manual_seed(2140)
    # Rigid nonidentity camera, off-axis oblique overlapping disks.
    angle = .17
    w = torch.tensor([[math.cos(angle), 0, math.sin(angle), -.12],
                      [0, 1, 0, .08], [-math.sin(angle), 0, math.cos(angle), .15],
                      [0, 0, 0, 1]], dtype=torch.float64)
    cam = -w[:3, :3].T @ w[:3, 3]
    s = RasterizerSettings(w, cam, torch.tensor([.03, .07, .11], dtype=torch.float64),
                          16, 13, 11, 14., 12., 4.9, 6.2, .2, 20., False)
    p = [
        torch.tensor([[.18, -.17, 2.10], [.30, -.04, 2.60]], dtype=torch.float64),
        torch.tensor([[.58, .43], [.52, .62]], dtype=torch.float64).log(),
        torch.tensor([[.92, .17, .29, -.13], [.83, -.25, .24, .18]], dtype=torch.float64),
        torch.tensor([[.37], [-.23]], dtype=torch.float64),
        torch.randn(2, 1, 3, dtype=torch.float64) * .12,
        torch.randn(2, 15, 3, dtype=torch.float64) * .05,
    ]
    return p, s


def settings_to(s, device):
    return s._replace(w2c=s.w2c.to(device=device, dtype=torch.float32),
                      cam_position=s.cam_position.to(device=device, dtype=torch.float32),
                      bg_color=s.bg_color.to(device=device, dtype=torch.float32))


def objective_weights(s):
    g = torch.Generator().manual_seed(719)
    weights = torch.rand((10, s.height, s.width), dtype=torch.float64, generator=g) + .3
    weights /= s.width*s.height
    # Keep each auxiliary's native scale visible, especially inverse-depth distortion.
    weights[9] *= 1000
    masks = {
        "rgb": [0, 1, 2], "depth": [3], "alpha": [4], "normal": [5, 6, 7],
        "median": [8], "distortion": [9], "aux": list(range(3, 10)), "mixed": list(range(10)),
    }
    result = {}
    for name, channels in masks.items():
        v = torch.zeros_like(weights)
        v[channels] = weights[channels]
        result[name] = v
    return result


def differentiable(params):
    return [p.clone().detach().requires_grad_() for p in params]


def crossing_fixture(tie=False):
    s = RasterizerSettings(torch.eye(4, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
                          torch.zeros(3, dtype=torch.float64), 1, 1, 1, 1., 1., 0., .5, .2, 20., False)
    # Ray(.5,0,1): centres2<3, but the tilted first plane intersects at4>3.
    p = [
        torch.tensor([[0., 0., 2.], [0., 0., 3.]], dtype=torch.float64),
        torch.full((2, 2), math.log(4.), dtype=torch.float64),
        torch.tensor([[math.cos(math.pi/8), 0., -math.sin(math.pi/8), 0.],
                      [1., 0., 0., 0.]], dtype=torch.float64),
        torch.full((2, 1), -2., dtype=torch.float64),
        torch.zeros((2, 1, 3), dtype=torch.float64), torch.zeros((2, 0, 3), dtype=torch.float64),
    ]
    if tie:
        # Exactly representable rotations: both intersections at1, centres1<2.
        p[0] = torch.tensor([[0., 0., 1.], [.5, 0., 2.]], dtype=torch.float64)
        p[2] = torch.tensor([[1., 0., 0., 0.], [.5, .5, .5, .5]], dtype=torch.float64)
    return p, s


def grads(output, params, weight):
    result = torch.autograd.grad((output * weight).sum(), params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, result)]


def cpu_gate():
    p, s = fixture()
    result = {}
    for name in ("rgb", "aux", "mixed"):
        q = differentiable(p)
        weight = objective_weights(s)[name]
        analytic = grads(oracle(q, s), q, weight)
        worst = 0.
        for family, tensor in enumerate(p):
            for index in range(tensor.numel()):
                eps = 1e-5
                plus, minus = [t.clone() for t in p], [t.clone() for t in p]
                plus[family].view(-1)[index] += eps
                minus[family].view(-1)[index] -= eps
                fd = ((oracle(plus, s)-oracle(minus, s))*weight).sum()/(2*eps)
                value = analytic[family].view(-1)[index]
                torch.testing.assert_close(value, fd, atol=2e-7, rtol=2e-5)
                worst = max(worst, float((value-fd).abs()))
        if name == "aux":
            assert torch.count_nonzero(analytic[4]) == torch.count_nonzero(analytic[5]) == 0
        result[name] = {"all_scalar_fd_checks": sum(t.numel() for t in p), "max_abs_error": worst}
    p, s = crossing_fixture()
    q = differentiable(p)
    weight = torch.zeros((10, 1, 1), dtype=torch.float64)
    weight[8] = 1
    output = oracle(q, s)
    torch.testing.assert_close(output[8, 0, 0], torch.tensor(4., dtype=torch.float64), atol=1e-12, rtol=0)
    analytic = grads(output, q, weight)
    torch.testing.assert_close(analytic[0], torch.tensor([[-2., 0., 2.], [0., 0., 0.]], dtype=torch.float64),
                               atol=1e-12, rtol=0)
    worst = 0.
    for family, tensor in enumerate(p):
        for index in range(tensor.numel()):
            plus, minus = [t.clone() for t in p], [t.clone() for t in p]
            plus[family].view(-1)[index] += 1e-5
            minus[family].view(-1)[index] -= 1e-5
            fd = ((oracle(plus, s)-oracle(minus, s))*weight).sum()/2e-5
            expected = analytic[family].view(-1)[index]
            torch.testing.assert_close(fd, expected, atol=2e-7, rtol=2e-5)
            worst = max(worst, (fd-expected).abs().item())
    result["crossing_median"] = {"checks": sum(t.numel() for t in p), "max_abs_error": worst}
    result["lowpass"] = lowpass_controls(native=False)
    return result


def median_crossing_control():
    result = {}
    for tie in (False, True):
        p, s = crossing_fixture(tie)
        ray = torch.tensor([.5, 0., 1.], dtype=torch.float64)
        depths, alphas = [], []
        for i in range(2):
            axes = rotate(p[2][i], torch.eye(3, dtype=torch.float64))
            u, v = axes[:2]*p[1][i].exp()[:, None]
            uvz = torch.linalg.solve(torch.stack((u, v, -ray), -1), -p[0][i])
            a = (p[3][i].sigmoid()*torch.exp(-.5*uvz[:2].square().sum())).clamp(max=.99)
            depths.append(uvz[2].item())
            alphas.append(a.item())
        assert p[0][0, 2] < p[0][1, 2]
        assert all(s.near_plane < z < s.far_plane for z in depths)
        assert all(1/255 < a < .5 for a in alphas) and 1-alphas[0] > .5
        assert depths[0] == depths[1] if tie else depths[0] > depths[1]
        qq, refs = differentiable([t.cuda().float() for t in p]), differentiable(p)
        native, reference = diff_rasterize_disks(*qq, settings_to(s, "cuda")), oracle(refs, s)
        value = native[8, 0, 0].item()
        print("MEDIAN_CROSSING" if not tie else "MEDIAN_TIE",
              {"centres": p[0][:, 2].tolist(), "depths": depths, "alpha": alphas,
               "pre_hit_T": [1., 1-alphas[0]], "native_median": value}, flush=True)
        torch.testing.assert_close(native.cpu().double(), reference.detach(), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(native[8, 0, 0], native.new_tensor(1. if tie else 4.), atol=2e-5, rtol=0)
        weight = torch.zeros_like(native)
        weight[8] = 1
        actual, expected = grads(native, qq, weight), grads(reference, refs, weight.cpu().double())
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.cpu().double(), b, atol=1e-5, rtol=8e-4)
            assert torch.count_nonzero(a[1]) == 0, "median adjoint reached nonwinning plane"
        assert actual[0][0].abs().max() > .9 and actual[2][0].abs().max() > .9
        worst = 0.
        if not tie:  # A tie has no unique derivative; test deterministic first-winner selection only.
            for family, tensor in enumerate(p):
                for index in range(tensor.numel()):
                    plus, minus = [t.cuda().float() for t in p], [t.cuda().float() for t in p]
                    plus[family].view(-1)[index] += .001
                    minus[family].view(-1)[index] -= .001
                    fd = (diff_rasterize_disks(*plus, settings_to(s, "cuda"))[8, 0, 0]
                          -diff_rasterize_disks(*minus, settings_to(s, "cuda"))[8, 0, 0])/.002
                    expected_fd = actual[family].view(-1)[index]
                    torch.testing.assert_close(fd, expected_fd, atol=7e-4, rtol=2e-2)
                    worst = max(worst, (fd-expected_fd).abs().item())
        result["first_max_tie" if tie else "crossing"] = {
            "centres": p[0][:, 2].tolist(), "depths": depths, "pre_hit_T": [1., 1-alphas[0]],
            "native_median": value, "mean_gradient": actual[0].cpu().tolist(),
            "nonwinner_all_gradients_exact_zero": True,
            "native_fd_checks": 0 if tie else sum(t.numel() for t in p), "fd_max_abs_error": worst,
        }
    return result


def lowpass_controls(native):
    _, original_camera = fixture()
    camera = RasterizerSettings(torch.eye(4, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
                               torch.zeros(3, dtype=torch.float64),
                               16, 9, 9, 10., 10., 4.5, 4.5, .2, 20., False)
    tilt = [math.cos(math.pi/8), 0., -math.sin(math.pi/8), 0.]
    cases = (
        ("disk", [.025, -.015, 2.], [.7, .5], tilt, camera, (5, 4)),
        ("screen", [.025, -.015, 2.], [.02, .02], tilt, camera, (5, 4)),
        ("screen_cap", [.002, 0., 2.], [1e-6, 1e-6], tilt, camera, (4, 4)),
        ("rotated_screen", [.025, -.015, 2.], [.02, .02], tilt,
         camera._replace(w2c=original_camera.w2c, cam_position=original_camera.cam_position), (5, 4)),
        ("tiny_union", [.05, 0., 2.], [1e-6, 1e-6], tilt,
         camera._replace(width=33, center_x=15.5), (16, 4)),
        ("clipped", [.01, 0., .21], [.002, .002], tilt, camera, (5, 4)),
        ("tie", [1., 2., 2.], [1., 1.], [.5, .5, .5, .5],
         camera._replace(width=1, height=1, focal_x=2., focal_y=1., center_x=-1.5, center_y=-.5), (0, 0)),
        ("parallel", [0., 0., 2.], [.02, .02], [.5, .5, .5, .5],
         camera._replace(width=1, height=1, center_x=.5, center_y=.5), (0, 0)),
        ("near_parallel", [2e-12, 0., 2.], [1., 1.], [.5, .5, .5, .5],
         camera._replace(width=1, height=1, focal_x=5e11, center_x=0., center_y=.5), (0, 0)),
        ("zero_tangents", [0., 0., 2.], [0., 0.], tilt, camera, (4, 4)),
        ("below_alpha", [0., 0., 2.], [.02, .02], tilt, camera, (4, 4)),
    )
    result = {}
    for name, c, scales, quat, s, (x, y) in cases:
        c = torch.tensor(c, dtype=torch.float64)
        mean = (c-s.w2c[:3, 3]) @ s.w2c[:3, :3]
        p = [mean[None], torch.tensor([scales], dtype=torch.float64).log(),
             torch.tensor([quat], dtype=torch.float64),
             torch.tensor([[math.log(.4/.6)]], dtype=torch.float64),
             torch.zeros(1, 1, 3, dtype=torch.float64), torch.zeros(1, 15, 3, dtype=torch.float64)]
        if name in ("screen_cap", "below_alpha"):
            p[3][:] = 6. if name == "screen_cap" else -10.
        settings = settings_to(s, "cuda") if native else s
        render = (lambda q: diff_rasterize_disks(*q, settings)) if native else (lambda q: oracle(q, s))
        data = [t.cuda().float() for t in p] if native else p
        q, refs = differentiable(data), differentiable(p)
        output, reference = render(q), oracle(refs, s)
        torch.testing.assert_close(output.detach().cpu().double(), reference.detach(), atol=4e-5, rtol=4e-5)
        weight = torch.zeros_like(output)
        weight[:, y, x] = output.new_tensor([.7, .9, .4, .2, .3, .2, -.1, .4, .5, 1000.])
        actual = grads(output, q, weight)
        expected = grads(reference, refs, weight.cpu().double())
        errors = []
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.cpu().double(), b, atol=4e-5, rtol=3e-3,
                                       msg=lambda msg: f"lowpass {name}: {msg}")
            errors.append((a.cpu()-b).abs().max().item())
        if name in ("parallel", "near_parallel", "zero_tangents", "below_alpha"):
            if name == "near_parallel":
                assert 0 < .5/s.focal_x < 1e-10
                assert s.near_plane < c[0]/(.5/s.focal_x) < s.far_plane
            assert torch.count_nonzero(output[3:]) == 0
            assert all(torch.count_nonzero(g) == 0 for g in actual)
            result[name] = {"rejected": True}
            continue
        axes = rotate(p[2][0], torch.eye(3, dtype=torch.float64))
        u, v = (axes[:2]*p[1][0].exp()[:, None]) @ s.w2c[:3, :3].T
        ray = c.new_tensor([(x+.5-s.center_x)/s.focal_x, (y+.5-s.center_y)/s.focal_y, 1.])
        uvz = torch.linalg.solve(torch.stack((u, v, -ray), -1), -c)
        gd = torch.exp(-.5*uvz[:2].square().sum())
        delta = c.new_tensor([s.focal_x*(ray[0]-c[0]/c[2]), s.focal_y*(ray[1]-c[1]/c[2])])
        gs = torch.exp(-delta.square().sum())
        if name == "tie":
            assert gd == gs
        elif name == "disk":
            assert gd > gs
        else:
            assert gs > gd
        rgb_weight = torch.zeros_like(output)
        rgb_weight[0, y, x] = 1
        qq = differentiable(data)
        rgb_grads = grads(render(qq), qq, rgb_weight)
        if name == "tie":
            assert rgb_grads[1].min() > .01, "disk-tie footprint adjoint was lost to screen"
        elif name != "disk":
            assert torch.count_nonzero(rgb_grads[1]) == torch.count_nonzero(rgb_grads[2]) == 0
            # Red=.5 with zero SH coefficients: exact projected-centre footprint adjoint.
            gxy = .5*p[3][0, 0].sigmoid()*gs*2*delta
            if name == "screen_cap":
                assert p[3][0, 0].sigmoid()*gs > .99
                assert all(torch.count_nonzero(g) == 0 for g in rgb_grads[:4])
                gxy *= 0
            gc = torch.stack((gxy[0]*s.focal_x/c[2], gxy[1]*s.focal_y/c[2],
                              -(gxy[0]*s.focal_x*c[0]+gxy[1]*s.focal_y*c[1])/c[2].square()))
            torch.testing.assert_close(rgb_grads[0][0].cpu().double(), gc @ s.w2c[:3, :3],
                                       atol=2e-5, rtol=8e-4)
            for channel in (3, 5):  # Screen footprint has no direct q gradient, but depth/normal still do.
                qq = differentiable(data)
                aux_weight = torch.zeros_like(output)
                aux_weight[channel, y, x] = 1
                aux_grads = grads(render(qq), qq, aux_weight)
                assert aux_grads[2].abs().max() > 1e-5
                assert torch.count_nonzero(aux_grads[4]) == torch.count_nonzero(aux_grads[5]) == 0
        if name == "tiny_union":
            assert output[4, y, x] > .2, "screen-only support lost across tile boundary"
            if native:
                counts = _C.disk_forward(*data, *settings.as_tuple()[:-1], data[0].new_empty(0), False, False)[-1].tolist()
                assert counts[0] == 2, counts
        if name == "clipped":
            assert torch.count_nonzero(output[3:, 4, 3]) == 0
            torch.testing.assert_close((output[3, y, x]/output[4, y, x]).cpu().double(),
                                       uvz[2], atol=2e-5, rtol=2e-5)
        worst, count, worst_coordinate = 0., 0, None
        if name != "tie":  # No central finite differences on the nonsmooth max tie.
            for family, indices in {0: range(3), 1: range(2), 2: range(4), 3: [0], 4: [0], 5: [0, 3, 9, 24, 42]}.items():
                for index in indices:
                    eps = .001 if native else 1e-5
                    plus, minus = [t.clone() for t in data], [t.clone() for t in data]
                    plus[family].view(-1)[index] += eps
                    minus[family].view(-1)[index] -= eps
                    fd = ((render(plus)-render(minus))*weight).sum()/(2*eps)
                    target = actual[family].view(-1)[index]
                    torch.testing.assert_close(fd, target, atol=7e-4 if native else 2e-7,
                                               rtol=2e-2 if native else 2e-5,
                                               msg=lambda msg: f"lowpass FD {name} family{family}: {msg}")
                    error = (fd-target).abs().item()
                    if error > worst:
                        worst = error
                        worst_coordinate = {"family": family, "index": index,
                                            "fd": fd.item(), "backward": target.item(),
                                            "relative_error": error/abs(target.item()) if abs(target.item()) > 1e-6 else None}
                    count += 1
        result[name] = {"disk_gaussian": gd.item(), "screen_gaussian": gs.item(),
                        "alpha": output[4, y, x].item(),
                        "normalized_depth": (output[3, y, x]/output[4, y, x]).item(),
                        "median": output[8, y, x].item(),
                        "max_output_error": (output.detach().cpu()-reference.detach()).abs().max().item(),
                        "gradient_max_errors": errors, "fd_checks": count, "fd_max_abs_error": worst,
                        "fd_worst_coordinate": worst_coordinate}
        if native and name == "clipped":
            table = []
            for step in (.001, .0005, .00025):
                plus, minus = [t.clone() for t in data], [t.clone() for t in data]
                plus[0][0, 0] += step
                minus[0][0, 0] -= step
                fd = (((render(plus)-render(minus))*weight).sum()/(2*step)).item()
                pp, mm = [t.clone() for t in p], [t.clone() for t in p]
                pp[0][0, 0] += step
                mm[0][0, 0] -= step
                double_fd = (((oracle(pp, s)-oracle(mm, s))*weight.cpu().double()).sum()/(2*step)).item()
                table.append({"step": step, "native_fd": fd, "double_fd": double_fd,
                              "native_backward": actual[0][0, 0].item(),
                              "abs_error": abs(fd-actual[0][0, 0].item()),
                              "fp32_vs_double_fd": fd-double_fd})
            assert table[2]["abs_error"] < table[1]["abs_error"] < table[0]["abs_error"]
            result[name]["mean_x_fd_convergence"] = table
        print("LOWPASS", "native" if native else "double", name, result[name], flush=True)
    return result


def stock_3d_control(p, s, reference=None):
    import FasterGSCudaBackend.torch_bindings.rasterization as raster
    baseline = _C if reference is None else reference
    p = [t.cuda().float() for t in p]
    p[1] = torch.cat((p[1], p[1][:, :1]-.8), dim=1)
    s = settings_to(s, "cuda")
    output = []
    try:
        for native in (baseline, _C, baseline):
            raster._C = native
            q = differentiable(p)
            rgb = diff_rasterize(*q, torch.empty(0, device="cuda"), s)
            gg = grads(rgb, q, torch.ones_like(rgb)/rgb.numel())
            output.append((rgb.detach(), gg))
    finally:
        raster._C = _C
    maximum = 0.
    for actual in output[1:]:
        torch.testing.assert_close(actual[0], output[0][0], atol=0, rtol=0)
        for a, b in zip(actual[1], output[0][1]):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
            maximum = max(maximum, (a-b).abs().max().item())
    path = Path(baseline.__file__)
    return {"binary": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "control": "official-reference" if reference is not None else "same-build-repeatability",
            "rgb_bitwise_equal": True, "max_gradient_difference_and_repeat_jitter": maximum}


def gpu_gate(stock_reference=None):
    p, s = fixture()
    ps = [t.cuda().float() for t in p]
    ss = settings_to(s, "cuda")
    reference = oracle(p, s).detach()
    native = diff_rasterize_disks(*ps, ss)
    torch.testing.assert_close(native.cpu().double(), reference, atol=2e-5, rtol=2e-5)
    result = {"binary": _C.__file__, "sha256": hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "max_output_error": (native.cpu()-reference).abs().max().item(),
              "objectives": {}}
    native_grads = {}
    for name, weight in objective_weights(s).items():
        q, qq = differentiable(p), differentiable(ps)
        expected = grads(oracle(q, s), q, weight)
        actual = grads(diff_rasterize_disks(*qq, ss), qq, weight.cuda().float())
        errors = []
        for family, (a, b) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(a.cpu().double(), b, atol=1e-5, rtol=8e-4,
                                       msg=lambda msg: f"{name} family{family}: {msg}")
            errors.append((a.cpu()-b).abs().max().item())
        if name not in ("rgb", "mixed"):
            assert torch.count_nonzero(actual[4]) == torch.count_nonzero(actual[5]) == 0, name
        if name == "rgb":
            assert all(t.abs().max() > 1e-7 for t in actual), "RGB positive control must reach every family"
        native_grads[name] = actual
        result["objectives"][name] = {"max_abs_gradient_errors": errors,
                                      "sh_exact_zero": name not in ("rgb", "mixed")}
        print(name, result["objectives"][name], flush=True)
    for rgb, aux, mixed in zip(native_grads["rgb"], native_grads["aux"], native_grads["mixed"]):
        torch.testing.assert_close(rgb+aux, mixed, atol=2e-6, rtol=2e-5)
    # Native float32 FD controls: all means, both scales, quaternion components,
    # opacity, SH0 and nonzero SHrest terms of each degree, for BOTH disks.
    fd_result = {}
    selected = {0: list(range(6)), 1: list(range(4)), 2: list(range(8)),
                3: [0, 1], 4: list(range(6)), 5: [0, 3, 9, 24, 42, 45, 48, 54, 69, 87]}
    for name in ("rgb", "aux", "mixed"):
        weight = objective_weights(s)[name].cuda().float()
        worst = 0.
        for family, indices in selected.items():
            for index in indices:
                plus, minus = [t.clone() for t in ps], [t.clone() for t in ps]
                eps = 1e-3
                plus[family].view(-1)[index] += eps
                minus[family].view(-1)[index] -= eps
                fd = ((diff_rasterize_disks(*plus, ss)-diff_rasterize_disks(*minus, ss))*weight).sum()/(2*eps)
                expected = native_grads[name][family].view(-1)[index]
                torch.testing.assert_close(fd, expected, atol=7e-4, rtol=2e-2,
                                           msg=lambda msg: f"native FD {name} family{family} index{index}: {msg}")
                worst = max(worst, (fd-expected).abs().item())
        fd_result[name] = {"checks": sum(map(len, selected.values())), "epsilon": .001, "max_abs_error": worst}
        print("native_fd", name, fd_result[name], flush=True)
    result["native_fd"] = fd_result

    # Cross bucket boundaries and terminate, away from (1-.99)^2 == 1e-4.
    many = [t.repeat(20, *([1]*(t.ndim-1))) for t in p]
    many[0][:, 2] += torch.arange(40, dtype=torch.float64)*.002
    many[3][:] = 2.3
    expected = oracle(many, s)
    qq = differentiable([t.cuda().float() for t in many])
    actual = diff_rasterize_disks(*qq, ss)
    torch.testing.assert_close(actual.cpu().double(), expected.detach(), atol=4e-5, rtol=4e-5)
    weight = objective_weights(s)["mixed"]
    refs = differentiable(many)
    errors = []
    for a, b in zip(grads(actual, qq, weight.cuda().float()), grads(oracle(refs, s), refs, weight)):
        torch.testing.assert_close(a.cpu().double(), b, atol=4e-5, rtol=3e-3)
        errors.append((a.cpu()-b).abs().max().item())
    assert actual[4].max() > 1-1e-4, "termination positive control"
    state = _C.disk_forward(*[t.cuda().float() for t in many], *ss.as_tuple()[:-1],
                           ps[0].new_empty(0), False, False)[-1].tolist()
    assert state[0] >= 40 and state[1] >= 2, state
    result["forty_overlap_termination"] = {
        "max_output_error": (actual.cpu()-expected.detach()).abs().max().item(),
        "max_gradient_errors": errors, "instances_bucket_upper_selector": state,
        "maximum_alpha": actual[4].max().item(),
    }
    print("forty_overlap_termination", result["forty_overlap_termination"], flush=True)

    result["geometry_cases"] = {}
    near = [t.clone() for t in p]
    near[0][0] = (torch.tensor([.1, -.03, .4])-s.w2c[:3, 3]) @ s.w2c[:3, :3]
    near[0][1, 2] = -20
    near[1][0] = torch.tensor([.5, .45]).log()
    clipped_color = [t.clone() for t in p]
    clipped_color[4][0, 0] = torch.tensor([-3., -4., 1.])
    inactive_sh = [t.clone() for t in p]
    inactive_sh[5] = p[5][:, :0]
    saturation = [t.clone() for t in p]
    saturation[3][:] = 6
    for name, pp, setting in (
        ("multiple_partial_tiles", p, s._replace(width=35, height=19, focal_x=22., focal_y=20., center_x=14.3, center_y=10.2)),
        ("near_crossing", near, s), ("clamped_color", clipped_color, s),
        ("sh0_only_empty_rest", inactive_sh, s._replace(active_sh_bases=1)),
        ("inactive_sh_rest", p, s._replace(active_sh_bases=1)),
        ("alpha_cap", saturation, s),
    ):
        refs = differentiable(pp)
        qq = differentiable([t.cuda().float() for t in pp])
        expected = oracle(refs, setting)
        actual = diff_rasterize_disks(*qq, settings_to(setting, "cuda"))
        torch.testing.assert_close(actual.cpu().double(), expected.detach(), atol=4e-5, rtol=4e-5)
        weight = objective_weights(setting)["mixed"]
        errors = []
        for a, b in zip(grads(actual, qq, weight.cuda().float()), grads(expected, refs, weight)):
            torch.testing.assert_close(a.cpu().double(), b, atol=4e-5, rtol=3e-3,
                                       msg=lambda msg: f"{name}: {msg}")
            errors.append((a.cpu()-b).abs().max().item() if a.numel() else 0.)
        result["geometry_cases"][name] = errors
        print(name, errors, flush=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qq = differentiable(ps)
        actual = diff_rasterize_disks(*qq, ss)
        torch.testing.assert_close(actual.cpu().double(), reference, atol=2e-5, rtol=2e-5)
        for a, b in zip(grads(actual, qq, objective_weights(s)["mixed"].cuda().float()), native_grads["mixed"]):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    stream.synchronize()
    result["nondefault_stream"] = True

    # No visibility, invalid quaternion and malformed boundary inputs.
    invisible = [t.clone() for t in ps]
    invisible[0][:, 2] = -20
    q = differentiable(invisible)
    empty = diff_rasterize_disks(*q, ss)
    assert torch.count_nonzero(empty[3:]) == 0
    torch.testing.assert_close(empty[:3], ss.bg_color[:, None, None].expand_as(empty[:3]), atol=0, rtol=0)
    assert all(torch.count_nonzero(g) == 0 for g in grads(empty, q, torch.ones_like(empty)))
    invalid = [t.clone() for t in ps]
    invalid[2][:] = 0
    assert torch.count_nonzero(diff_rasterize_disks(*invalid, ss)[3:]) == 0
    invalid[2] = ps[2].clone()
    invalid[1][:] = float("inf")
    assert torch.count_nonzero(diff_rasterize_disks(*invalid, ss)[3:]) == 0
    for scales in (torch.zeros(2, 3, device="cuda"), torch.zeros(1, 2, device="cuda"),
                   torch.zeros(2, device="cuda"), torch.zeros(2, 2, dtype=torch.float64, device="cuda")):
        try:
            diff_rasterize_disks(ps[0], scales, *ps[2:], ss)
        except RuntimeError as error:
            assert ("disk scales must be Nx2" if scales.dtype == torch.float32
                    else "disk input must be contiguous float32") in str(error), str(error)
        else:
            raise AssertionError("invalid disk scale input accepted")
    for bad in (ss._replace(width=0), ss._replace(width=65521), ss._replace(width=65520, height=65520),
                ss._replace(focal_x=0), ss._replace(far_plane=.1), ss._replace(proper_antialiasing=True)):
        try:
            diff_rasterize_disks(*ps, bad)
        except (RuntimeError, ValueError) as error:
            expected = ("antialiasing" if bad.proper_antialiasing else
                        "dimensions" if bad.width in (0, 65521, 65520) else "invalid disk camera")
            assert expected in str(error), str(error)
        else:
            raise AssertionError("invalid/unsupported disk camera accepted")
    result["invisible_invalid_quaternion_and_scale_guards"] = True
    # Cropped outputs are constant zeros. Their adjoints must not affect geometry
    # or the pixel-shift proxy, even when the upstream gradient on them is nonzero.
    weights = objective_weights(s)["mixed"].cuda().float()
    for channels in (3, 8, 9):
        full_params, partial_params = differentiable(ps), differentiable(ps)
        shift = torch.zeros((len(ps[0]), 2), device="cuda", requires_grad=True)
        other_shift = shift.detach().clone().requires_grad_()
        full, meta = diff_rasterize_disks(*full_params, ss, image_shift=shift, return_metadata=True)
        partial, new_meta = diff_rasterize_disks(*partial_params, ss, image_shift=other_shift,
                                                  return_metadata=True, output_channels=channels)
        torch.testing.assert_close(partial[:channels], full[:channels], atol=1e-6, rtol=1e-6)
        assert torch.equal(meta, new_meta) and torch.count_nonzero(partial[channels:]) == 0
        mask = weights.clone()
        mask[channels:] = 0
        for expected, actual in zip(grads(full, full_params+[shift], mask),
                                    grads(partial, partial_params+[other_shift], weights)):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=8e-4)
    result["loss_specific_channels_and_shift_gradients"] = True
    single = differentiable([t[:1].clone() for t in ps])
    one = diff_rasterize_disks(*single, ss)
    assert torch.count_nonzero(one[9]) == 0, "single disk distortion must be exactly zero"
    assert all(torch.count_nonzero(g) == 0 for g in grads(one, single, torch.nn.functional.one_hot(
        torch.tensor(9, device="cuda"), 10).float()[:, None, None].expand_as(one)))
    assert native[9].max() > 1e-6, "overlap distortion positive control"
    result["single_zero_and_overlap_nonzero_distortion"] = True
    result["median_crossing"] = median_crossing_control()
    result["lowpass"] = lowpass_controls(native=True)
    result["stock_3d"] = stock_3d_control(p, s, stock_reference)
    torch.cuda.synchronize()
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock-reference", type=Path,
                        help="Optional freshly built official _C shared library for paired 3D control")
    args = parser.parse_args()
    result = {"cpu_double_fd": cpu_gate()}
    print(json.dumps(result, indent=2), flush=True)
    if args.gpu:
        reference = None
        if args.stock_reference:
            spec = importlib.util.spec_from_file_location("stock_reference._C", args.stock_reference)
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
        result["gpu"] = gpu_gate(reference)
    assert not any(name.startswith(("diff_surfel_rasterization", "Methods.Faster2DGS", "gsplat"))
                   for name in sys.modules), "comparison/research renderer imported by implementation test"
    result["implementation_import_exclusion"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("TRUE2D_PRIMITIVE_PASS", flush=True)


if __name__ == "__main__":
    main()
