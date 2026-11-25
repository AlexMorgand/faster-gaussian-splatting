"""Synthetic N2 topology/optimizer, renderer/calibration and actual trainer gates."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def expect_error(callback, text):
    try:
        callback()
    except Exception as error:
        assert text in str(error), str(error)
        return str(error)
    else:
        raise AssertionError(f"missing refusal: {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import Framework
    destination = args.output.parent/(args.output.stem+"_files")
    destination.mkdir(exist_ok=False)
    Framework.Directories.OUTPUT_DIR = destination/"output"
    Framework.Directories.CACHE_DIR = args.output.parent/"cache"
    os.environ["HF_HUB_CACHE"] = str(Framework.Directories.CACHE_DIR/"huggingface")
    Framework.setup(require_custom_config=True, config_path=str(args.config))
    from Cameras.Perspective import PerspectiveCamera
    from Cameras.utils import SharedCameraSettings, quaternion_to_rotation_matrix
    from Datasets.utils import BasicPointCloud, ImageData, View, save_image
    from Methods.FasterGS.Model import Gaussians, FasterGSModel
    from Methods.FasterGS.Trainer import FasterGSTrainer
    from Methods.FasterGS.Renderer import FasterGSRenderer, extract_settings
    old_defaults = deepcopy(FasterGSTrainer.get_default_parameters())
    old_renderer_defaults = deepcopy(FasterGSRenderer.get_default_parameters())
    old_model_defaults = deepcopy(FasterGSModel.get_default_parameters())
    from Methods.FasterGS.disks.Model import Gaussians2D, FasterGS2DModel
    from Methods.FasterGS.disks.Renderer import FasterGS2DRenderer, depth_to_normal
    from Methods.FasterGS.disks.Trainer import FasterGS2DTrainer
    assert FasterGSTrainer.get_default_parameters() == old_defaults
    assert FasterGSRenderer.get_default_parameters() == old_renderer_defaults
    assert FasterGSModel.get_default_parameters() == old_model_defaults
    camera = PerspectiveCamera(
        shared_settings=SharedCameraSettings(np.zeros(3, dtype=np.float32), .2, 20.),
        width=48, height=32, focal_x=40., focal_y=42., center_x=21.3, center_y=17.2)
    angle = .13
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    c2w[:3, 3] = [.08, -.03, .1]
    target_path = destination/"synthetic.png"
    save_image(target_path, torch.rand(3, camera.height, camera.width, device="cuda")*.5)
    view = View(camera, c2w, 0, rgb=ImageData(target_path, n_channels=3))
    second = c2w.copy()
    second[0, 3] += 10.
    view2 = View(camera, second, 1)
    grid = torch.tensor([[-.4, -.3, 2.], [-.1, -.3, 2.], [.2, -.3, 2.], [.4, -.3, 2.1],
                         [-.4, .3, 2.], [-.1, .3, 2.], [.2, .3, 2.], [.4, .3, 2.1]], device="cuda")
    positions = grid @ view.rotation.T+view.position
    cloud = BasicPointCloud(positions, torch.rand_like(positions))
    results = {"outputs": str(destination), "new_uncalibrated_threshold": .1}

    def new_gaussians(kind=Gaussians2D):
        g = kind(3, pretrained=True)
        g.initialize_from_point_cloud(cloud, False)
        wrapper = SimpleNamespace(**deepcopy(FasterGS2DTrainer.get_default_parameters()))
        g.training_setup(wrapper, 10.)
        g.reset_densification_info(False)
        return g

    mapping = {"means": "_means", "scales": "_scales", "rotations": "_rotations",
               "opacities": "_opacities", "sh_coefficients_0": "_sh_coefficients_0",
               "sh_coefficients_rest": "_sh_coefficients_rest"}

    def check_groups(g):
        for group in g.optimizer.param_groups:
            parameter = getattr(g, mapping[group["name"]])
            assert group["params"][0] is parameter
            if g.optimizer.state.get(parameter):
                assert g.optimizer.state[parameter]["exp_avg"].shape == parameter.shape
                assert g.optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape

    for kind in (Gaussians2D, Gaussians):
        g = new_gaussians(kind)
        assert g.raw_scales.shape == (8, kind.scale_dim)
        covariance = g.covariances
        rotation = quaternion_to_rotation_matrix(g.raw_rotations)
        expected = (rotation[:, :, :kind.scale_dim]*g.scales[:, None, :])
        torch.testing.assert_close(covariance, expected @ expected.transpose(1, 2))
        if kind is Gaussians2D:
            assert torch.linalg.eigvalsh(covariance)[:, 0].abs().max() < 1e-6
        else:
            torch.testing.assert_close(g.raw_rotations, g.means.new_tensor([1., 0, 0, 0]).expand(8, 4), atol=0, rtol=0)
        vertices = g.as_ply_dict()["vertex"]
        path = destination/f"roundtrip_{kind.scale_dim}.ply"
        PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
        np.testing.assert_array_equal(PlyData.read(path)["vertex"].data, vertices)
        assert [name for name in vertices.dtype.names if name.startswith("scale_")] == [
            f"scale_{i}" for i in range(kind.scale_dim)]
        results[f"initialization_covariance_ply_{kind.scale_dim}"] = True

    metric = new_gaussians()
    shift_grad = metric.means.new_zeros(8, 2)
    shift_grad[0] = shift_grad.new_tensor([2, 3])
    shift_grad[1] = 100
    metadata = metric.means.new_zeros(2, 8)
    metadata[0, 0], metadata[1, 0] = 5, 1
    metadata[0, 1] = 3  # Supported in both views, fully occluded.
    metric.accumulate_image_shift(shift_grad, metadata, 48, 32)
    metadata[0, 0] = 7
    metric.accumulate_image_shift(shift_grad*0, metadata, 48, 32)
    expected_score = math.sqrt(48**2+48**2)/2
    torch.testing.assert_close(metric.densification_info[1, 0]/metric.densification_info[0, 0],
                               metric.means.new_tensor(expected_score))
    assert metric.densification_info[0, 0] == 2 and metric._max_radii2D[0] == 7
    assert metric.densification_info[0, 1] == 2
    assert metric.densification_info[1, 1] == 0 and metric._max_radii2D[1] == 0
    assert torch.count_nonzero(metric.densification_info[:, 2:]) == 0
    before_metric = metric.densification_info.clone()
    expect_error(lambda: metric.accumulate_image_shift(shift_grad.new_full((8, 2), 1e30), metadata, 48, 32),
                 "nonfinite NDC")
    torch.testing.assert_close(metric.densification_info, before_metric, atol=0, rtol=0)
    results["visible_view_average_ndc_norm"] = expected_score

    for kind in (Gaussians2D, Gaussians):
        grace_g = new_gaussians(kind)
        with torch.no_grad():
            grace_g._scales[:] = math.log(.03)
            grace_g._scales[0] = math.log(.01)
            grace_g._scales[1] = math.log(.2)
            grace_g._rotations[2] = 0
            grace_g._opacities[:] = torch.logit(grace_g._opacities.new_tensor(.01))
            grace_g._densification_info[0] = 1
            grace_g._densification_info[1, :2] = 1
        split_parent = grace_g.means[1].clone()
        grace_g.adaptive_density_control(.5, None, False)
        assert len(grace_g.means) == 9
        assert (grace_g.raw_rotations.square().sum(1) >= 1e-8).all()
        assert not (grace_g.means == split_parent).all(1).any()
        assert grace_g.opacities.max() < .011
        if kind is Gaussians:
            assert (grace_g.means[-2:, 2]-split_parent[2]).abs().max() > 1e-4
    results["reset_grace_retains_opacity_but_prunes_split_parent_and_degenerate_quaternion"] = True
    results["three_dimensional_split_keeps_normal_axis_extent"] = True

    # A radius threshold applies to the observed parent. Its clone starts unseen.
    screen = new_gaussians()
    with torch.no_grad():
        screen._scales.fill_(math.log(.01))
        screen._densification_info[0].fill_(1)
        screen._densification_info[1, 7] = 1
        screen._max_radii2D.copy_(torch.arange(8, device="cuda"))
    screen.sort(torch.arange(7, -1, -1, device="cuda"))
    screen.adaptive_density_control(.5, None, True, additional_prune_mask=screen._max_radii2D >= 6)
    torch.testing.assert_close(screen._max_radii2D, screen.means.new_tensor([5, 4, 3, 2, 1, 0, 0]))

    g = new_gaussians()
    for group in g.optimizer.param_groups:
        group["params"][0].grad = torch.ones_like(group["params"][0])*.1
    g.optimizer.step()
    g.optimizer.zero_grad(set_to_none=True)
    for group in g.optimizer.param_groups:
        parameter = group["params"][0]
        row = (torch.arange(len(parameter), device="cuda", dtype=parameter.dtype)+1).reshape(
            len(parameter), *([1]*(parameter.ndim-1)))
        g.optimizer.state[parameter]["exp_avg"].copy_(row.expand_as(parameter)*.001)
        g.optimizer.state[parameter]["exp_avg_sq"].copy_(row.expand_as(parameter)*.00001)
    with torch.no_grad():
        g._scales[:] = math.log(.03)
        g._scales[0] = math.log(.01)
        g._scales[1] = math.log(.2)
        g._opacities[:] = torch.logit(g._opacities.new_tensor(.2))
        g._densification_info[0] = 1
        g._densification_info[1, :2] = 1
    parent_mean, parent_scale = g.means[1].clone(), g.raw_scales[1].clone()
    parent_normal = quaternion_to_rotation_matrix(g.raw_rotations[1:2])[0, :, 2]
    old_parameters = {group["name"]: group["params"][0] for group in g.optimizer.param_groups}
    old_moments = {name: g.optimizer.state[param]["exp_avg"].clone() for name, param in old_parameters.items()}
    for parameter in old_parameters.values():
        parameter.grad = torch.ones_like(parameter)
    g.adaptive_density_control(.5, .005, False)
    assert len(g.means) == 10 and g._max_radii2D.shape == (10,)
    assert g._densification_info is None
    torch.testing.assert_close((g.means[-2:]-parent_mean) @ parent_normal, torch.zeros(2, device="cuda"), atol=1e-6, rtol=0)
    torch.testing.assert_close(g.raw_scales[-2:], (parent_scale-math.log(1.6)).expand(2, 2))
    check_groups(g)
    for group in g.optimizer.param_groups:
        parameter = group["params"][0]
        assert parameter is not old_parameters[group["name"]] and parameter.grad is None
        expected = old_moments[group["name"]][torch.tensor([0, 2, 3, 4, 5, 6, 7], device="cuda")]
        torch.testing.assert_close(g.optimizer.state[parameter]["exp_avg"][:7], expected, atol=0, rtol=0)
        assert torch.count_nonzero(g.optimizer.state[parameter]["exp_avg"][7:]) == 0
    before = [group["params"][0].clone() for group in g.optimizer.param_groups]
    g.optimizer.step()
    assert all(torch.equal(p, group["params"][0]) for p, group in zip(before, g.optimizer.param_groups))
    mask = torch.zeros(10, dtype=torch.bool, device="cuda")
    mask[[2, 4]] = True
    before_prune = {group["name"]: g.optimizer.state[group["params"][0]]["exp_avg"].clone()
                    for group in g.optimizer.param_groups}
    g.prune(mask)
    for group in g.optimizer.param_groups:
        torch.testing.assert_close(g.optimizer.state[group["params"][0]]["exp_avg"],
                                   before_prune[group["name"]][~mask], atol=0, rtol=0)
    g.reset_densification_info(False)
    order = torch.randperm(8, device="cuda")
    ordered_means = g.means[order].detach().clone()
    ordered_moments = {group["name"]: {
        key: g.optimizer.state[group["params"][0]][key][order].clone()
        for key in ("exp_avg", "exp_avg_sq")} for group in g.optimizer.param_groups}
    g.sort(order)
    torch.testing.assert_close(g.means, ordered_means, atol=0, rtol=0)
    for group in g.optimizer.param_groups:
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(g.optimizer.state[group["params"][0]][key],
                                       ordered_moments[group["name"]][key], atol=0, rtol=0)
    check_groups(g)
    g.apply_morton_ordering()
    check_groups(g)
    results["clone_split_tangent_plane_prune_sort_morton_state_and_grad_none"] = True
    original_id = id(g._opacities)
    g._opacities.grad = torch.full_like(g._opacities, .3)
    g._means.grad = torch.full_like(g._means, .2)
    old_means = g.means.detach().clone()
    with torch.no_grad():
        g.reset_opacities()
    assert id(g._opacities) == original_id and torch.all(g._opacities.grad == .3)
    assert torch.count_nonzero(g.optimizer.state[g._opacities]["exp_avg"]) == 0
    assert torch.count_nonzero(g.optimizer.state[g._opacities]["exp_avg_sq"]) == 0
    assert g.opacities.max() <= .010001
    g.optimizer.step()
    assert not torch.equal(g.means, old_means)
    check_groups(g)
    results["opacity_reset_identity_state_reset_gradient_retention_and_other_group_update"] = True
    from Optim.adam_utils import extend_param_groups
    from FasterGSCudaBackend.torch_bindings.adam import FusedAdam
    first = torch.nn.Parameter(torch.ones(2, 2, device="cuda"))
    untouched = torch.nn.Parameter(torch.ones(2, 1, device="cuda"))
    optimizer = FusedAdam([{"params": [first], "name": "first"}, {"params": [untouched], "name": "other"}],
                          lr=.01, eps=1e-15)
    first.grad, untouched.grad = torch.ones_like(first), torch.ones_like(untouched)
    replaced = extend_param_groups(optimizer, {"first": first.new_zeros(1, 2)})["first"]
    assert replaced.grad is None and optimizer.param_groups[1]["params"][0] is untouched
    before_first, before_other = replaced.clone(), untouched.clone()
    optimizer.step()
    assert torch.equal(replaced, before_first) and not torch.equal(untouched, before_other)
    results["topology_helper_unaffected_group_still_steps"] = True

    model = FasterGS2DModel("synthetic_training").build()
    renderer = FasterGS2DRenderer(model)
    trainer = FasterGS2DTrainer(model=model, renderer=renderer)
    class Dataset:
        point_cloud = cloud
        default_camera = camera
        def train(self):
            return self
        def __iter__(self):
            return iter((view, view2))
    dataset = Dataset()
    trainer.setup_gaussians(0, dataset)
    trainer.train_sampler = SimpleNamespace(get=lambda **kwargs: {"view": view})
    callbacks = [fn.__name__ for fn in trainer._gather_callbacks(0)]
    assert "densify" not in callbacks and "reset_opacities" not in callbacks and "reset_opacities_extra" not in callbacks
    for iteration, expected in ((500, False), (600, True), (14900, True), (15000, False)):
        assert trainer._should_densify(iteration) is expected
    for iteration, expected in ((0, False), (3000, True), (12000, True), (15000, False), (18000, False)):
        assert trainer._should_reset_opacities(iteration, dataset) is expected
    camera.background_color = np.ones(3, dtype=np.float32)
    assert trainer._should_reset_opacities(500, dataset)
    camera.background_color = np.zeros(3, dtype=np.float32)
    observed = []
    original_densify = model.gaussians.adaptive_density_control
    model.gaussians.adaptive_density_control = lambda *args, **kwargs: observed.append((args, kwargs))
    trainer._last_opacity_reset_iteration = 3000
    trainer._run_densify(3100, dataset)
    trainer._run_densify(3200, dataset)
    assert observed[0][0][1] is None and observed[0][0][2] is False
    assert observed[1][0][1] == .05 and observed[1][0][2] is True
    model.gaussians.adaptive_density_control = original_densify
    trainer._last_opacity_reset_iteration = None
    g = model.gaussians
    for iteration in (2999, 3000, 3001, 7000, 7001):
        # Controlled state, not a pretend thousands-step recovery after reset.
        with torch.no_grad():
            g._opacities[:] = torch.logit(g._opacities.new_tensor(.2))
        if iteration == 3000:
            events = []
            with ExitStack() as hooks:
                for owner, name in ((g, "accumulate_image_shift"), (g, "adaptive_density_control"),
                                    (g, "reset_opacities"), (g.optimizer, "step")):
                    original = getattr(owner, name)
                    def record(*args, name=name, original=original, **kwargs):
                        events.append(name)
                        return original(*args, **kwargs)
                    hooks.enter_context(patch.object(owner, name, side_effect=record))
                trainer.training_iteration(iteration, dataset)
            assert events == ["accumulate_image_shift", "adaptive_density_control", "reset_opacities", "step"]
        else:
            trainer.training_iteration(iteration, dataset)
        assert len(g.means) > 0
        weights = {metric.name: metric.weight for metric in trainer.loss.loss_metrics}
        assert weights["DISTORTION_REGULARIZATION"] == (.1 if iteration > 3000 else 0)
        assert weights["NORMAL_REGULARIZATION"] == (.05 if iteration > 7000 else 0)
        assert all(torch.isfinite(parameter).all() for parameter in g.parameters())
        check_groups(g)
    results["actual_training_iterations"] = [2999, 3000, 3001, 7000, 7001]
    results["schedule_callbacks_reset_grace"] = True
    trainer._last_opacity_reset_iteration = 3000
    original_stride = trainer.DENSIFICATION_INTERVAL
    trainer.DENSIFICATION_INTERVAL = 700
    observed.clear()
    model.gaussians.adaptive_density_control = lambda *args, **kwargs: observed.append((args, kwargs))
    trainer._run_densify(3500, dataset)
    assert observed[-1][0][1] is None
    trainer.DENSIFICATION_INTERVAL = original_stride
    trainer._last_opacity_reset_iteration = 500
    trainer._run_densify(600, dataset)
    assert observed[-1][0][1] is None
    model.gaussians.adaptive_density_control = original_densify

    original_iterations = trainer.NUM_ITERATIONS
    trainer.NUM_ITERATIONS = 1000
    for field, bad, message in (("DENSIFICATION_GRAD_THRESHOLD", None, "explicit finite positive"),
                                 ("DENSIFICATION_GRAD_THRESHOLD", float("nan"), "explicit finite positive"),
                                 ("USE_MCMC", True, "MCMC"),
                                 ("DENSIFICATION_USE_ABS_GRAD", True, "ABSgrad")):
        original = getattr(trainer, field)
        setattr(trainer, field, bad)
        expect_error(trainer.validate_contract, message)
        setattr(trainer, field, original)
    trainer.NUM_ITERATIONS = 20
    trainer.DENSIFICATION_GRAD_THRESHOLD = None
    trainer.validate_contract()
    expect_error(lambda: trainer._run_densify(600, dataset), "growth requires")
    trainer.DENSIFICATION_GRAD_THRESHOLD = .1
    trainer.NUM_ITERATIONS = original_iterations
    trainer.FILTER_3D.USE = True
    expect_error(trainer.validate_contract, "3D filters")
    trainer.FILTER_3D.USE = False
    trainer.SPEEDYSPLAT_PRUNING.USE = True
    expect_error(trainer.validate_contract, "3D pruning")
    trainer.SPEEDYSPLAT_PRUNING.USE = False
    for function, text in ((lambda: g.setup_3d_filter(None, None), "3D filters"),
                           (lambda: g.mcmc_densification(), "MCMC"),
                           (lambda: g.post_optimizer_step(True), "MCMC noise"),
                           (lambda: g.reset_densification_info(True), "ABSgrad"),
                           (lambda: renderer.compute_pruning_scores(None), "3D pruning"),
                           (lambda: renderer.render_image_benchmark(None), "3D inference")):
        expect_error(function, text)
    results["unsupported_and_threshold_fail_closed"] = True
    expect_error(lambda: g.load_state_dict({"_scales": torch.zeros(len(g.means), 3, device="cuda")}, strict=False),
                 "requires Nx2")
    model.PPISP.USE = True
    expect_error(model.build, "does not support PPISP")
    model.PPISP.USE = False
    assert FasterGSModel.get_default_parameters() == old_model_defaults
    renderer.FORCE_OPTIMIZED_INFERENCE = True
    expect_error(lambda: renderer.render_image(view), "3D inference")
    renderer.FORCE_OPTIMIZED_INFERENCE = False

    # Independently unproject a calibrated oblique plane.
    yy, xx = torch.meshgrid(torch.arange(camera.height, device="cuda"),
                            torch.arange(camera.width, device="cuda"), indexing="ij")
    rayx, rayy = (xx+.5-camera.center_x)/camera.focal_x, (yy+.5-camera.center_y)/camera.focal_y
    depth = (-2/(.2*rayx-.1*rayy-1))[None]
    normals = depth_to_normal(view, depth)
    expect_error(lambda: depth_to_normal(view, depth[:, :2]), "at least a 3x3")
    expected_normal = g.means.new_tensor([.2, -.1, -1.])
    expected_normal = view.rotation @ (expected_normal/expected_normal.norm())
    torch.testing.assert_close(normals[:, 1:-1, 1:-1], expected_normal[:, None, None].expand_as(normals[:, 1:-1, 1:-1]),
                               atol=2e-5, rtol=2e-5)
    # Actual final PR57 policy, not parse-only or paper Eq14 loss parity.
    tangent = view.rotation[:, 0]
    tangent = tangent-expected_normal*(tangent @ expected_normal)
    tangent = tangent/tangent.norm()
    rendered_direction = .8*expected_normal+.6*tangent
    normal_metric = next(metric for metric in trainer.loss.loss_metrics if metric.name == "NORMAL_REGULARIZATION")
    policy = []
    for value in (0., .25, 1.):
        alpha = torch.tensor(value, device="cuda", requires_grad=True)
        depth_leaf = depth.detach().clone().requires_grad_()
        output = torch.cat((torch.zeros(3, 32, 48, device="cuda"), alpha*depth_leaf,
                            alpha.expand(1, 32, 48),
                            alpha*rendered_direction[:, None, None].expand(3, 32, 48),
                            depth_leaf, torch.zeros(1, 32, 48, device="cuda")))
        parsed = renderer._parse(view, output, True)
        package = {key: parsed[key][:, 1:-1, 1:-1] for key in ("rend_normal", "surf_normal")}
        loss = normal_metric.metric_func(rendered=package["rend_normal"], surface=package["surf_normal"])
        gn, ga, gd = torch.autograd.grad(loss, (package["rend_normal"], alpha, depth_leaf))
        torch.testing.assert_close(loss, loss.new_tensor(1-.8*value), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(gn, -package["surf_normal"].detach()/(30*46), atol=1e-7, rtol=1e-5)
        assert ga.abs() < 2e-5 and torch.isfinite(gd).all()
        if value > 0:
            assert gd.abs().max() > 1e-6
        policy.append({"alpha": value, "loss": loss.item(), "alpha_gradient": ga.item(),
                       "depth_gradient_max": gd.abs().max().item()})
    results["preserved_PR57_final_detached_alpha_normal_loss_policy"] = policy
    model.eval()
    displayed = renderer.render_image(view, to_chw=False)
    assert displayed["normal"].shape == (32, 48, 3)
    assert torch.isfinite(displayed["normal"]).all()
    visible = displayed["alpha"][..., 0] > 1e-4
    assert visible.any()
    assert torch.count_nonzero(displayed["normal"][~visible]) == 0
    signed = displayed["normal"][visible]*2-1
    assert signed.norm(dim=-1).min() > .99 and signed.std(dim=0).max() > .005
    renderer.RENDER_DIFFUSE = renderer.RENDER_SPECULAR = False
    assert renderer.render_image(view, to_chw=True)["rgb"].abs().max() < 1e-6
    renderer.RENDER_DIFFUSE = renderer.RENDER_SPECULAR = True
    results["calibrated_oblique_depth_normal_and_alpha_masked_display"] = True
    with patch.object(view.camera, "width", 32768), patch.object(view.camera, "height", 32768):
        expect_error(lambda: renderer.render_image(view), "insufficient headroom")
    from FasterGSCudaBackend.torch_bindings.disks import preflight_disks
    def hazardous_parameters(n):
        return [torch.tensor([0., 0., 1.], device="cuda").expand(n, 3).contiguous(),
                torch.full((n, 2), math.log(10.), device="cuda"),
                torch.full((n, 4), .5, device="cuda"), torch.zeros(n, 1, device="cuda"),
                torch.zeros(n, 1, 3, device="cuda"), torch.zeros(n, 0, 3, device="cuda")]
    big = extract_settings(view, 1, camera.background_color, False)._replace(width=1024, height=1024)
    single_count = preflight_disks(*hazardous_parameters(1), big)
    assert single_count > 4096
    expect_error(lambda: preflight_disks(*hazardous_parameters(1),
                                       big._replace(width=32768, height=32768)),
                 "insufficient headroom before preflight")
    hazard = hazardous_parameters(4097)
    expect_error(lambda: preflight_disks(*hazard, big), "instance/buffer actual demand")
    overflow = hazardous_parameters(1048577)
    overflow_message = expect_error(
        lambda: preflight_disks(*overflow, big), "instance/buffer actual demand")
    expected_count = single_count*1048577
    assert expected_count > 2**32
    assert f"actual_instances={expected_count}" in overflow_message, overflow_message
    results["capacity_refusal_before_render"] = True
    results["int64_instance_sum_prevents_uint32_wrap"] = overflow_message
    from FasterGSCudaBackend import _C
    from FasterGSCudaBackend.torch_bindings.rasterization import RasterizerSettings
    edge_settings = RasterizerSettings(torch.eye(4, device="cuda"), torch.zeros(3, device="cuda"),
        torch.zeros(3, device="cuda"), 1, 33, 9, 10., 10., 15.5, 4.5, .2, 20., False)
    edge = [torch.tensor([[.05, 0., 2.]], device="cuda"), torch.full((1, 2), math.log(1e-6), device="cuda"),
            torch.tensor([[1., 0., 0., 0.]], device="cuda"), torch.tensor([[0.]], device="cuda"),
            torch.zeros(1, 1, 3, device="cuda"), torch.zeros(1, 0, 3, device="cuda")]
    for label, z, log_scale in (("tile_edge", 2., math.log(1e-6)), ("horizon", .21, 0.), ("culled", .1, 0.)):
        edge[0][:, 2], edge[1][:] = z, log_scale
        edge[2][:] = edge[2].new_tensor(
            [math.cos(math.pi/8), 0., -math.sin(math.pi/8), 0.] if label == "horizon" else [1., 0, 0, 0])
        count_only = preflight_disks(*edge, edge_settings)
        buffers = _C.disk_forward(*edge, *edge_settings.as_tuple()[:-1], edge[0].new_empty(0), True, False)
        actual_count = buffers[-1][0].item()
        assert count_only == actual_count
        visible = buffers[-2][1, 0].item()
        assert visible == (0 if label == "culled" else 1)
        results[f"capacity_{label}"] = {"count_only": count_only, "actual": actual_count, "visible": visible}
    model.num_iterations_trained = 5
    checkpoint = destination/"synthetic.pt"
    model.save(checkpoint)
    loaded = FasterGS2DModel.load(str(checkpoint))
    loaded.eval()
    again = FasterGS2DRenderer(loaded).render_image(view, to_chw=False)
    torch.testing.assert_close(again["normal"], displayed["normal"], atol=1e-6, rtol=1e-5)
    results["real_model_checkpoint_roundtrip"] = str(checkpoint)
    from Methods.FasterGS.disks.Loss import FasterGS2DLoss
    sh_updates = []
    for degree in range(4):
        g.active_sh_degree, g.active_sh_bases = degree, (degree+1)**2
        path = destination/f"sh{degree}.pt"
        model.save(path)
        adam_path = destination/f"sh{degree}.adam"
        torch.save(g.optimizer.state_dict(), adam_path)
        reloaded = FasterGS2DModel.load(str(path))
        rg = reloaded.gaussians
        rg.training_setup(trainer, g.training_cameras_extent)
        rg.optimizer.load_state_dict(torch.load(adam_path, weights_only=True))
        rg._densification_info = g._densification_info.clone()
        rg._max_radii2D = g._max_radii2D.clone()
        pairs = []
        for subject, render in ((model, renderer), (reloaded, FasterGS2DRenderer(reloaded))):
            subject.train()
            subject.gaussians.optimizer.zero_grad(set_to_none=True)
            package = render.render_image_training(view, True, camera.background_color, compute_surf_normal=True)
            objective = FasterGS2DLoss(trainer.LOSS, subject)
            objective(package, view.rgb).backward()
            sg = subject.gaussians
            assert torch.count_nonzero(sg._sh_coefficients_rest.grad[:, (degree+1)**2-1:]) == 0
            sg.accumulate_image_shift(package["image_shift"].grad, package["metadata"], 48, 32)
            pairs.append({key: package[key].detach().clone()
                          for key in ("rgb", "rend_normal", "surf_normal")})
            sg.optimizer.step()
            sg.optimizer.zero_grad(set_to_none=True)
        for key in pairs[0]:
            torch.testing.assert_close(pairs[0][key], pairs[1][key], atol=0, rtol=0)
        for key, value in g.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, rg.state_dict()[key], atol=2e-5, rtol=2e-3)
            else:
                assert value == rg.state_dict()[key]
        torch.testing.assert_close(g._densification_info, rg._densification_info, atol=2e-5, rtol=2e-3)
        torch.testing.assert_close(g._max_radii2D, rg._max_radii2D, atol=0, rtol=0)
        for original_group, restored_group in zip(g.optimizer.param_groups, rg.optimizer.param_groups):
            original_state = g.optimizer.state[original_group["params"][0]]
            restored_state = rg.optimizer.state[restored_group["params"][0]]
            for key in original_state:
                if isinstance(original_state[key], torch.Tensor):
                    torch.testing.assert_close(original_state[key], restored_state[key], atol=2e-5, rtol=2e-3)
                else:
                    assert original_state[key] == restored_state[key]
        sh_updates.append(degree)
    results["sh_degrees_with_rgb_normal_bitwise_reload_and_actual_adam_next_update"] = sh_updates
    assert FasterGSTrainer.get_default_parameters() == old_defaults, "experimental import changed 3D defaults"
    assert FasterGSRenderer.get_default_parameters() == old_renderer_defaults
    assert FasterGSModel.get_default_parameters() == old_model_defaults
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2), flush=True)
    print("TRUE2D_INTEGRATION_PASS", flush=True)


if __name__ == "__main__":
    main()
