"""Disk training recipe adapted from MIT-licensed NeRF-ICG revision 57ddd49."""
from copy import deepcopy
import math

import torch

import Framework
from Logging import Logger
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGS.Trainer import FasterGSTrainer
from .Loss import FasterGS2DLoss
from .Renderer import RenderLayout


class FasterGS2DTrainer(FasterGSTrainer):
    loss_type = FasterGS2DLoss
    _configuration = deepcopy(FasterGSTrainer.get_default_parameters())
    _configuration.update(DENSIFICATION_GRAD_THRESHOLD=None, DENSIFICATION_MAX_SCREEN_SIZE=None,
                          DENSIFICATION_START_ITERATION=500, DENSIFICATION_END_ITERATION=15_000,
                          DENSIFICATION_OPACITY_CULL=.05, DISTORTION_START_ITERATION=3_000,
                          NORMAL_START_ITERATION=7_000)
    _configuration.LOSS.update(LAMBDA_DISTORTION=0.0, LAMBDA_NORMAL=.05)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_contract()
        self._last_opacity_reset_iteration = None

    def validate_contract(self):
        for enabled, name in ((self.USE_MCMC, "MCMC"), (self.FILTER_3D.USE, "3D filters"),
                              (self.SPEEDYSPLAT_PRUNING.USE, "3D pruning scorer"),
                              (self.DENSIFICATION_USE_ABS_GRAD, "ABSgrad")):
            if enabled:
                raise Framework.TrainingError(f"FasterGS2D does not support {name}")
        value = self.DENSIFICATION_GRAD_THRESHOLD
        if (value is None and self.NUM_ITERATIONS > self.DENSIFICATION_START_ITERATION) or (
            value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0)
        ):
            raise Framework.TrainingError("FasterGS2D requires an explicit finite positive DENSIFICATION_GRAD_THRESHOLD for its image-shift metric")
        radius = self.DENSIFICATION_MAX_SCREEN_SIZE
        if radius is not None and (type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0):
            raise Framework.TrainingError("FasterGS2D screen pruning threshold must be None or finite positive pixels")
        if self.model.ppisp is not None:
            raise Framework.TrainingError("FasterGS2D experimental training does not support PPISP/distillation")
        if any(type(interval) is not int or interval <= 0
               for interval in (self.DENSIFICATION_INTERVAL, self.OPACITY_RESET_INTERVAL)):
            raise Framework.TrainingError("FasterGS2D schedule intervals must be positive")

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, iteration, dataset):
        self.validate_contract()
        super().setup_gaussians(iteration, dataset)
        Logger.log_info(f"FasterGS2D densification: per-supported-view mean ||dL/dpixel_shift*(W/2,H/2)||; threshold={self.DENSIFICATION_GRAD_THRESHOLD}")

    def _should_densify(self, iteration):
        return (self.DENSIFICATION_START_ITERATION < iteration < self.DENSIFICATION_END_ITERATION
                and iteration % self.DENSIFICATION_INTERVAL == 0)

    def _should_reset_opacities(self, iteration, dataset):
        return 0 < iteration < self.DENSIFICATION_END_ITERATION and (
            iteration % self.OPACITY_RESET_INTERVAL == 0
            or (iteration == self.EXTRA_OPACITY_RESET_ITERATION
                and bool(dataset.default_camera.background_color.any())))

    @training_callback(priority=100, active=False)
    def densify(self, *args):
        """Reflective override: run after backward, not the inherited pre-loss callback."""

    @training_callback(priority=90, active=False)
    def reset_opacities(self, *args):
        """Reflective override: reset after densify, before optimizer."""

    @training_callback(priority=89, active=False)
    def reset_opacities_extra(self, *args):
        """Reflective override: folded into the explicit post-backward reset gate."""

    @torch.no_grad()
    def _run_densify(self, iteration, dataset):
        if self.DENSIFICATION_GRAD_THRESHOLD is None:
            raise Framework.TrainingError("FasterGS2D growth requires an explicit finite positive image-shift threshold")
        reset = self._last_opacity_reset_iteration
        grace = reset is not None and iteration == (reset//self.DENSIFICATION_INTERVAL+1)*self.DENSIFICATION_INTERVAL
        large = iteration > self.OPACITY_RESET_INTERVAL and not grace
        gaussians = self.model.gaussians
        screen_mask = None
        if large and self.DENSIFICATION_MAX_SCREEN_SIZE is not None:
            screen_mask = gaussians._max_radii2D > self.DENSIFICATION_MAX_SCREEN_SIZE
        gaussians.adaptive_density_control(
            self.DENSIFICATION_GRAD_THRESHOLD, None if grace else self.DENSIFICATION_OPACITY_CULL,
            large, additional_prune_mask=screen_mask)
        gaussians.reset_densification_info(False)

    def _render_training_view(self, iteration, view, bg_color, collect):
        normal_active = iteration > self.NORMAL_START_ITERATION
        self.loss.set_geometry_loss_weights(
            distortion_scale=float(iteration > self.DISTORTION_START_ITERATION),
            normal_scale=float(normal_active))
        use_normal = normal_active and self.loss.base_normal > 0
        use_distortion = iteration > self.DISTORTION_START_ITERATION and self.loss.base_distortion > 0
        if use_distortion:
            layout = RenderLayout.DISTORTION
        elif use_normal:
            layout = RenderLayout.MEDIAN_DEPTH_NORMALS if self.renderer.DEPTH_RATIO > 0 else RenderLayout.EXPECTED_DEPTH_NORMALS
        else:
            layout = RenderLayout.RGB
        return self.renderer.render_image_training(
            view, collect, bg_color, compute_surf_normal=use_normal, output_channels=layout)

    @torch.no_grad()
    def _after_backward(self, iteration, dataset, view, rendered, collect):
        g = self.model.gaussians
        if collect:
            g.accumulate_image_shift(rendered["image_shift"].grad, rendered["metadata"],
                                     view.camera.width, view.camera.height)
        if self._should_densify(iteration):
            self._run_densify(iteration, dataset)
        if self._should_reset_opacities(iteration, dataset):
            g.reset_opacities()
            self._last_opacity_reset_iteration = iteration

    @post_training_callback(priority=1000)
    def finalize(self, iteration, dataset):
        # Old trainer checkpoints can carry the inherited 3D opacity threshold.
        self.MIN_OPACITY_AFTER_TRAINING = 0.0
        super().finalize(iteration, dataset)
