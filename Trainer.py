"""FasterGS/Trainer.py"""

from pathlib import Path

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, apply_background_color, get_supervision_alpha
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGS.Loss import FasterGSLoss
from Methods.FasterGS.utils import enable_expandable_segments, carve
from Optim.Samplers.DatasetSamplers import DatasetSampler


def _training_extent_camera_position(view) -> torch.Tensor:
    """Use physical capture positions for turntable views, not the fixed render camera."""
    original_c2w = view.exif.get('turntable_original_c2w')
    if original_c2w is None:
        return view.position
    return torch.as_tensor(original_c2w[:3, 3], dtype=view.position.dtype, device=view.position.device)


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=600,  # while official code states 500, densification actually starts at 600 there
    DENSIFICATION_END_ITERATION=14_900,  # should be set to 24900 when using MCMC; while official code states 15000, densification actually stops at 14900 there
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,  # only used when USE_MCMC=False
    DENSIFICATION_PERCENT_DENSE=0.01,  # only used when USE_MCMC=False
    SPEEDYSPLAT_PRUNING=Framework.ConfigParameterList(
        USE=False,  # only used when USE_MCMC=False
        START_ITERATION=6_000,
        END_ITERATION=30_000,
        INTERVAL=3_000,
        SOFT_PRUNING_RATIO=0.8,
        HARD_PRUNING_RATIO=0.3,
    ),
    USE_MCMC=False,
    MAX_PRIMITIVES=1_000_000,  # only used when USE_MCMC=True
    OPACITY_RESET_INTERVAL=3_000,  # will be skipped when USE_MCMC=True
    EXTRA_OPACITY_RESET_ITERATION=500,  # will be skipped when USE_MCMC=True
    MORTON_ORDERING_INTERVAL=5000,  # lowering to 2500 or 1000 may improve performance when number of Gaussians is high
    MORTON_ORDERING_END_ITERATION=15000,  # should be set to 25000 when using MCMC
    FILTER_3D=Framework.ConfigParameterList(
        USE=False,
        ORIGINAL_FORMULATION=False,  # if True, the original formulation from the Mip-Splatting paper is used
        FILTER_VARIANCE=0.2,
    ),
    USE_RANDOM_BACKGROUND_COLOR=False,  # prevents the model from overfitting to the background color
    RANDOM_BACKGROUND_IF_ALPHA_OR_MASK=True,  # when alpha/mask is available, composite GT and render on same random color
    WHITE_BACKGROUND=False,  # 3DGS-DR --white-background: fixed render/GT background; disables random-bg-from-alpha
    INITIALIZATION_GAUSSIAN_PLY_PATH=None,  # optional Gaussian PLY initialization (e.g. Mesh2Splat output)
    # If set to [r, g, b] in [0, 1], replaces all SH from the PLY with this view-neutral color (DC only, rest zero).
    INITIALIZATION_GAUSSIAN_PLY_DEFAULT_RGB=None,
    FREEZE_INITIAL_GAUSSIAN_GEOMETRY=True,  # when using INITIALIZATION_GAUSSIAN_PLY_PATH, freeze means/scales/rotations
    KEEP_THIN_SPLATS=True,  # when using INITIALIZATION_GAUSSIAN_PLY_PATH, avoid opacity resets that remove very thin details
    ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT=False,  # default off to preserve initialization geometry
    DENSIFICATION_GRAD_THRESHOLD_MULTIPLIER_GAUSSIAN_PLY_INIT=4.0,  # used only when densification is enabled with Gaussian PLY init
    DENSIFICATION_INTERVAL_MULTIPLIER_GAUSSIAN_PLY_INIT=4,  # used only when densification is enabled with Gaussian PLY init
    NORMALIZE_LOSS_BY_MASK_AREA=True,  # legacy; no effect with current loss (kept for configs)
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    RANDOM_INITIALIZATION=Framework.ConfigParameterList(
        FORCE=False,  # if True, the point cloud from the dataset will be ignored
        IGNORE_GAUSSIAN_PLY=False,  # if True, skip INITIALIZATION_GAUSSIAN_PLY_PATH and use point cloud or random AABB init
        N_POINTS=100_000,  # number of random points to be sampled within the scene bounding box
        ENABLE_CARVING=True,  # removes points that are never in-frustum in any training view
        CARVING_IN_ALL_FRUSTUMS=False,  # removes points not in-frustum in all views
        CARVING_ENFORCE_ALPHA=False,  # removes points that project to a pixel with alpha=0 in any view where the point is in-frustum
    ),
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,  # weight for the per-pixel L1 loss on the rgb image
        LAMBDA_DSSIM=0.2,  # weight for the DSSIM loss on the rgb image
        LAMBDA_OPACITY_REGULARIZATION=0.0,  # should be set to 0.01 when using MCMC
        LAMBDA_SCALE_REGULARIZATION=0.0,  # should be set to 0.01 when using MCMC
        LAMBDA_NORMAL=0.0,  # mesh normal supervision (3DGS-DR default 0.1 when enabled)
        LAMBDA_ALBEDO_PRIOR=0.0,  # pull rasterized base_color toward mesh albedo (3DGS-DR default 0.02)
        LAMBDA_REFL_PRIOR=0.0,  # pull reflection strength toward mesh metallic (3DGS-DR default 0.05)
        REFL_PRIOR_SCALE=0.4,  # scale metallic target before reflection-strength prior
        LAMBDA_ENVMAP_ANCHOR=0.0,  # pull cubemap back toward HDRI bake (set when ENVMAP_HDRI is used)
        NORMAL_LOSS_UNTIL_ITERATION=0,  # 0 = no upper bound
        ALBEDO_PRIOR_UNTIL_ITERATION=0,
        REFL_PRIOR_UNTIL_ITERATION=0,
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
        LEARNING_RATE_SH_COEFFICIENTS_0=0.0025,
        LEARNING_RATE_SH_COEFFICIENTS_REST=0.000125,  # 0.0025 / 20
        LEARNING_RATE_OPACITIES=0.025,  # use 0.05 (old default in official code) with MCMC densification or Speedy-Splat pruning to match the respective paper
        LEARNING_RATE_SCALES=0.005,
        LEARNING_RATE_ROTATIONS=0.001,
        LEARNING_RATE_REFLECTION_STRENGTH=0.006,  # deferred reflection: per-Gaussian reflection strength
    ),
    # Deferred reflection schedule (3DGS-DR). Only active when MODEL.DEFERRED_REFLECTION.USE is set.
    DEFERRED_REFLECTION_SCHEDULE=Framework.ConfigParameterList(
        INIT_UNTIL_ITERATION=3_000,      # view-independent bootstrap; reflection/env optimization off, SH pinned to degree 0
        PROPAGATION_INTERVAL=1_000,      # normal propagation cadence (3DGS-DR: every 1000 iters during prop window)
        PROPAGATION_END_ITERATION=8_000, # base window; add LONGER_PROPAGATION_ITERATIONS for scenes like toaster
        LONGER_PROPAGATION_ITERATIONS=0, # 3DGS-DR --longer_prop_iter (e.g. 24_000 for toaster)
        PROPAGATION_ENLARGE_SCALE=1.5,   # scale-up factor for the two longest axes of reflective Gaussians
        PROPAGATION_MIN_OPACITY=0.9,     # reset_opacity1 ceiling during propagation
        PROPAGATION_OPACITY_FLOOR=0.01,  # reset_opacity0 floor on opacity-reset iterations
        PROPAGATION_MIN_REFLECTION=1e-3,
        SCALE_ENLARGE_THRESHOLD=0.02,    # enlarge_refl_scales REFL_MSK_THR (3DGS-DR default)
        COLOR_SABOTAGE_THRESHOLD=0.05,   # dist_color REFL_MSK_THR (3DGS-DR default)
        REFLECTION_THRESHOLD=0.1,        # r_i > this counts as reflective for specular termination
        COLOR_SABOTAGE_NOISE=0.4,        # dist_color DIST_RANGE (3DGS-DR default)
        SPECULAR_TERMINATION_PATIENCE=0, # 0 = disabled (3DGS-DR runs the full propagation window)
        OPAC_LR0_INTERVAL=200,         # 3DGS-DR opac_lr0_interval; 0 disables opacity-lr cycling during propagation
        ENVMAP_LEARNING_RATE=0.01,
        DENSIFICATION_INTERVAL_DURING_PROPAGATION=500,  # 3DGS-DR densification_interval_when_prop
        # 3DGS-DR parity experiments (enable cumulatively; defaults preserve pre-parity behaviour).
        PARITY_DIFFUSE_BOOTSTRAP=True,  # iter <= INIT: diffuse-only rasterizer (3DGS-DR c3 parity)
        PARITY_SKIP_VANILLA_OPACITY_RESET_DURING_PROP=False,  # no FasterGS reset_opacities during prop
        PARITY_PROPAGATION_AFTER_INIT=False,  # propagation window (INIT, PROP_END] not [INIT, PROP_END]
        # Real-scene env scope (3DGS-DR --use_env_scope): limit reflection learning to a world-space sphere.
        USE_ENV_SCOPE=False,
        ENV_SCOPE_CENTER=[0.0, 0.0, 0.0],
        ENV_SCOPE_RADIUS=0.0,
        REFL_MASK_LOSS_WEIGHT=0.4,  # 3DGS-DR REFL_MSK_LOSS_W
    ),
)
class FasterGSTrainer(GuiTrainer):
    """Defines the trainer for the FasterGS variant."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            if enable_expandable_segments():
                self.requires_empty_cache = False
                Logger.log_info('using "expandable_segments:True" with the torch cuda memory allocator')
        super().__init__(**kwargs)
        self.train_sampler = None
        self.loss = None
        self._gaussian_ply_init_active = False
        # deferred reflection state
        self.env_optimizer = None
        self._dr_active = False
        self._dr_specular_terminated = False
        self._dr_reflective_history: list[int] = []
        self._dr_best_reflective = 0
        self._dr_stall_count = 0
        self._envmap_bake_snapshot: dict[str, torch.Tensor] | None = None
        self._env_scope_center: torch.Tensor | None = None
        self._env_scope_radius_sq: float | None = None

    def _env_scope_outside_mask(self) -> torch.Tensor | None:
        """Gaussians outside the env-scope sphere (3DGS-DR ``get_outside_msk``)."""
        if self._env_scope_center is None or self._env_scope_radius_sq is None:
            return None
        dist_sq = torch.sum((self.model.gaussians.means - self._env_scope_center) ** 2, dim=-1)
        return dist_sq > self._env_scope_radius_sq

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        dataset.train()
        self._gaussian_ply_init_active = False
        camera_centers = torch.stack([_training_extent_camera_position(view) for view in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        if radius <= 1e-6:
            bbox_radius = (0.5 * torch.linalg.norm(dataset.bounding_box.size)).item()
            radius = max(bbox_radius, 1.0)
            Logger.log_warning(
                f'training camera extent was degenerate; using fallback extent {radius:.2f}'
            )
        Logger.log_info(f'training cameras extent: {radius:.2f}')

        ply_path = self.INITIALIZATION_GAUSSIAN_PLY_PATH
        use_ply = ply_path not in (None, '') and not self.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY
        if self.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY and ply_path not in (None, ''):
            Logger.log_info(
                'RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY: skipping INITIALIZATION_GAUSSIAN_PLY_PATH; '
                'using COLMAP point cloud or random AABB initialization'
            )

        if use_ply:
            self.model.gaussians.initialize_from_gaussian_ply(str(Path(ply_path).expanduser()))
            self._gaussian_ply_init_active = True
            if getattr(dataset, 'scene_alignment_transform', None) is not None:
                self.model.gaussians.apply_scene_alignment_transform(dataset.scene_alignment_transform)
            default_rgb = self.INITIALIZATION_GAUSSIAN_PLY_DEFAULT_RGB
            if default_rgb is not None:
                rgb_list = list(default_rgb) if isinstance(default_rgb, (list, tuple)) else [default_rgb]
                if len(rgb_list) != 3:
                    raise Framework.TrainingError(
                        f'INITIALIZATION_GAUSSIAN_PLY_DEFAULT_RGB must have length 3, got {default_rgb!r}'
                    )
                rgb_t = torch.tensor([float(rgb_list[0]), float(rgb_list[1]), float(rgb_list[2])], dtype=torch.float32)
                self.model.gaussians.reset_spherical_harmonics_to_rgb(rgb_t)
                Logger.log_info(f'reset PLY-init SH to default RGB {rgb_list}')
        elif dataset.point_cloud is not None and not self.RANDOM_INITIALIZATION.FORCE:
            point_cloud = dataset.point_cloud
            self.model.gaussians.initialize_from_point_cloud(point_cloud, self.USE_MCMC)
        else:
            samples = torch.rand((self.RANDOM_INITIALIZATION.N_POINTS, 3), dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)
            positions = samples * dataset.bounding_box.size + dataset.bounding_box.min
            if self.RANDOM_INITIALIZATION.ENABLE_CARVING:
                positions = carve(positions, dataset, self.RANDOM_INITIALIZATION.CARVING_IN_ALL_FRUSTUMS, self.RANDOM_INITIALIZATION.CARVING_ENFORCE_ALPHA)
            point_cloud = BasicPointCloud(positions)
            self.model.gaussians.initialize_from_point_cloud(point_cloud, self.USE_MCMC)
            Logger.log_info(
                f'random AABB Gaussian initialization: {self.RANDOM_INITIALIZATION.N_POINTS:,} points '
                f'(carving={self.RANDOM_INITIALIZATION.ENABLE_CARVING})'
            )
        self.model.gaussians.training_setup(self, radius, freeze_geometry=self._gaussian_ply_init_active and self.FREEZE_INITIAL_GAUSSIAN_GEOMETRY)
        if not self.USE_MCMC:
            self.model.gaussians.reset_densification_info()
        if self.FILTER_3D.USE and not self._gaussian_ply_init_active:
            self.model.gaussians.setup_3d_filter(self.FILTER_3D, dataset)
        if self.model.ppisp is not None:
            self.model.ppisp.initialize(dataset, self.NUM_ITERATIONS)

        # deferred reflection: environment map optimizer + view-independent bootstrap
        self._dr_active = self.model.gaussians.deferred_reflection
        if self._dr_active:
            if self.model.environment is None:
                raise Framework.TrainingError('deferred reflection enabled but model.environment is not built')
            dr_cfg = self.model.DEFERRED_REFLECTION
            hdri_path = getattr(dr_cfg, 'ENVMAP_HDRI', None)
            if hdri_path not in (None, ''):
                from Methods.FasterGS.envmap_utils import bake_hdri_into_environment_map, snapshot_envmap_parameters
                # setup_gaussians runs under @torch.no_grad(); baking needs a live autograd graph.
                with torch.enable_grad():
                    bake_hdri_into_environment_map(
                        self.model.environment,
                        str(Path(hdri_path).expanduser()),
                        exposure=float(getattr(dr_cfg, 'ENVMAP_EXPOSURE', 1.0)),
                        yaw_deg=float(getattr(dr_cfg, 'ENVMAP_YAW_DEG', 0.0)),
                    )
                self._envmap_bake_snapshot = snapshot_envmap_parameters(self.model.environment)
                anchor_lambda = float(getattr(self.LOSS, 'LAMBDA_ENVMAP_ANCHOR', 0.0))
                if anchor_lambda > 0.0:
                    Logger.log_info(
                        f'envmap HDRI anchor enabled (lambda={anchor_lambda}); '
                        'cubemap stays near bake while fitting proxy2real specular'
                    )
            self.env_optimizer = torch.optim.Adam(
                self.model.environment.parameters(),
                lr=self.DEFERRED_REFLECTION_SCHEDULE.ENVMAP_LEARNING_RATE,
                eps=1e-15,
            )
            if getattr(dr_cfg, 'FREEZE_ENVMAP', False):
                self._set_envmap_lr(0.0)
                Logger.log_info('freeze_envmap: cubemap fixed after HDRI bake')
            # freeze reflection strength during the view-independent bootstrap
            self._set_reflection_strength_lr(0.0)
            dr_sched = self.DEFERRED_REFLECTION_SCHEDULE
            longer = dr_sched.LONGER_PROPAGATION_ITERATIONS
            prop_end = dr_sched.PROPAGATION_END_ITERATION + longer
            if longer > 0:
                expected_densify_end = dr_sched.PROPAGATION_END_ITERATION + longer
                if self.DENSIFICATION_END_ITERATION < expected_densify_end:
                    Logger.log_warning(
                        f'extending DENSIFICATION_END_ITERATION '
                        f'{self.DENSIFICATION_END_ITERATION} -> {expected_densify_end} '
                        f'(LONGER_PROPAGATION_ITERATIONS={longer})'
                    )
                    self.DENSIFICATION_END_ITERATION = expected_densify_end
            Logger.log_info(
                f'deferred reflection active: bootstrap until iter {dr_sched.INIT_UNTIL_ITERATION}, '
                f'propagation until iter {prop_end} every {dr_sched.PROPAGATION_INTERVAL} iters, '
                f'densify until iter {self.DENSIFICATION_END_ITERATION}, '
                f'opac_lr0_interval={dr_sched.OPAC_LR0_INTERVAL}'
            )
            parity_flags = [
                name
                for name, enabled in (
                    ('diffuse_bootstrap', dr_sched.PARITY_DIFFUSE_BOOTSTRAP),
                    ('skip_vanilla_opacity_reset', dr_sched.PARITY_SKIP_VANILLA_OPACITY_RESET_DURING_PROP),
                    ('propagation_after_init', dr_sched.PARITY_PROPAGATION_AFTER_INIT),
                )
                if enabled
            ]
            if parity_flags:
                Logger.log_info(f'3DGS-DR parity flags: {", ".join(parity_flags)}')
            if dr_sched.USE_ENV_SCOPE:
                center = [float(c) for c in dr_sched.ENV_SCOPE_CENTER]
                radius = float(dr_sched.ENV_SCOPE_RADIUS)
                if radius <= 0.0:
                    raise Framework.TrainingError('USE_ENV_SCOPE=True requires ENV_SCOPE_RADIUS > 0')
                self._env_scope_center = torch.tensor(center, dtype=torch.float32, device='cuda')
                self._env_scope_radius_sq = radius * radius
                Logger.log_info(
                    f'env scope active: center={center}, radius={radius:.4f}, '
                    f'refl_mask_loss_weight={dr_sched.REFL_MASK_LOSS_WEIGHT}'
                )

        self.loss = FasterGSLoss(loss_config=self.LOSS, model=self.model)
        normals_path = getattr(Framework.config.DATASET, 'EXTERNAL_NORMALS_PATH', None)
        if normals_path not in (None, ''):
            Logger.log_info(
                f'mesh normal prior enabled from "{normals_path}" '
                f'(hijack={self.model.DEFERRED_REFLECTION.MESH_NORMAL_HIJACK}, '
                f'lambda_normal={self.LOSS.LAMBDA_NORMAL})'
            )
        if self.WHITE_BACKGROUND or bool(getattr(Framework.config.DATASET, 'WHITE_BACKGROUND', False)):
            Logger.log_info('white-background training: fixed render/GT background (3DGS-DR --white-background parity)')
        if self._gaussian_ply_init_active:
            Logger.log_info(
                f'using Gaussian PLY initialization from "{Path(ply_path).expanduser()}" '
                f'(freeze_geometry={self.FREEZE_INITIAL_GAUSSIAN_GEOMETRY}, '
                f'densification={self.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT})'
            )

    def _set_reflection_strength_lr(self, lr: float) -> None:
        """Sets the learning rate of the reflection-strength optimizer group (DR only)."""
        if self.model.gaussians.optimizer is None:
            return
        for group in self.model.gaussians.optimizer.param_groups:
            if group['name'] == 'reflection_strength':
                group['lr'] = lr

    def _set_envmap_lr(self, lr: float) -> None:
        """Sets the learning rate of the environment-map optimizer (DR only)."""
        if self.env_optimizer is None:
            return
        for group in self.env_optimizer.param_groups:
            group['lr'] = lr

    def _set_opacity_lr(self, lr: float) -> None:
        """Sets the learning rate of the opacity optimizer group (DR propagation schedule)."""
        self.model.gaussians.set_opacity_lr(lr)

    def _dr_propagation_window(self, iteration: int) -> bool:
        """True when iteration is inside the 3DGS-DR normal-propagation maintenance window."""
        if not self._dr_active:
            return False
        cfg = self.DEFERRED_REFLECTION_SCHEDULE
        prop_end = cfg.PROPAGATION_END_ITERATION + cfg.LONGER_PROPAGATION_ITERATIONS
        init_until = cfg.INIT_UNTIL_ITERATION
        if cfg.PARITY_PROPAGATION_AFTER_INIT:
            return init_until < iteration <= prop_end
        return init_until <= iteration <= prop_end

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, iteration: int, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree.

        With deferred reflection, higher-order SH is delayed until specular
        termination so it does not interfere with reflection discovery (paper 3.3).
        A propagation-end fallback ensures SH still ramps up even if the reflective
        count never plateaus within the propagation window.
        """
        if self._dr_active and not self._dr_specular_terminated:
            prop_end = (
                self.DEFERRED_REFLECTION_SCHEDULE.PROPAGATION_END_ITERATION
                + self.DEFERRED_REFLECTION_SCHEDULE.LONGER_PROPAGATION_ITERATIONS
            )
            if iteration <= prop_end:
                return
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=95, start_iteration='DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION', end_iteration='DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION')
    @torch.no_grad()
    def dr_enable_reflection(self, *_) -> None:
        """End the view-independent bootstrap: enable reflection-strength optimization."""
        if not self._dr_active:
            return
        self._set_reflection_strength_lr(self.OPTIMIZER.LEARNING_RATE_REFLECTION_STRENGTH)
        Logger.log_info('deferred reflection: bootstrap complete, reflection strength + env map now optimizing')

    @training_callback(priority=85, start_iteration='DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION', end_iteration='NUM_ITERATIONS', iteration_stride='DEFERRED_REFLECTION_SCHEDULE.PROPAGATION_INTERVAL')
    @torch.no_grad()
    def dr_normal_propagation(self, iteration: int, _dataset: 'BaseDataset') -> None:
        """Normal propagation schedule matching 3DGS-DR train.py (paper 3.3).

        Every ``PROPAGATION_INTERVAL`` steps until ``PROPAGATION_END + LONGER_PROP``:
          - on ``OPACITY_RESET_INTERVAL`` multiples: ``reset_opacity0`` + ``reset_refl``
          - otherwise: ``reset_opacity1`` + ``dist_color`` + ``enlarge_refl_scales``
          - optional ``OPAC_LR0_INTERVAL`` cycling: zero opacity lr between resets
        """
        if not self._dr_active or self._dr_specular_terminated:
            return
        if not self._dr_propagation_window(iteration):
            return
        cfg = self.DEFERRED_REFLECTION_SCHEDULE
        prop_end = cfg.PROPAGATION_END_ITERATION + cfg.LONGER_PROPAGATION_ITERATIONS
        outside_msk = self._env_scope_outside_mask()

        on_opacity_reset = iteration % self.OPACITY_RESET_INTERVAL == 0
        if on_opacity_reset:
            self.model.gaussians.dr_reset_opacity_floor(cfg.PROPAGATION_OPACITY_FLOOR)
            self.model.gaussians.dr_bump_reflection_strength(
                cfg.PROPAGATION_MIN_REFLECTION,
                exclusive_msk=outside_msk,
            )
        else:
            self.model.gaussians.dr_reset_opacity_ceiling(
                cfg.PROPAGATION_MIN_OPACITY,
                exclusive_msk=outside_msk,
            )
            self.model.gaussians.color_sabotage(
                refl_threshold=cfg.COLOR_SABOTAGE_THRESHOLD,
                noise=cfg.COLOR_SABOTAGE_NOISE,
                exclusive_msk=outside_msk,
            )
            self.model.gaussians.dr_enlarge_reflective_scales(
                refl_threshold=cfg.SCALE_ENLARGE_THRESHOLD,
                enlarge_scale=cfg.PROPAGATION_ENLARGE_SCALE,
                exclusive_msk=outside_msk,
            )
            if cfg.OPAC_LR0_INTERVAL > 0 and iteration != prop_end:
                self._set_opacity_lr(0.0)

        if cfg.SPECULAR_TERMINATION_PATIENCE > 0:
            n_reflective = self.model.gaussians.n_reflective(cfg.REFLECTION_THRESHOLD)
            if n_reflective > self._dr_best_reflective:
                self._dr_best_reflective = n_reflective
                self._dr_stall_count = 0
            else:
                self._dr_stall_count += 1
                if self._dr_stall_count >= cfg.SPECULAR_TERMINATION_PATIENCE:
                    self._dr_specular_terminated = True
                    Logger.log_info(
                        f'deferred reflection: specular termination at iter {iteration} '
                        f'({n_reflective:,} reflective Gaussians); enabling higher-order SH'
                    )

    @training_callback(
        priority=84,
        start_iteration='DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION',
        end_iteration='NUM_ITERATIONS',
        iteration_stride='DEFERRED_REFLECTION_SCHEDULE.OPAC_LR0_INTERVAL',
    )
    @torch.no_grad()
    def dr_restore_opacity_lr(self, iteration: int, *_) -> None:
        """3DGS-DR: periodically restore full opacity lr during the propagation window."""
        if not self._dr_active:
            return
        cfg = self.DEFERRED_REFLECTION_SCHEDULE
        if cfg.OPAC_LR0_INTERVAL <= 0:
            return
        prop_end = cfg.PROPAGATION_END_ITERATION + cfg.LONGER_PROPAGATION_ITERATIONS
        if cfg.INIT_UNTIL_ITERATION < iteration <= prop_end and iteration % cfg.OPAC_LR0_INTERVAL == 0:
            self._set_opacity_lr(self.OPTIMIZER.LEARNING_RATE_OPACITIES)

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if self._gaussian_ply_init_active and not self.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT:
            return

        interval_multiplier = self.DENSIFICATION_INTERVAL_MULTIPLIER_GAUSSIAN_PLY_INIT if self._gaussian_ply_init_active else 1
        effective_interval = max(1, self.DENSIFICATION_INTERVAL * interval_multiplier)
        if self._dr_active:
            dr_cfg = self.DEFERRED_REFLECTION_SCHEDULE
            prop_end = dr_cfg.PROPAGATION_END_ITERATION + dr_cfg.LONGER_PROPAGATION_ITERATIONS
            if dr_cfg.INIT_UNTIL_ITERATION < iteration <= prop_end:
                effective_interval = max(1, dr_cfg.DENSIFICATION_INTERVAL_DURING_PROPAGATION * interval_multiplier)
        if (iteration - self.DENSIFICATION_START_ITERATION) % effective_interval != 0:
            return

        if self.USE_MCMC:
            self.model.gaussians.mcmc_densification(min_opacity=0.005, cap_max=self.MAX_PRIMITIVES)
        else:
            grad_threshold = self.DENSIFICATION_GRAD_THRESHOLD
            if self._gaussian_ply_init_active:
                grad_threshold *= self.DENSIFICATION_GRAD_THRESHOLD_MULTIPLIER_GAUSSIAN_PLY_INIT
            self.model.gaussians.adaptive_density_control(grad_threshold, 0.005, iteration > self.OPACITY_RESET_INTERVAL)

            if self.SPEEDYSPLAT_PRUNING.USE and self.SPEEDYSPLAT_PRUNING.START_ITERATION <= iteration < self.SPEEDYSPLAT_PRUNING.END_ITERATION and iteration % self.SPEEDYSPLAT_PRUNING.INTERVAL == 0:
                # Soft Pruning (see https://github.com/j-alex-hanson/speedy-splat/blob/e480b2c3944e4aac4e251307216fe1b8d6a0afc3/train.py#L178-L188)
                scores = self.renderer.compute_pruning_scores(dataset.train())
                self.model.gaussians.importance_pruning(scores, pruning_ratio=self.SPEEDYSPLAT_PRUNING.SOFT_PRUNING_RATIO)

            if iteration < self.DENSIFICATION_END_ITERATION:
                self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()
        if self.FILTER_3D.USE and not self._gaussian_ply_init_active:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(priority=99, end_iteration='MORTON_ORDERING_END_ITERATION', iteration_stride='MORTON_ORDERING_INTERVAL')
    @torch.no_grad()
    def morton_ordering(self, *_) -> None:
        """Apply morton ordering to all Gaussian parameters and their optimizer states."""
        self.model.gaussians.apply_morton_ordering()

    @training_callback(active='FILTER_3D.USE', priority=95, start_iteration='DENSIFICATION_END_ITERATION', iteration_stride=100)
    @torch.no_grad()
    def recompute_3d_filter(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Recompute 3D filter."""
        if self._gaussian_ply_init_active:
            return
        if self.DENSIFICATION_END_ITERATION < iteration < self.NUM_ITERATIONS - 100:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(priority=90, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def reset_opacities(self, iteration: int, *_) -> None:
        """Reset opacities."""
        if self._gaussian_ply_init_active and self.KEEP_THIN_SPLATS:
            return
        dr_cfg = self.DEFERRED_REFLECTION_SCHEDULE
        if (
            self._dr_active
            and dr_cfg.PARITY_SKIP_VANILLA_OPACITY_RESET_DURING_PROP
            and self._dr_propagation_window(iteration)
        ):
            return
        if not self.USE_MCMC:
            self.model.gaussians.reset_opacities()

    @training_callback(priority=90, start_iteration='EXTRA_OPACITY_RESET_ITERATION', end_iteration='EXTRA_OPACITY_RESET_ITERATION')
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a white background."""
        if self._gaussian_ply_init_active and self.KEEP_THIN_SPLATS:
            return
        # original implementation only supports black or white background, this is an attempt to make it work with any color
        if not self.USE_MCMC and dataset.default_camera.background_color.sum() != 0.0:
            Logger.log_info('resetting opacities one additional time because using non-black background')
            self.model.gaussians.reset_opacities()

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random view
        view = self.train_sampler.get(dataset=dataset)['view']
        supervision_alpha = get_supervision_alpha(view)
        dataset_white_bg = bool(getattr(Framework.config.DATASET, 'WHITE_BACKGROUND', False))
        use_white_background = self.WHITE_BACKGROUND or dataset_white_bg
        use_random_bg = (
            not use_white_background
            and (
                self.USE_RANDOM_BACKGROUND_COLOR
                or (self.RANDOM_BACKGROUND_IF_ALPHA_OR_MASK and supervision_alpha is not None)
            )
        )
        bg_color = torch.rand_like(view.camera.background_color) if use_random_bg else view.camera.background_color
        dr_sched = self.DEFERRED_REFLECTION_SCHEDULE
        bootstrap_diffuse = (
            self._dr_active
            and dr_sched.PARITY_DIFFUSE_BOOTSTRAP
            and iteration <= dr_sched.INIT_UNTIL_ITERATION
        )
        render_out = self.renderer.render_image_training(
            view=view,
            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
            use_diffuse_bootstrap=bootstrap_diffuse,
        )
        image = render_out['rgb']
        # calculate loss
        # compose gt with background color if needed  # FIXME: integrate into data model
        rgb_gt = view.rgb
        if supervision_alpha is not None:
            rgb_gt = apply_background_color(rgb_gt, supervision_alpha, bg_color)
        loss = self.loss(image, rgb_gt)
        normal_until = int(getattr(self.LOSS, 'NORMAL_LOSS_UNTIL_ITERATION', 0) or 0)
        apply_normal_loss = (
            self.LOSS.LAMBDA_NORMAL > 0.0
            and iteration >= self.DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION
            and (normal_until <= 0 or iteration <= normal_until)
            and 'gaussian_normal' in render_out
            and 'mesh_normal' in render_out
        )
        if apply_normal_loss:
            loss = loss + self.loss.mesh_normal_supervision_loss(
                render_out['gaussian_normal'],
                render_out['mesh_normal'],
                render_out.get('mesh_normal_mask'),
            )
        dr_cfg = self.model.DEFERRED_REFLECTION
        force_albedo = bool(getattr(dr_cfg, 'FORCE_ALBEDO_BASE_COLOR', False))
        refl_until = int(getattr(self.LOSS, 'REFL_PRIOR_UNTIL_ITERATION', 0) or 0)
        if (
            self.LOSS.LAMBDA_REFL_PRIOR > 0.0
            and iteration >= self.DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION
            and (refl_until <= 0 or iteration <= refl_until)
            and 'reflection_strength' in render_out
            and 'mesh_metallic' in render_out
        ):
            loss = loss + self.loss.reflection_strength_prior_loss(
                render_out['reflection_strength'],
                render_out['mesh_metallic'],
                render_out.get('mesh_metallic_mask'),
            )
        albedo_until = int(getattr(self.LOSS, 'ALBEDO_PRIOR_UNTIL_ITERATION', 0) or 0)
        if (
            not force_albedo
            and self.LOSS.LAMBDA_ALBEDO_PRIOR > 0.0
            and iteration >= self.DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION
            and (albedo_until <= 0 or iteration <= albedo_until)
            and 'base_color' in render_out
            and 'mesh_albedo' in render_out
        ):
            loss = loss + self.loss.albedo_prior_loss(
                render_out['base_color'],
                render_out['mesh_albedo'],
                render_out.get('mesh_albedo_mask'),
                render_out.get('mesh_metallic'),
            )
        anchor_lambda = float(getattr(self.LOSS, 'LAMBDA_ENVMAP_ANCHOR', 0.0))
        if (
            anchor_lambda > 0.0
            and self._envmap_bake_snapshot is not None
            and iteration >= self.DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION
        ):
            from Methods.FasterGS.envmap_utils import envmap_anchor_loss
            loss = loss + anchor_lambda * envmap_anchor_loss(
                self.model.environment,
                self._envmap_bake_snapshot,
            )
        dr_sched = self.DEFERRED_REFLECTION_SCHEDULE
        if (
            self._dr_active
            and dr_sched.USE_ENV_SCOPE
            and iteration >= dr_sched.INIT_UNTIL_ITERATION
        ):
            outside_msk = self._env_scope_outside_mask()
            if outside_msk is not None and outside_msk.any():
                refls = self.model.gaussians.reflection_strength.flatten()
                loss = loss + dr_sched.REFL_MASK_LOSS_WEIGHT * refls[outside_msk].mean()
        # backward
        loss.backward()
        # optimizer step
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        # deferred reflection: optimize the environment map after the bootstrap
        if self._dr_active and self.env_optimizer is not None:
            if iteration >= self.DEFERRED_REFLECTION_SCHEDULE.INIT_UNTIL_ITERATION:
                self.env_optimizer.step()
            self.env_optimizer.zero_grad(set_to_none=True)
        if self.model.ppisp is not None:
            self.model.ppisp.step()

    @training_callback(active='SPEEDYSPLAT_PRUNING.USE', priority=70, start_iteration='SPEEDYSPLAT_PRUNING.START_ITERATION', end_iteration='SPEEDYSPLAT_PRUNING.END_ITERATION', iteration_stride='SPEEDYSPLAT_PRUNING.INTERVAL')
    @torch.no_grad()
    def hard_pruning(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Speedy-Splat Hard Pruning (see https://github.com/j-alex-hanson/speedy-splat/blob/e480b2c3944e4aac4e251307216fe1b8d6a0afc3/train.py#L202-L213)."""
        if iteration >= self.DENSIFICATION_END_ITERATION + self.DENSIFICATION_INTERVAL:
            scores = self.renderer.compute_pruning_scores(dataset.train())
            self.model.gaussians.importance_pruning(scores, pruning_ratio=self.SPEEDYSPLAT_PRUNING.HARD_PRUNING_RATIO)

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds Gaussian count to default Weights & Biases logging."""
        Framework.wandb.log({
            '#Gaussians': self.model.gaussians.means.shape[0]
        }, step=iteration)
        # default logging
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, _, dataset: 'BaseDataset') -> None:
        """Clean up after training."""
        n_gaussians = self.model.gaussians.training_cleanup(min_opacity=self.MIN_OPACITY_AFTER_TRAINING)
        Logger.log_info(f'final number of Gaussians: {n_gaussians:,}')
        with open(str(self.output_directory / 'n_gaussians.txt'), 'w') as n_gaussians_file:
            n_gaussians_file.write(
                f'Final number of Gaussians: {n_gaussians:,}\n'
                f'\n'
                f'N_Gaussians:{n_gaussians}'
            )
        if self.model.ppisp is not None and self.model.ppisp.config.controller_distillation:
            Logger.log_info(f'distilling PPISP controller')
            with torch.enable_grad():
                self.model.train()
                dataset.train()
                self.loss.train()
                for _ in Logger.log_progress(range(self.model.ppisp.config.controller_training_steps)):
                    # get random view
                    view = self.train_sampler.get(dataset=dataset)['view']
                    # render
                    image = self.renderer.ppisp_controller_distillation(view=view)
                    # calculate loss
                    # compose gt with background color if needed  # FIXME: integrate into data model
                    rgb_gt = view.rgb
                    if (supervision_alpha := get_supervision_alpha(view)) is not None:
                        rgb_gt = apply_background_color(rgb_gt, supervision_alpha, view.camera.background_color)
                    loss = self.loss(image, rgb_gt)
                    # backward
                    loss.backward()
                    # optimizer step
                    self.model.ppisp.step()
            self.model.ppisp.create_report(self.output_directory)
