"""FasterGS/Trainer.py"""

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
        self.loss = FasterGSLoss(loss_config=self.LOSS, model=self.model)
        if self._gaussian_ply_init_active:
            Logger.log_info(
                f'using Gaussian PLY initialization from "{Path(ply_path).expanduser()}" '
                f'(freeze_geometry={self.FREEZE_INITIAL_GAUSSIAN_GEOMETRY}, '
                f'densification={self.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT})'
            )

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if self._gaussian_ply_init_active and not self.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT:
            return

        interval_multiplier = self.DENSIFICATION_INTERVAL_MULTIPLIER_GAUSSIAN_PLY_INIT if self._gaussian_ply_init_active else 1
        effective_interval = max(1, self.DENSIFICATION_INTERVAL * interval_multiplier)
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
    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        if self._gaussian_ply_init_active and self.KEEP_THIN_SPLATS:
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
        use_random_bg = self.USE_RANDOM_BACKGROUND_COLOR or (
            self.RANDOM_BACKGROUND_IF_ALPHA_OR_MASK and supervision_alpha is not None
        )
        bg_color = torch.rand_like(view.camera.background_color) if use_random_bg else view.camera.background_color
        image = self.renderer.render_image_training(
            view=view,
            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
        )
        # calculate loss
        # compose gt with background color if needed  # FIXME: integrate into data model
        rgb_gt = view.rgb
        if supervision_alpha is not None:
            rgb_gt = apply_background_color(rgb_gt, supervision_alpha, bg_color)
        loss = self.loss(image, rgb_gt)
        # backward
        loss.backward()
        # optimizer step
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
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
