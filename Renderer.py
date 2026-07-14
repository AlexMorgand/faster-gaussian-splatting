"""FasterGS/Renderer.py"""

import math

import torch

import Framework
from Cameras.utils import invert_3d_affine, rotation_matrix_to_quaternion
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View, transform_world_normal_map, world_normal_foreground_mask
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGS.Model import FasterGSModel
from Methods.FasterGS.FasterGSCudaBackend import diff_rasterize, diff_rasterize_dr, rasterize, update_pruning_scores, RasterizerSettings
from Methods.FasterGS.DeferredShading import compose_deferred


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
    proper_antialiasing: bool,
    sh_rotation: torch.Tensor | None = None,
    w2c: torch.Tensor | None = None,
    cam_position: torch.Tensor | None = None,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGS renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    if sh_rotation is None:
        sh_rotation = torch.eye(3, dtype=torch.float32, device=view.position.device)
    if w2c is None:
        w2c = view.w2c
    if cam_position is None:
        cam_position = view.position
    return RasterizerSettings(
        w2c,
        cam_position,
        sh_rotation,
        bg_color,
        active_sh_bases,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        proper_antialiasing,
    )


def _quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions in (w, x, y, z) layout."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
    PROPER_ANTIALIASING=False,
    FORCE_OPTIMIZED_INFERENCE=False,
)
class FasterGSRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGS renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGS renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def _get_turntable_render_data(
        self,
        view: View,
        means: torch.Tensor,
        rotations: torch.Tensor,
        use_original_camera_fast_path: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Applies per-view object rotation for fixed-camera turntable captures, if present."""
        original_c2w = view.exif.get('turntable_original_c2w')
        if use_original_camera_fast_path and original_c2w is not None:
            # fixed-camera turntable rendering is equivalent to rendering the untransformed
            # Gaussians from the original COLMAP camera, and avoids per-splat tensor transforms.
            original_w2c = view.exif.get('turntable_original_w2c')
            if original_w2c is None:
                original_w2c = invert_3d_affine(original_c2w)
                view.exif['turntable_original_w2c'] = original_w2c
            w2c = torch.as_tensor(original_w2c, dtype=means.dtype, device=means.device)
            cam_position = torch.as_tensor(original_c2w[:3, 3], dtype=means.dtype, device=means.device)
            sh_rotation = torch.eye(3, dtype=means.dtype, device=means.device)
            return means, rotations, sh_rotation, w2c, cam_position

        transform = view.exif.get('turntable_object_transform')
        if transform is None:
            return means, rotations, torch.eye(3, dtype=means.dtype, device=means.device), None, None

        T = torch.as_tensor(transform, dtype=means.dtype, device=means.device)
        R = T[:3, :3]
        t = T[:3, 3]
        means_t = means @ R.T + t

        q_R = rotation_matrix_to_quaternion(R.unsqueeze(0)).to(dtype=rotations.dtype, device=rotations.device)[0]
        rotations_t = _quaternion_multiply(q_R.expand_as(rotations), rotations)

        return means_t, rotations_t, R.T.contiguous(), None, None

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.render_image_benchmark(view, to_chw=to_chw or benchmark)
        elif self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def _deferred_features(self, view: View, cam_position: torch.Tensor | None) -> torch.Tensor:
        """Per-Gaussian deferred-reflection feature tensor (N, 4) = (normal.xyz, refl strength)."""
        camera_position = cam_position if cam_position is not None else view.position
        normals = self.model.gaussians.min_axis_normals(camera_position)  # (N, 3), world space
        refl = self.model.gaussians.reflection_strength  # (N, 1)
        return torch.cat([normals, refl], dim=1).contiguous()

    def _mesh_normal_terms(self, view: View) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Returns optional mesh world normals and a foreground mask for DR priors."""
        mesh_normal = view.world_normal
        if mesh_normal is None:
            return None, None
        alignment = view.exif.get('scene_alignment_transform')
        if alignment is not None:
            mesh_normal = transform_world_normal_map(mesh_normal, alignment)
        mask = view.segmentation
        if mask is None:
            mask = world_normal_foreground_mask(mesh_normal)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        # 3DGS-DR: zero GT normals outside the asset mask before hijack / supervision.
        mesh_normal = mesh_normal * mask
        return mesh_normal, mask

    def _compose_deferred(
        self,
        base_image: torch.Tensor,
        feature_map: torch.Tensor,
        view: View,
    ) -> dict[str, torch.Tensor]:
        mesh_normal, mesh_mask = self._mesh_normal_terms(view)
        hijack = mesh_normal is not None and bool(
            getattr(Framework.config.MODEL.DEFERRED_REFLECTION, 'MESH_NORMAL_HIJACK', True)
        )
        return compose_deferred(
            base_image,
            feature_map,
            view,
            self.model.environment,
            mesh_normal_map=mesh_normal,
            mesh_normal_mask=mesh_mask,
            mesh_normal_hijack=hijack,
        )

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        means, rotations, sh_rotation, w2c, cam_position = self._get_turntable_render_data(
            view,
            self.model.gaussians.means,
            self.model.gaussians.raw_rotations,
            use_original_camera_fast_path=True,
        )
        settings = extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, self.PROPER_ANTIALIASING, sh_rotation, w2c, cam_position)
        if self.model.gaussians.deferred_reflection:
            features = self._deferred_features(view, cam_position)
            base_image, feature_map = diff_rasterize_dr(
                means=means,
                scales=self.model.gaussians.raw_scales,
                rotations=rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                features=features,
                densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
                rasterizer_settings=settings,
            )
            decomposition = self._compose_deferred(base_image, feature_map, view)
            image = decomposition['rgb']
        else:
            image = diff_rasterize(
                means=means,
                scales=self.model.gaussians.raw_scales,
                rotations=rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
                rasterizer_settings=settings,
            )
            decomposition = None
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        outputs: dict[str, torch.Tensor] = {'rgb': image}
        if decomposition is not None:
            outputs['gaussian_normal'] = decomposition['gaussian_normal']
            if decomposition['mesh_normal'] is not None:
                outputs['mesh_normal'] = decomposition['mesh_normal']
                outputs['mesh_normal_mask'] = decomposition['mesh_normal_mask']
        return outputs

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        means, rotations, sh_rotation, w2c, cam_position = self._get_turntable_render_data(
            view,
            self.model.gaussians.means,
            self.model.gaussians.raw_rotations,
            use_original_camera_fast_path=True,
        )
        settings = extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, self.PROPER_ANTIALIASING, sh_rotation, w2c, cam_position)
        scales = self.model.gaussians.raw_scales + math.log(max(self.SCALE_MODIFIER, 1e-6))
        decomposition = None
        if self.model.gaussians.deferred_reflection:
            features = self._deferred_features(view, cam_position)
            base_image, feature_map = diff_rasterize_dr(
                means=means,
                scales=scales,
                rotations=rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                features=features,
                densification_info=torch.empty(0),
                rasterizer_settings=settings,
            )
            decomposition = self._compose_deferred(base_image, feature_map, view)
            image = decomposition['rgb']
        else:
            image = diff_rasterize(
                means=means,
                scales=scales,
                rotations=rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                densification_info=torch.empty(0),
                rasterizer_settings=settings,
            )
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        else:
            image = image.clamp(0.0, 1.0)

        def _to_display(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if to_chw else tensor.permute(1, 2, 0)

        outputs = {'rgb': _to_display(image)}
        # expose the deferred-reflection decomposition so ICGui's output-mode picker
        # can show base color / reflection color / reflection strength / normal
        if decomposition is not None:
            outputs['base_color'] = _to_display(decomposition['base_color'].clamp(0.0, 1.0))
            outputs['reflection_color'] = _to_display(decomposition['reflection_color'].clamp(0.0, 1.0))
            outputs['reflection_strength'] = _to_display(decomposition['reflection_strength'].clamp(0.0, 1.0))
            outputs['normal'] = _to_display((decomposition['normal'] * 0.5 + 0.5).clamp(0.0, 1.0))
        return outputs

    @torch.inference_mode()
    def render_image_benchmark(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        # the optimized inference rasterizer has no deferred-reflection path; fall back to
        # the DR inference renderer so reflections are never silently dropped in the GUI
        if self.model.gaussians.deferred_reflection:
            with torch.no_grad():
                return self.render_image_inference(view, to_chw=to_chw)
        means, rotations, sh_rotation, w2c, cam_position = self._get_turntable_render_data(
            view,
            self.model.gaussians.means,
            self.model.gaussians.raw_rotations,
            use_original_camera_fast_path=True,
        )
        image = rasterize(
            means=means,
            scales=self.model.gaussians.raw_scales,
            rotations=rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, self.PROPER_ANTIALIASING, sh_rotation, w2c, cam_position),
            to_chw=to_chw,
            clamp_output=self.model.ppisp is None,
        )
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        return {'rgb': image}

    def ppisp_controller_distillation(self, view: View) -> torch.Tensor:
        """Renders an image for a given view where only the PPISP module will receive gradients."""
        image = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, self.PROPER_ANTIALIASING),
            to_chw=True,
            clamp_output=False,
        )
        image = self.model.ppisp(image, view)
        return image

    @torch.inference_mode()
    def compute_pruning_scores(self, dataset: BaseDataset) -> torch.Tensor:
        """Computes the pruning scores for the current dataset."""
        scores = torch.zeros(self.model.gaussians.means.shape[0], device=self.model.gaussians.means.device, dtype=torch.float32)
        for view in dataset:
            means, rotations, sh_rotation, w2c, cam_position = self._get_turntable_render_data(
                view,
                self.model.gaussians.means,
                self.model.gaussians.raw_rotations,
                use_original_camera_fast_path=True,
            )
            update_pruning_scores(
                scores=scores,
                means=means,
                scales=self.model.gaussians.raw_scales,
                rotations=rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, self.PROPER_ANTIALIASING, sh_rotation, w2c, cam_position),
            )
        return scores

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
