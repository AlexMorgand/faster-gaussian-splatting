"""FasterGS/Model.py"""

import math

import torch
import numpy as np
from plyfile import PlyData

import Framework
from Cameras.Perspective import PerspectiveCamera
from CudaUtils.MortonEncoding import morton_encode
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.Model import BaseModel
from Cameras.utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, invert_3d_affine
from Methods.FasterGS.FasterGSCudaBackend import FusedAdam, update_3d_filter, relocation_adjustment, add_noise
from Optim.adam_utils import replace_param_group_data, prune_param_groups, extend_param_groups, sort_param_groups, reset_state
from Optim.lr_utils import LRDecayPolicy
from Optim.knn_utils import compute_root_mean_squared_knn_distances
from Optim.ppisp import PPISPWrapper

# splat-transform / SuperSplat "compressed PLY": chunked quantized splats (see playcanvas/splat-transform decompress-ply.ts)
_COMPRESSED_PLY_CHUNK_SIZE = 256
_SH_C0 = 0.28209479177387814


class Gaussians(torch.nn.Module):
    """Stores a set of 3D Gaussians."""

    def __init__(self, sh_degree: int, pretrained: bool, deferred_reflection: bool = False, refl_init_value: float = 1e-3) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        # deferred reflection (3DGS-DR): per-Gaussian scalar reflection strength.
        # Gated so the base FasterGS path is unaffected when disabled.
        self.deferred_reflection = deferred_reflection
        self.refl_init_value = float(refl_init_value)
        self.register_parameter('_means', None)
        self.register_parameter('_sh_coefficients_0', None)
        self.register_parameter('_sh_coefficients_rest', None)
        self.register_parameter('_scales', None)
        self.register_parameter('_rotations', None)
        self.register_parameter('_opacities', None)
        self.register_parameter('_reflection_strength', None)
        self._densification_info = None
        self.optimizer = None
        self.percent_dense = 0.0
        self.training_cameras_extent = 1.0
        self._filter_3d = None
        self.use_original_3d_filter = False
        self.use_optimized_3d_filter = False
        self.distance2filter = 0
        self.lr_means = 0.0
        self.lr_means_scheduler = None

    @staticmethod
    def _get_ply_property_names(vertices) -> list[str]:
        return [prop.name for prop in vertices.properties]

    @staticmethod
    def _require_ply_properties(path: str, property_names: list[str], required: list[str]) -> None:
        missing = [name for name in required if name not in property_names]
        if missing:
            raise Framework.ModelError(f'invalid Gaussian PLY "{path}": missing required vertex properties {missing}')

    @staticmethod
    def _unpack_unorm_uint32(values: np.ndarray, bits: int) -> np.ndarray:
        t = float((1 << bits) - 1)
        return (values.astype(np.uint32) & ((1 << bits) - 1)).astype(np.float32) / t

    @classmethod
    def _is_compressed_splat_transform_ply(cls, plydata: PlyData) -> bool:
        """True if PLY matches splat-transform compressed schema (chunk + packed_vertex)."""
        if 'chunk' not in plydata or 'vertex' not in plydata:
            return False
        vertex_props = cls._get_ply_property_names(plydata['vertex'])
        required_v = ('packed_position', 'packed_rotation', 'packed_scale', 'packed_color')
        if not all(p in vertex_props for p in required_v):
            return False
        n_vertex = plydata['vertex'].count
        n_chunk = plydata['chunk'].count
        return int(np.ceil(n_vertex / _COMPRESSED_PLY_CHUNK_SIZE)) == n_chunk

    @classmethod
    def _decompress_splat_transform_ply(cls, plydata: PlyData, path: str, max_sh_degree: int) -> tuple[np.ndarray, ...]:
        """Decompress splat-transform compressed PLY to same arrays as standard Gaussian PLY."""
        chunk_el = plydata['chunk']
        vtx_el = plydata['vertex']

        min_x = np.asarray(chunk_el['min_x'], dtype=np.float32)
        min_y = np.asarray(chunk_el['min_y'], dtype=np.float32)
        min_z = np.asarray(chunk_el['min_z'], dtype=np.float32)
        max_x = np.asarray(chunk_el['max_x'], dtype=np.float32)
        max_y = np.asarray(chunk_el['max_y'], dtype=np.float32)
        max_z = np.asarray(chunk_el['max_z'], dtype=np.float32)
        min_scale_x = np.asarray(chunk_el['min_scale_x'], dtype=np.float32)
        min_scale_y = np.asarray(chunk_el['min_scale_y'], dtype=np.float32)
        min_scale_z = np.asarray(chunk_el['min_scale_z'], dtype=np.float32)
        max_scale_x = np.asarray(chunk_el['max_scale_x'], dtype=np.float32)
        max_scale_y = np.asarray(chunk_el['max_scale_y'], dtype=np.float32)
        max_scale_z = np.asarray(chunk_el['max_scale_z'], dtype=np.float32)

        prop_chunk = cls._get_ply_property_names(chunk_el)
        has_chunk_colors = 'min_r' in prop_chunk
        if has_chunk_colors:
            min_r = np.asarray(chunk_el['min_r'], dtype=np.float32)
            min_g = np.asarray(chunk_el['min_g'], dtype=np.float32)
            min_b = np.asarray(chunk_el['min_b'], dtype=np.float32)
            max_r = np.asarray(chunk_el['max_r'], dtype=np.float32)
            max_g = np.asarray(chunk_el['max_g'], dtype=np.float32)
            max_b = np.asarray(chunk_el['max_b'], dtype=np.float32)

        packed_position = np.asarray(vtx_el['packed_position']).astype(np.uint32)
        packed_rotation = np.asarray(vtx_el['packed_rotation']).astype(np.uint32)
        packed_scale = np.asarray(vtx_el['packed_scale']).astype(np.uint32)
        packed_color = np.asarray(vtx_el['packed_color']).astype(np.uint32)

        n = packed_position.shape[0]
        ci = np.minimum(np.arange(n, dtype=np.int64) // _COMPRESSED_PLY_CHUNK_SIZE, min_x.shape[0] - 1)

        px = cls._unpack_unorm_uint32(packed_position >> 21, 11)
        py = cls._unpack_unorm_uint32(packed_position >> 11, 10)
        pz = cls._unpack_unorm_uint32(packed_position, 11)
        means = np.empty((n, 3), dtype=np.float32)
        means[:, 0] = min_x[ci] * (1.0 - px) + max_x[ci] * px
        means[:, 1] = min_y[ci] * (1.0 - py) + max_y[ci] * py
        means[:, 2] = min_z[ci] * (1.0 - pz) + max_z[ci] * pz

        sx = cls._unpack_unorm_uint32(packed_scale >> 21, 11)
        sy = cls._unpack_unorm_uint32(packed_scale >> 11, 10)
        sz = cls._unpack_unorm_uint32(packed_scale, 11)
        scales_lin = np.empty((n, 3), dtype=np.float32)
        scales_lin[:, 0] = min_scale_x[ci] * (1.0 - sx) + max_scale_x[ci] * sx
        scales_lin[:, 1] = min_scale_y[ci] * (1.0 - sy) + max_scale_y[ci] * sy
        scales_lin[:, 2] = min_scale_z[ci] * (1.0 - sz) + max_scale_z[ci] * sz
        scales = np.log(np.maximum(scales_lin, 1e-12))

        cx = cls._unpack_unorm_uint32(packed_color >> 24, 8)
        cy = cls._unpack_unorm_uint32(packed_color >> 16, 8)
        cz = cls._unpack_unorm_uint32(packed_color >> 8, 8)
        cw = cls._unpack_unorm_uint32(packed_color, 8)
        if has_chunk_colors:
            cr = min_r[ci] * (1.0 - cx) + max_r[ci] * cx
            cg = min_g[ci] * (1.0 - cy) + max_g[ci] * cy
            cb = min_b[ci] * (1.0 - cz) + max_b[ci] * cz
        else:
            cr, cg, cb = cx, cy, cz
        sh0 = np.stack([(cr - 0.5) / _SH_C0, (cg - 0.5) / _SH_C0, (cb - 0.5) / _SH_C0], axis=1)[:, None, :]

        opacity_alpha = np.clip(cw, 1e-6, 1.0 - 1e-6)
        opacities = (np.log(opacity_alpha / (1.0 - opacity_alpha))).astype(np.float32)[:, None]

        norm = np.float32(np.sqrt(2.0))
        a = (cls._unpack_unorm_uint32(packed_rotation >> 20, 10) - 0.5) * norm
        b = (cls._unpack_unorm_uint32(packed_rotation >> 10, 10) - 0.5) * norm
        c = (cls._unpack_unorm_uint32(packed_rotation, 10) - 0.5) * norm
        m = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c))).astype(np.float32)
        which = packed_rotation >> 30
        r0 = np.where(which == 0, m, np.where(which == 1, a, np.where(which == 2, a, a)))
        r1 = np.where(which == 0, a, np.where(which == 1, m, np.where(which == 2, b, b)))
        r2 = np.where(which == 0, b, np.where(which == 1, b, np.where(which == 2, m, c)))
        r3 = np.where(which == 0, c, np.where(which == 1, c, np.where(which == 2, c, m)))
        rotations = np.stack([r0, r1, r2, r3], axis=1)

        expected_rest = (max_sh_degree + 1) ** 2 - 1
        sh_rest = np.zeros((n, expected_rest, 3), dtype=np.float32)
        if 'sh' in plydata:
            sh_el = plydata['sh']
            sh_rest_names = sorted(
                [p.name for p in sh_el.properties if p.name.startswith('f_rest_')],
                key=lambda x: int(x.split('_')[-1]),
            )
            n_sh_cols = len(sh_rest_names)
            if n_sh_cols > 0 and sh_el.count != n:
                raise Framework.ModelError(
                    f'compressed Gaussian PLY "{path}": vertex count {n} != sh count {sh_el.count}'
                )
            num_rest_in_file = n_sh_cols // 3
            if n_sh_cols % 3 != 0:
                raise Framework.ModelError(f'compressed Gaussian PLY "{path}": unexpected SH column count {n_sh_cols}')
            for k, name in enumerate(sh_rest_names):
                col = np.asarray(sh_el[name], dtype=np.uint8)
                n_lin = np.where(col == 0, 0.0, np.where(col == 255, 1.0, (col.astype(np.float32) + 0.5) / 256.0))
                decoded = ((n_lin - 0.5) * 8.0).astype(np.float32)
                basis = k % num_rest_in_file
                channel = k // num_rest_in_file
                if basis < expected_rest:
                    sh_rest[:, basis, channel] = decoded

        Logger.log_info(f'decompressed splat-transform compressed PLY ({n:,} splats)')
        return means, sh0, sh_rest, scales, rotations, opacities

    @property
    def means(self) -> torch.Tensor:
        """Returns the Gaussians' means (N, 3)."""
        return self._means

    @property
    def scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales (N, 3)."""
        scales = self._scales.exp()
        if self.use_original_3d_filter:
            scales = (scales.square() + self._filter_3d).sqrt()
        return scales

    @property
    def raw_scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales in logspace (N, 3)."""
        raw_scales = self._scales
        if self.use_original_3d_filter:
            scales = (raw_scales.exp().square() + self._filter_3d).sqrt()
            raw_scales = scales.log()
        return raw_scales

    @property
    def rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotations as quaternions (N, 4)."""
        return torch.nn.functional.normalize(self._rotations)

    @property
    def raw_rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotations as unnormalized quaternions (N, 4)."""
        return self._rotations

    @property
    def opacities(self) -> torch.Tensor:
        """Returns the Gaussians' opacities (N, 1)."""
        opacities = self._opacities.sigmoid()
        if self.use_original_3d_filter:
            scales_square = self._scales.exp().square()
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + self._filter_3d
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities = opacities * coef[..., None]
        return opacities

    @property
    def raw_opacities(self) -> torch.Tensor:
        """Returns the Gaussians' unactivated opacities (N, 1)."""
        raw_opacities = self._opacities
        if self.use_original_3d_filter:
            scales_square = self._scales.exp().square()
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + self._filter_3d
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities = raw_opacities.sigmoid() * coef[..., None]
            raw_opacities = opacities.logit(eps=1e-6)
        return raw_opacities

    @property
    def sh_coefficients(self) -> torch.Tensor:
        """Returns the Gaussians' SH coefficients for all bases (N, (max_degree + 1) ** 2, 3)."""
        return torch.cat([self._sh_coefficients_0, self._sh_coefficients_rest], dim=1)

    @property
    def sh_coefficients_0(self) -> torch.Tensor:
        """Returns the Gaussians' SH coefficients for the 0th, view-independent basis (N, 1, 3)."""
        return self._sh_coefficients_0

    @property
    def sh_coefficients_rest(self) -> torch.Tensor:
        """Returns the Gaussians' SH coefficients for all view-dependent bases (N, (max_degree + 1) ** 2 - 1, 3)."""
        return self._sh_coefficients_rest

    @property
    def densification_info(self) -> torch.Tensor:
        """Returns the current densification info buffers (2, N)."""
        return self._densification_info

    @property
    def covariances(self) -> torch.Tensor:
        """Returns the Gaussians' covariance matrices (N, 3, 3)."""
        R = quaternion_to_rotation_matrix(self.rotations, normalize=False)
        S = torch.diag_embed(self.scales)
        RS = R @ S
        return RS @ RS.transpose(-2, -1)

    @property
    def reflection_strength(self) -> torch.Tensor:
        """Returns the Gaussians' reflection strength in [0, 1] (N, 1)."""
        return self._reflection_strength.sigmoid()

    @property
    def raw_reflection_strength(self) -> torch.Tensor:
        """Returns the Gaussians' unactivated (logit) reflection strength (N, 1)."""
        return self._reflection_strength

    def _make_reflection_strength(self, n: int) -> torch.nn.Parameter:
        """Creates an initial reflection-strength parameter (logit of refl_init_value)."""
        v = min(max(self.refl_init_value, 1e-6), 1.0 - 1e-6)
        logit = math.log(v / (1.0 - v))
        values = torch.full((n, 1), fill_value=logit, dtype=torch.float32, device='cuda')
        return torch.nn.Parameter(values.contiguous())

    def min_axis_normals(self, camera_position: torch.Tensor) -> torch.Tensor:
        """Per-Gaussian normal = shortest ellipsoid axis, flipped to face the camera (N, 3).

        Matches 3DGS-DR ``get_min_axis``: unit axis from the rotation matrix, flipped
        when it points away from the camera. **Not** normalized here — the rasterizer
        blends raw normals and ``compose_deferred`` normalizes the per-pixel map before
        reflection (critical for correct cubemap queries).
        """
        R = quaternion_to_rotation_matrix(self.rotations, normalize=False)  # (N, 3, 3), columns are axes
        scales = self.scales  # (N, 3)
        min_idx = scales.argmin(dim=1)  # (N,) shortest axis
        n = R.shape[0]
        normals = R[torch.arange(n, device=R.device), :, min_idx]  # (N, 3)
        to_camera = camera_position.reshape(1, 3) - self._means  # (N, 3)
        flip = (normals * to_camera).sum(dim=-1, keepdim=True) < 0.0
        normals = torch.where(flip, -normals, normals)
        return normals

    def set_opacity_lr(self, lr: float) -> None:
        """3DGS-DR ``set_opacity_lr``: update the opacity optimizer group learning rate."""
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group['name'] == 'opacities':
                group['lr'] = lr

    @torch.no_grad()
    def dr_reset_opacity_floor(self, floor: float = 0.01) -> None:
        """3DGS-DR ``reset_opacity0``: clamp high opacities down to ``floor``."""
        floor_logit = math.log(floor / (1.0 - floor))
        below = self.opacities < floor
        new_opacities = torch.where(below, self._opacities, torch.full_like(self._opacities, floor_logit))
        replace_param_group_data(self.optimizer, new_opacities, 'opacities')
        self._opacities = self._get_param('opacities')

    @torch.no_grad()
    def dr_reset_opacity_ceiling(
        self,
        ceiling: float = 0.9,
        exclusive_msk: torch.Tensor | None = None,
    ) -> None:
        """3DGS-DR ``reset_opacity1``: raise low opacities up to ``ceiling``."""
        ceil_logit = math.log(ceiling / (1.0 - ceiling))
        skip = self.opacities.flatten() > ceiling
        if exclusive_msk is not None:
            skip = torch.logical_or(skip, exclusive_msk)
        new_opacities = torch.where(
            skip.reshape_as(self._opacities),
            self._opacities,
            torch.full_like(self._opacities, ceil_logit),
        )
        replace_param_group_data(self.optimizer, new_opacities, 'opacities')
        self._opacities = self._get_param('opacities')

    @torch.no_grad()
    def dr_bump_reflection_strength(
        self,
        min_reflection: float = 1e-3,
        exclusive_msk: torch.Tensor | None = None,
    ) -> None:
        """3DGS-DR ``reset_refl``: raise reflection strength to at least ``min_reflection``."""
        if self._reflection_strength is None:
            return
        min_refl_logit = math.log(min_reflection / (1.0 - min_reflection))
        new_reflection = torch.maximum(
            self._reflection_strength,
            torch.full_like(self._reflection_strength, min_refl_logit),
        )
        if exclusive_msk is not None:
            new_reflection[exclusive_msk] = self._reflection_strength[exclusive_msk]
        replace_param_group_data(self.optimizer, new_reflection, 'reflection_strength')
        self._reflection_strength = self._get_param('reflection_strength')

    @torch.no_grad()
    def dr_enlarge_reflective_scales(
        self,
        refl_threshold: float = 0.02,
        enlarge_scale: float = 1.5,
        exclusive_msk: torch.Tensor | None = None,
    ) -> None:
        """3DGS-DR ``reset_scale`` / ``enlarge_refl_scales``: scale the two longest axes."""
        if self._reflection_strength is None:
            return
        skip = self.reflection_strength.flatten() < refl_threshold
        if exclusive_msk is not None:
            skip = torch.logical_or(skip, exclusive_msk)
        if not (~skip).any():
            return
        scales = self._scales  # logspace (N, 3)
        min_idx = self.scales.argmin(dim=1)
        enlarge = torch.full_like(scales, fill_value=math.log(enlarge_scale))
        enlarge[torch.arange(scales.shape[0], device=scales.device), min_idx] = 0.0
        enlarge[skip] = 0.0
        new_scales = scales + enlarge
        replace_param_group_data(self.optimizer, new_scales, 'scales')
        self._scales = self._get_param('scales')

    @torch.no_grad()
    def normal_propagation(self, refl_threshold: float = 0.1, enlarge_scale: float = 1.5,
                           min_opacity: float = 0.9, min_reflection: float = 1e-3) -> None:
        """Legacy combined propagation step (opacity + reflection + scale). Prefer the
        split ``dr_*`` helpers for 3DGS-DR parity."""
        self.dr_reset_opacity_ceiling(min_opacity)
        self.dr_bump_reflection_strength(min_reflection)
        self.dr_enlarge_reflective_scales(refl_threshold=refl_threshold, enlarge_scale=enlarge_scale)

    @torch.no_grad()
    def color_sabotage(
        self,
        refl_threshold: float = 0.05,
        noise: float = 0.4,
        exclusive_msk: torch.Tensor | None = None,
    ) -> None:
        """3DGS-DR ``dist_color``: perturb diffuse SH of non-reflective Gaussians."""
        if self._reflection_strength is None:
            return
        skip = self.reflection_strength.flatten() > refl_threshold
        if exclusive_msk is not None:
            skip = torch.logical_or(skip, exclusive_msk)
        if not (~skip).any():
            return
        sh0 = self._sh_coefficients_0.clone()
        perturb = (torch.rand_like(sh0) * 2.0 - 1.0) * noise
        perturb[skip] = 0.0
        replace_param_group_data(self.optimizer, sh0 + perturb, 'sh_coefficients_0')
        self._sh_coefficients_0 = self._get_param('sh_coefficients_0')

    def n_reflective(self, refl_threshold: float = 0.1) -> int:
        """Number of Gaussians with reflection strength above the threshold."""
        if self._reflection_strength is None:
            return 0
        return int((self.reflection_strength.flatten() > refl_threshold).sum().item())

    def _get_param(self, name: str) -> torch.nn.Parameter:
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                return group['params'][0]
        raise KeyError(name)

    def opacity_regularization_loss(self) -> torch.Tensor:
        """Encourages the Gaussians' opacities to be small."""
        return self.opacities.mean()

    def scale_regularization_loss(self) -> torch.Tensor:
        """Encourages the Gaussians' scales to be small."""
        return self.scales.mean()

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    def setup_3d_filter(self, filter_config: Framework.ConfigParameterList, dataset: 'BaseDataset') -> None:
        """Sets up a 3D filter (see https://arxiv.org/abs/2311.16493)."""
        if filter_config.ORIGINAL_FORMULATION:
            self.use_original_3d_filter = True
            Logger.log_info(f'using mip-splatting 3d filter with variance {filter_config.FILTER_VARIANCE}')
        else:
            self.use_optimized_3d_filter = True
            Logger.log_info(f'using optimized 3d filter with variance {filter_config.FILTER_VARIANCE}')
        max_focal = 1e-12
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('update3dfilter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('update3dfilter ignores all distortion parameters')
            max_focal = max(max_focal, max(view.camera.focal_x, view.camera.focal_y))
        # assume max_focal is focal length of the highest resolution camera
        self.distance2filter = math.sqrt(filter_config.FILTER_VARIANCE) / max_focal
        self.compute_3d_filter(dataset)

    def compute_3d_filter(self, dataset: 'BaseDataset', clipping_tolerance: float = 0.15) -> None:
        """Computes the 3D filter."""
        positions = self.means
        filter_3d = torch.full((positions.shape[0], 1), fill_value=torch.finfo(torch.float32).max, device=positions.device, dtype=torch.float32)
        visibility_mask = torch.zeros((positions.shape[0], 1), device=positions.device, dtype=torch.bool)
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('update3dfilter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('update3dfilter ignores all distortion parameters')
            view_w2c = view.w2c
            # In turntable mode the dataset c2w is fixed; use original capture cameras
            # for visibility-derived filtering to avoid over/under-smoothing artifacts.
            original_c2w = view.exif.get('turntable_original_c2w')
            if original_c2w is not None:
                original_w2c = view.exif.get('turntable_original_w2c')
                if original_w2c is None:
                    original_w2c = invert_3d_affine(original_c2w)
                    view.exif['turntable_original_w2c'] = original_w2c
                view_w2c = torch.as_tensor(original_w2c, dtype=positions.dtype, device=positions.device)
            update_3d_filter(
                positions,
                view_w2c,
                filter_3d,
                visibility_mask,
                view.camera.width,
                view.camera.height,
                view.camera.focal_x,
                view.camera.focal_y,
                view.camera.center_x,
                view.camera.center_y,
                view.camera.near_plane,
                clipping_tolerance,
                self.distance2filter,
            )
        filter_3d_max = filter_3d[visibility_mask].max()
        filter_3d = torch.where(visibility_mask, filter_3d, filter_3d_max, out=filter_3d)
        if self.use_original_3d_filter:
            filter_3d = filter_3d.square()  # original implementation always needs this in squared form
        elif self.use_optimized_3d_filter:
            filter_3d = filter_3d.log()  # optimized implementation uses this to directly clamp scales in logspace
        self._filter_3d = filter_3d

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, use_mcmc: bool) -> None:
        """Initializes the model from a point cloud."""
        # initial means
        means = point_cloud.positions.cuda()
        n_initial_gaussians = means.shape[0]
        Logger.log_info(f'number of Gaussians at initialization: {n_initial_gaussians:,}')
        # initial sh coefficients
        rgbs = torch.full_like(means, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        sh_coefficients_0 = ((rgbs - 0.5) / 0.28209479177387814)[:, None, :]
        sh_coefficients_rest = torch.zeros((n_initial_gaussians, (self.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device='cuda')
        # initial scales
        distances = compute_root_mean_squared_knn_distances(means)
        distances = distances * 0.1 if use_mcmc else distances
        scales = distances.log()[..., None].repeat(1, 3)
        # initial rotations
        rotations = torch.zeros((n_initial_gaussians, 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1.0
        # initial opacities
        initial_opacity = 0.5 if use_mcmc else 0.1
        initial_opacity_logit = math.log(initial_opacity / (1.0 - initial_opacity))
        opacities = torch.full((n_initial_gaussians, 1), fill_value=initial_opacity_logit, dtype=torch.float32, device='cuda')
        # setup parameters
        self._means = torch.nn.Parameter(means.contiguous())
        self._sh_coefficients_0 = torch.nn.Parameter(sh_coefficients_0.contiguous())
        self._sh_coefficients_rest = torch.nn.Parameter(sh_coefficients_rest.contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())
        if self.deferred_reflection:
            self._reflection_strength = self._make_reflection_strength(n_initial_gaussians)

    def initialize_from_gaussian_ply(self, path: str) -> None:
        """Initializes the model from a Gaussian PLY with explicit Gaussian attributes."""
        try:
            plydata = PlyData.read(path)
        except Exception as exc:
            raise Framework.ModelError(f'failed to read Gaussian PLY "{path}": {exc}') from exc
        if 'vertex' not in plydata:
            raise Framework.ModelError(f'invalid Gaussian PLY "{path}": no "vertex" element')

        if self._is_compressed_splat_transform_ply(plydata):
            means, sh_coefficients_0, sh_coefficients_rest, scales, rotations, opacities = self._decompress_splat_transform_ply(
                plydata, path, self.max_sh_degree
            )
            n_initial_gaussians = means.shape[0]
            Logger.log_info(f'number of Gaussians loaded from PLY: {n_initial_gaussians:,}')
            self._means = torch.nn.Parameter(torch.from_numpy(means).cuda().contiguous())
            self._sh_coefficients_0 = torch.nn.Parameter(torch.from_numpy(sh_coefficients_0).cuda().contiguous())
            self._sh_coefficients_rest = torch.nn.Parameter(torch.from_numpy(sh_coefficients_rest).cuda().contiguous())
            self._scales = torch.nn.Parameter(torch.from_numpy(scales).cuda().contiguous())
            self._rotations = torch.nn.Parameter(torch.from_numpy(rotations).cuda().contiguous())
            self._opacities = torch.nn.Parameter(torch.from_numpy(opacities).cuda().contiguous())
            if self.deferred_reflection:
                self._reflection_strength = self._make_reflection_strength(n_initial_gaussians)
            return

        vertices = plydata['vertex']
        property_names = self._get_ply_property_names(vertices)
        self._require_ply_properties(path, property_names, ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])

        means = np.column_stack((vertices['x'], vertices['y'], vertices['z'])).astype(np.float32)
        sh0_names = ['f_dc_0', 'f_dc_1', 'f_dc_2']
        has_sh0 = all(name in property_names for name in sh0_names)
        if has_sh0:
            sh_coefficients_0 = np.column_stack((vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2'])).astype(np.float32)[:, None, :]
        else:
            sh_coefficients_0 = np.zeros((means.shape[0], 1, 3), dtype=np.float32)

        f_rest_names = sorted(
            [name for name in property_names if name.startswith('f_rest_')],
            key=lambda n: int(n.split('_')[-1]),
        )
        expected_rest = (self.max_sh_degree + 1) ** 2 - 1
        if len(f_rest_names) >= 3:
            f_rest = np.stack([vertices[name] for name in f_rest_names], axis=1).astype(np.float32)
            rest_bases_available = f_rest.shape[1] // 3
            rest_bases_used = min(rest_bases_available, expected_rest)
            sh_coefficients_rest = np.zeros((means.shape[0], expected_rest, 3), dtype=np.float32)
            sh_coefficients_rest[:, :rest_bases_used, :] = f_rest[:, :rest_bases_used * 3].reshape(means.shape[0], rest_bases_used, 3)
        else:
            sh_coefficients_rest = np.zeros((means.shape[0], expected_rest, 3), dtype=np.float32)

        scales = np.column_stack((vertices['scale_0'], vertices['scale_1'], vertices['scale_2'])).astype(np.float32)
        rotations = np.column_stack((vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3'])).astype(np.float32)
        opacities = np.asarray(vertices['opacity']).astype(np.float32)[:, None]

        n_initial_gaussians = means.shape[0]
        Logger.log_info(f'number of Gaussians loaded from PLY: {n_initial_gaussians:,}')

        self._means = torch.nn.Parameter(torch.from_numpy(means).cuda().contiguous())
        self._sh_coefficients_0 = torch.nn.Parameter(torch.from_numpy(sh_coefficients_0).cuda().contiguous())
        self._sh_coefficients_rest = torch.nn.Parameter(torch.from_numpy(sh_coefficients_rest).cuda().contiguous())
        self._scales = torch.nn.Parameter(torch.from_numpy(scales).cuda().contiguous())
        self._rotations = torch.nn.Parameter(torch.from_numpy(rotations).cuda().contiguous())
        self._opacities = torch.nn.Parameter(torch.from_numpy(opacities).cuda().contiguous())
        if self.deferred_reflection:
            self._reflection_strength = self._make_reflection_strength(n_initial_gaussians)

    @torch.no_grad()
    def reset_spherical_harmonics_to_rgb(self, rgb: torch.Tensor) -> None:
        """
        Replace all SH with a view-neutral appearance: degree-0 from linear RGB in [0, 1], higher degrees zero.

        Matches initialize_from_point_cloud: sh0 = (rgb - 0.5) / SH_C0 per channel.
        """
        if self._sh_coefficients_0 is None or self._sh_coefficients_rest is None:
            return
        rgb = rgb.flatten().to(dtype=torch.float32, device=self._sh_coefficients_0.device).clamp(0.0, 1.0)
        if rgb.numel() != 3:
            raise Framework.ModelError(f'reset_spherical_harmonics_to_rgb expects RGB of length 3, got shape {tuple(rgb.shape)}')
        n = self._sh_coefficients_0.shape[0]
        sh0 = ((rgb - 0.5) / _SH_C0).view(1, 1, 3).expand(n, 1, 3).contiguous()
        self._sh_coefficients_0.data.copy_(sh0)
        self._sh_coefficients_rest.data.zero_()

    @torch.no_grad()
    def apply_scene_alignment_transform(self, transform: np.ndarray) -> None:
        """
        Apply the same rigid + optional uniform scale as dataset PCA (see BasicPointCloud.transform).

        Means use the full linear part L = T[:3,:3]; rotations use orthogonal part R = L/s with
        s = ||L[:, 0]|| (uniform scale from APPLY_PCA_RESCALE); log-scales get +log(s).
        """
        T = torch.as_tensor(np.asarray(transform, dtype=np.float32), device=self._means.device)
        L = T[:3, :3]
        t = T[:3, 3]
        means = self._means.data
        means.copy_(means @ L.T + t)
        s = torch.linalg.norm(L[:, 0]).clamp_min(1e-12)
        R_align = L / s
        Rs = quaternion_to_rotation_matrix(self._rotations.data, normalize=True)
        R_out = torch.matmul(R_align.unsqueeze(0), Rs)
        q = rotation_matrix_to_quaternion(R_out)
        self._rotations.data.copy_(q)
        if (s - 1.0).abs() > 1e-6:
            self._scales.data.add_(torch.log(s))
        Logger.log_info('applied dataset scene_alignment_transform to Gaussian PLY init (means, rotations, scales)')

    def training_setup(self, training_wrapper, training_cameras_extent: float, freeze_geometry: bool = False) -> None:
        """Sets up the optimizer."""
        self.percent_dense = training_wrapper.DENSIFICATION_PERCENT_DENSE
        self.training_cameras_extent = training_cameras_extent

        param_groups = [
            {'params': [self._means], 'lr': 0.0 if freeze_geometry else training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_INIT * self.training_cameras_extent, 'name': 'means'},
            {'params': [self._sh_coefficients_0], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SH_COEFFICIENTS_0, 'name': 'sh_coefficients_0'},
            {'params': [self._sh_coefficients_rest], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SH_COEFFICIENTS_REST, 'name': 'sh_coefficients_rest'},
            {'params': [self._opacities], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_OPACITIES, 'name': 'opacities'},
            {'params': [self._scales], 'lr': 0.0 if freeze_geometry else training_wrapper.OPTIMIZER.LEARNING_RATE_SCALES, 'name': 'scales'},
            {'params': [self._rotations], 'lr': 0.0 if freeze_geometry else training_wrapper.OPTIMIZER.LEARNING_RATE_ROTATIONS, 'name': 'rotations'}
        ]

        if self.deferred_reflection:
            if self._reflection_strength is None:
                self._reflection_strength = self._make_reflection_strength(self._means.shape[0])
            param_groups.append({
                'params': [self._reflection_strength],
                'lr': getattr(training_wrapper.OPTIMIZER, 'LEARNING_RATE_REFLECTION_STRENGTH', 0.006),
                'name': 'reflection_strength',
            })

        self.optimizer = FusedAdam(param_groups, lr=0.0, eps=1e-15)

        if freeze_geometry:
            self.lr_means_scheduler = None
            self.lr_means = 0.0
        else:
            self.lr_means_scheduler = LRDecayPolicy(
                lr_init=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_INIT * self.training_cameras_extent,
                lr_final=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_FINAL * self.training_cameras_extent,
                max_steps=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_MAX_STEPS
            )

    def update_learning_rate(self, iteration: int) -> None:
        """Computes the current learning rate for the given iteration."""
        if self.lr_means_scheduler is None:
            return
        self.lr_means = self.lr_means_scheduler(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'means':
                param_group['lr'] = self.lr_means

    def reset_opacities(self) -> None:
        """Resets the opacities to a fixed value."""
        opacities_new = self._opacities.clamp_max(-4.595119953155518)  # sigmoid(-4.595119953155518) = 0.01
        if self.use_original_3d_filter:
            # make sure that the current 3d filter has the same effect on the new opacities
            scales_square = self._scales.exp().square()
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + self._filter_3d
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities_new = (opacities_new.sigmoid() / coef[..., None]).logit(eps=1e-6)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def prune(self, prune_mask: torch.Tensor) -> None:
        """Prunes Gaussians that are not visible or too large."""
        valid_mask = ~prune_mask
        if not torch.any(valid_mask):
            Logger.log_warning('pruning would remove all Gaussians; skipping prune step')
            return
        param_groups = prune_param_groups(self.optimizer, valid_mask)

        self._means = param_groups['means']
        self._sh_coefficients_0 = param_groups['sh_coefficients_0']
        self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
        self._opacities = param_groups['opacities']
        self._scales = param_groups['scales']
        self._rotations = param_groups['rotations']
        if 'reflection_strength' in param_groups:
            self._reflection_strength = param_groups['reflection_strength']

        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, valid_mask].contiguous()
        if self._filter_3d is not None:
            self._filter_3d = self._filter_3d[valid_mask].contiguous()

    def sort(self, ordering: torch.Tensor) -> None:
        """Applies the given ordering to the Gaussians."""
        param_groups = sort_param_groups(self.optimizer, ordering)

        self._means = param_groups['means']
        self._sh_coefficients_0 = param_groups['sh_coefficients_0']
        self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
        self._opacities = param_groups['opacities']
        self._scales = param_groups['scales']
        self._rotations = param_groups['rotations']
        if 'reflection_strength' in param_groups:
            self._reflection_strength = param_groups['reflection_strength']

        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, ordering].contiguous()
        if self._filter_3d is not None:
            self._filter_3d = self._filter_3d[ordering].contiguous()

    def reset_densification_info(self):
        self._densification_info = torch.zeros((2, self._means.shape[0]), dtype=torch.float32, device='cuda')

    def adaptive_density_control(self, grad_threshold: float, min_opacity: float, prune_large_gaussians: bool) -> None:
        """Densify Gaussians and prune those that are not visible or too large."""
        densification_mask = self.densification_info[1] >= grad_threshold * self.densification_info[0].clamp_min(1.0)
        is_small = torch.max(self._scales, dim=1).values <= math.log(self.percent_dense * self.training_cameras_extent)

        # duplicate small gaussians
        duplicate_mask = densification_mask & is_small
        n_new_gaussians_duplicate = duplicate_mask.sum().item()
        duplicated_means = self._means[duplicate_mask]
        duplicated_sh_coefficients_0 = self._sh_coefficients_0[duplicate_mask]
        duplicated_sh_coefficients_rest = self._sh_coefficients_rest[duplicate_mask]
        duplicated_opacities = self._opacities[duplicate_mask]
        duplicated_scales = self._scales[duplicate_mask]
        duplicated_rotations = self._rotations[duplicate_mask]
        if self._reflection_strength is not None:
            duplicated_reflection = self._reflection_strength[duplicate_mask]

        # split large gaussians
        split_mask = densification_mask & ~is_small
        n_new_gaussians_split = 2 * split_mask.sum().item()
        split_scales = self._scales[split_mask].exp().expand(2, -1, -1).flatten(end_dim=1)
        split_rotations = self._rotations[split_mask].expand(2, -1, -1).flatten(end_dim=1)
        offsets = (quaternion_to_rotation_matrix(split_rotations) @ (split_scales * torch.randn_like(split_scales))[..., None])[..., 0]
        split_means = self._means[split_mask].expand(2, -1, -1).flatten(end_dim=1) + offsets
        split_scales = split_scales.mul(0.625).log()  # 1 / 1.6 = 0.625
        split_sh_coefficients_0 = self._sh_coefficients_0[split_mask].expand(2, -1, -1, -1).flatten(end_dim=1)
        split_sh_coefficients_rest = self._sh_coefficients_rest[split_mask].expand(2, -1, -1, -1).flatten(end_dim=1)
        split_opacities = self._opacities[split_mask].expand(2, -1, -1).flatten(end_dim=1)
        if self._reflection_strength is not None:
            split_reflection = self._reflection_strength[split_mask].expand(2, -1, -1).flatten(end_dim=1)

        # incorporate
        n_new_gaussians = n_new_gaussians_duplicate + n_new_gaussians_split
        extension = {
            'means': torch.cat([duplicated_means, split_means]),
            'sh_coefficients_0': torch.cat([duplicated_sh_coefficients_0, split_sh_coefficients_0]),
            'sh_coefficients_rest': torch.cat([duplicated_sh_coefficients_rest, split_sh_coefficients_rest]),
            'opacities': torch.cat([duplicated_opacities, split_opacities]),
            'scales': torch.cat([duplicated_scales, split_scales]),
            'rotations': torch.cat([duplicated_rotations, split_rotations])
        }
        if self._reflection_strength is not None:
            extension['reflection_strength'] = torch.cat([duplicated_reflection, split_reflection])
        param_groups = extend_param_groups(self.optimizer, extension)
        self._means = param_groups['means']
        self._sh_coefficients_0 = param_groups['sh_coefficients_0']
        self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
        self._opacities = param_groups['opacities']
        self._scales = param_groups['scales']
        self._rotations = param_groups['rotations']
        if 'reflection_strength' in param_groups:
            self._reflection_strength = param_groups['reflection_strength']

        # if they were set, densification info and 3d filter are now no longer valid
        self._densification_info = None
        self._filter_3d = None

        # prune
        prune_mask = torch.cat([split_mask, torch.zeros(n_new_gaussians, dtype=torch.bool, device='cuda')])
        prune_mask |= self._opacities.flatten() < math.log(min_opacity / (1 - min_opacity))
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        if prune_large_gaussians:
            prune_mask |= self._scales.max(dim=1).values > math.log(0.1 * self.training_cameras_extent)
        self.prune(prune_mask)

    def mcmc_densification(self, min_opacity: float, cap_max: int) -> None:
        """Relocates low-opacity/degenerate Gaussians and adds new ones up to a cap."""
        # relocate
        dead_mask = self._opacities.flatten() <= math.log(min_opacity / (1 - min_opacity))
        dead_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        n_dead_gaussians = dead_mask.sum().item()
        if n_dead_gaussians > 0:
            # sample existing Gaussians to copy to the dead ones, with probability proportional to opacity
            dead_indices = torch.where(dead_mask)[0]
            alive_indices = torch.where(~dead_mask)[0]
            opacities = self.opacities.flatten()
            sampled_indices = torch.multinomial(opacities[alive_indices], n_dead_gaussians, replacement=True)
            sampled_indices = alive_indices[sampled_indices]

            # compute the adjusted opacities and scales
            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1  # +1 for the original Gaussian
            adjusted_opacities, adjusted_scales = relocation_adjustment(
                opacities[sampled_indices],
                self._scales[sampled_indices].exp(),
                counts,
            )
            adjusted_opacities = adjusted_opacities.clamp(min_opacity, 1.0 - torch.finfo(torch.float32).eps).logit()
            adjusted_scales = adjusted_scales.log()

            # update existing sampled Gaussians
            self._opacities[sampled_indices] = adjusted_opacities
            self._scales[sampled_indices] = adjusted_scales

            # copy sampled Gaussians to the dead ones
            self._means[dead_indices] = self._means[sampled_indices]
            self._sh_coefficients_0[dead_indices] = self._sh_coefficients_0[sampled_indices]
            self._sh_coefficients_rest[dead_indices] = self._sh_coefficients_rest[sampled_indices]
            self._opacities[dead_indices] = adjusted_opacities
            self._scales[dead_indices] = adjusted_scales
            self._rotations[dead_indices] = self._rotations[sampled_indices]
            if self._reflection_strength is not None:
                self._reflection_strength[dead_indices] = self._reflection_strength[sampled_indices]

            # reset optimizer state for the sampled Gaussians
            reset_state(self.optimizer, indices=sampled_indices)

            # if they were set, densification info and 3d filter are now no longer valid
            self._densification_info = None
            self._filter_3d = None

        # add new Gaussians
        current_n_points = self._means.shape[0]
        n_target = min(cap_max, int(1.05 * current_n_points))
        n_added_gaussians = max(0, n_target - current_n_points)
        if n_added_gaussians > 0:
            # sample existing Gaussians to duplicate, with probability proportional to opacity
            opacities = self.opacities.flatten()
            sampled_indices = torch.multinomial(opacities, n_added_gaussians, replacement=True)

            # compute the adjusted opacities and scales
            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1  # +1 for the original Gaussian
            adjusted_opacities, adjusted_scales = relocation_adjustment(
                opacities[sampled_indices],
                self._scales[sampled_indices].exp(),
                counts,
            )
            adjusted_opacities = adjusted_opacities.clamp(min_opacity, 1.0 - torch.finfo(torch.float32).eps).logit()
            adjusted_scales = adjusted_scales.log()

            # update existing sampled Gaussians
            self._opacities[sampled_indices] = adjusted_opacities
            self._scales[sampled_indices] = adjusted_scales

            # add new Gaussians by duplicating the sampled ones
            extension = {
                'means': self._means[sampled_indices],
                'sh_coefficients_0': self._sh_coefficients_0[sampled_indices],
                'sh_coefficients_rest': self._sh_coefficients_rest[sampled_indices],
                'opacities': adjusted_opacities,
                'scales': adjusted_scales,
                'rotations': self._rotations[sampled_indices],
            }
            if self._reflection_strength is not None:
                extension['reflection_strength'] = self._reflection_strength[sampled_indices]
            param_groups = extend_param_groups(self.optimizer, extension)
            self._means = param_groups['means']
            self._sh_coefficients_0 = param_groups['sh_coefficients_0']
            self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
            self._opacities = param_groups['opacities']
            self._scales = param_groups['scales']
            self._rotations = param_groups['rotations']
            if 'reflection_strength' in param_groups:
                self._reflection_strength = param_groups['reflection_strength']

            # reset optimizer state for the sampled Gaussians
            reset_state(self.optimizer, indices=sampled_indices)

            # if they were set, densification info and 3d filter are now no longer valid
            self._densification_info = None
            self._filter_3d = None

    def apply_morton_ordering(self) -> None:
        """Applies Morton ordering to the Gaussians."""
        if self._means is None or self._means.shape[0] == 0:
            return
        morton_encoding = morton_encode(self._means.data)
        order = torch.argsort(morton_encoding)
        self.sort(order)

    def importance_pruning(self, scores: torch.Tensor, pruning_ratio: float) -> None:
        """Prunes the given percentage of Gaussians with the lowest importance score (from Speedy-Splat)."""
        k = int(pruning_ratio * (scores.numel() - 1)) + 1  # kthvalue is 1-based
        threshold = torch.kthvalue(scores, k).values
        prune_mask = scores <= threshold
        self.prune(prune_mask)

    @torch.no_grad()
    def post_optimizer_step(self, inject_noise: bool) -> None:
        """Applies modifications to the Gaussians after every optimizer step."""
        if inject_noise:
            add_noise(self.raw_scales, self.raw_rotations, self.raw_opacities, self.means, 5e5 * self.lr_means)
        if self.use_optimized_3d_filter:
            self._scales.clamp_min_(self._filter_3d)

    @torch.no_grad()
    def training_cleanup(self, min_opacity: float) -> int:
        """Cleans the model after training."""
        # bake 3d filter if used
        if self.use_optimized_3d_filter:
            # nothing to do, already baked in
            self.use_optimized_3d_filter = False
        elif self.use_original_3d_filter:
            # the 3d filter must be baked into the opacities before the scales to get the correct result
            self._opacities.data = self.raw_opacities
            self._scales.data = self.raw_scales
            self.use_original_3d_filter = False
        self._filter_3d = None

        # densification info no longer needed
        self._densification_info = None

        # prune low-opacity and degenerate Gaussians
        prune_mask = self.opacities.flatten() < min_opacity
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        self.prune(prune_mask)

        # sort by morton code
        self.apply_morton_ordering()

        # clear any leftover gradients and delete optimizer
        self.optimizer.zero_grad()
        self.optimizer = None

        return self.means.shape[0]

    @torch.no_grad()
    def as_ply_dict(self) -> dict[str, np.ndarray]:
        """Returns the model as a ply-compatible dictionary using structured numpy arrays."""
        if self.means.shape[0] == 0:
            return {}

        # construct attributes
        means = self.means.detach().contiguous().cpu().numpy()
        sh_0 = self.sh_coefficients_0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self.sh_coefficients_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.raw_opacities.detach().contiguous().cpu().numpy()  # most viewers expect unactivated opacities
        scales = self.raw_scales.detach().contiguous().cpu().numpy()  # most viewers expect unactivated scales
        rotations = self.rotations.detach().contiguous().cpu().numpy()
        attributes = np.concatenate((means, sh_0, sh_rest, opacities, scales, rotations), axis=1)

        # construct structured array
        attribute_names = (
              ['x', 'y', 'z']                                    # 3d mean
            + ['f_dc_0', 'f_dc_1', 'f_dc_2']                     # 0-th SH degree coefficients
            + [f'f_rest_{i}' for i in range(sh_rest.shape[-1])]  # remaining SH degree coefficients
            + ['opacity']                                        # opacity (pre-activation)
            + ['scale_0', 'scale_1', 'scale_2']                  # 3d scale (pre-activation)
            + ['rot_0', 'rot_1', 'rot_2', 'rot_3']               # rotation quaternion
        )
        dtype = 'f4'  # store all attributes as float32 for compatibility
        full_dtype = [(attribute_name, dtype) for attribute_name in attribute_names]
        vertices = np.empty(means.shape[0], dtype=full_dtype)

        # insert attributes into structured array
        vertices[:] = list(map(tuple, attributes))

        return {'vertex': vertices}


@Framework.Configurable.configure(
    SH_DEGREE=3,
    PPISP=Framework.ConfigParameterList(
        USE=False,
        CONTROLLER_TRAINING_STEPS=5_000,
        CONTROLLER_DISTILLATION=True,
    ),
    DEFERRED_REFLECTION=Framework.ConfigParameterList(
        USE=False,
        REFL_INIT_VALUE=1e-3,
        ENVMAP_RESOLUTION=256,
        MESH_NORMAL_HIJACK=False,
        FORCE_ALBEDO_BASE_COLOR=False,
        ENVMAP_HDRI=None,
        ENVMAP_EXPOSURE=1.0,
        ENVMAP_YAW_DEG=0.0,
        ENVMAP_ROLL_DEG=0.0,
        FREEZE_ENVMAP=False,
        ENVMAP_ANCHOR_LAMBDA=0.1,
    ),
)
class FasterGSModel(BaseModel):
    """Defines the FasterGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None
        self.ppisp: PPISPWrapper | None = None
        self.environment = None

    def build(self) -> 'FasterGSModel':
        """Builds the model."""
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians(
            self.SH_DEGREE, pretrained,
            deferred_reflection=self.DEFERRED_REFLECTION.USE,
            refl_init_value=self.DEFERRED_REFLECTION.REFL_INIT_VALUE,
        )
        if self.DEFERRED_REFLECTION.USE:
            from Methods.FasterGS.DeferredShading import EnvironmentMap
            self.environment = EnvironmentMap(resolution=self.DEFERRED_REFLECTION.ENVMAP_RESOLUTION).cuda()
        if self.PPISP.USE:
            self.ppisp = PPISPWrapper(self.PPISP)
        return self

    def get_ply_dict(self) -> dict[str, np.ndarray | list[str]]:
        """Returns the model as a ply-compatible dictionary using structured numpy arrays."""
        data: dict[str, np.ndarray | list[str]] = {}
        if self.gaussians is None or not (data := self.gaussians.as_ply_dict()):
            return data

        # add method-specific comments
        splat_render_mode = 'default' #'mip-0.1' if Framework.config.RENDERER.PROPER_ANTIALIASING else 'default'
        data['comments'] = [f'SplatRenderMode: {splat_render_mode}', 'Generated with NeRFICG/FasterGS']

        return data
