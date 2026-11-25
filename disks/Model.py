"""Two-axis Gaussian model.

Adapted from MIT-licensed NeRF-ICG revision 57ddd49.
Copyright (c) 2026 Moritz Kappel, Florian Hahlbohm, Timon Scholz. See LICENSE.
"""
from copy import deepcopy

import torch

import Framework
from Methods.FasterGS.Model import Gaussians, FasterGSModel
from Optim.knn_utils import compute_knn_distances


class Gaussians2D(Gaussians):
    scale_dim = 2
    _max_radii2D = None

    @staticmethod
    def initial_distances(points):
        distances = compute_knn_distances(points, n_neighbors=3)
        return distances.square().mean(dim=-1).clamp_min(1e-7).sqrt()

    def get_extra_state(self):
        degree, bases = self.active_sh_degree, self.active_sh_bases
        if (type(degree) is not int or type(bases) is not int
            or not 0 <= degree <= self.max_sh_degree or bases != (degree+1)**2):
            raise Framework.ModelError("FasterGS2D invalid active SH degree/bases")
        return {"version": 1, "active_sh_degree": degree, "active_sh_bases": bases}

    def set_extra_state(self, state):
        if (not isinstance(state, dict)
            or set(state) != {"version", "active_sh_degree", "active_sh_bases"}
            or type(state["version"]) is not int or state["version"] != 1
            or type(state["active_sh_degree"]) is not int or type(state["active_sh_bases"]) is not int
            or not 0 <= state["active_sh_degree"] <= self.max_sh_degree
            or state["active_sh_bases"] != (state["active_sh_degree"]+1)**2):
            raise Framework.ModelError("FasterGS2D invalid saved SH execution state")
        self.active_sh_degree = state["active_sh_degree"]
        self.active_sh_bases = state["active_sh_bases"]

    @torch.no_grad()
    def initialize_from_point_cloud(self, point_cloud, use_mcmc=False):
        if use_mcmc:
            raise Framework.ModelError("FasterGS2D does not support MCMC")
        super().initialize_from_point_cloud(point_cloud, False)
        self._rotations.uniform_()
        self._max_radii2D = self._means.new_zeros(len(self._means))

    def sort(self, ordering):
        super().sort(ordering)
        if self._max_radii2D is not None:
            self._max_radii2D = self._max_radii2D[ordering]

    def prune(self, prune_mask):
        super().prune(prune_mask)
        if self._max_radii2D is not None:
            radii = self._max_radii2D
            # Newly cloned or split disks have no observations yet.
            new_disks = len(prune_mask) - len(radii)
            if new_disks:
                radii = torch.cat((radii, radii.new_zeros(new_disks)))
            self._max_radii2D = radii[~prune_mask]

    def reset_densification_info(self, use_abs_grad=False):
        if use_abs_grad:
            raise Framework.ModelError("FasterGS2D ABSgrad requires an independent per-pixel accumulator")
        super().reset_densification_info(False)
        self._max_radii2D = self._means.new_zeros(len(self._means))

    @torch.no_grad()
    def accumulate_image_shift(self, shift_grad, metadata, width, height):
        """Average the accepted-hit gradient over views where the disk has support."""
        if (self._densification_info is None or shift_grad is None
            or shift_grad.shape != (len(self._means), 2) or metadata.shape != (2, len(self._means))):
            raise Framework.ModelError("FasterGS2D requires Nx2 pixel-shift gradients and 2xN visibility/radius metadata")
        visible = metadata[1] > 0
        if not torch.isfinite(shift_grad).all() or not torch.isfinite(metadata[:, visible]).all():
            raise Framework.ModelError("FasterGS2D nonfinite densification state")
        ndc_grad = shift_grad.detach()*shift_grad.new_tensor([width/2, height/2])
        score = ndc_grad.norm(dim=1)
        if not torch.isfinite(score[visible]).all():
            raise Framework.ModelError("FasterGS2D nonfinite NDC densification score")
        self._densification_info[0].add_(visible | (metadata[0] > 0))
        self._densification_info[1, visible] += score[visible]
        self._max_radii2D[visible] = torch.maximum(self._max_radii2D[visible], metadata[0, visible])

    def setup_3d_filter(self, *args, **kwargs):
        raise Framework.ModelError("FasterGS2D does not support 3D filters")

    compute_3d_filter = setup_3d_filter

    def mcmc_densification(self, *args, **kwargs):
        raise Framework.ModelError("FasterGS2D does not support MCMC")

    def post_optimizer_step(self, inject_noise=False):
        if inject_noise:
            raise Framework.ModelError("FasterGS2D does not support MCMC noise")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        scales = state_dict.get(prefix+"_scales")
        if scales is not None and (scales.ndim != 2 or scales.shape[1] != 2):
            raise Framework.ModelError("FasterGS2D checkpoint requires Nx2 scales; no 3D conversion")
        if prefix+"_extra_state" not in state_dict:
            raise Framework.ModelError("FasterGS2D checkpoint missing SH execution state; explicit verified migration required")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class FasterGS2DModel(FasterGSModel):
    # Isolate nested experimental configuration from the inherited 3D defaults.
    _configuration = deepcopy(FasterGSModel.get_default_parameters())
    gaussian_type = Gaussians2D

    def build(self):
        if self.PPISP.USE:
            raise Framework.ModelError("FasterGS2D experimental path does not support PPISP")
        if type(self.SH_DEGREE) is not int or not 0 <= self.SH_DEGREE <= 3:
            raise Framework.ModelError("FasterGS2D supports SH degree 0 through 3")
        return super().build()

    def get_ply_dict(self):
        data = super().get_ply_dict()
        if data:
            data["comments"].append("Experimental FasterGS2D: Nx2 perspective disks, not 3D splats")
        return data
