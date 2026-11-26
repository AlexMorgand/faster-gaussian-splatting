"""Geometry losses adapted from MIT-licensed NeRF-ICG revision 57ddd49."""
from Methods.FasterGS.Loss import FasterGSLoss


def distortion_loss(distortion):
    return distortion.mean()


def normal_loss(rendered, surface):
    return (1-(rendered*surface).sum(0)).mean()


def per_disk_normal_loss(alpha, weighted_disk_normal, unit_surface_normal):
    """Eq. 14 summed along a ray: sum(w_i) - dot(sum(w_i * n_i), N)."""
    return (alpha - (weighted_disk_normal * unit_surface_normal).sum(0, keepdim=True)).mean()


class FasterGS2DLoss(FasterGSLoss):
    normal_loss_mode = "normalized"  # Also applies to older serialized trainers.

    def __init__(self, loss_config, model):
        super().__init__(loss_config, model)
        self.base_distortion = float(loss_config.LAMBDA_DISTORTION)
        self.base_normal = float(loss_config.LAMBDA_NORMAL)
        self.add_loss_metric("DISTORTION_REGULARIZATION", distortion_loss, self.base_distortion)
        self.normal_loss_mode = loss_config.get("NORMAL_LOSS_MODE", "normalized")
        if self.normal_loss_mode not in ("normalized", "per_disk"):
            raise ValueError("NORMAL_LOSS_MODE must be normalized or per_disk")
        metric = per_disk_normal_loss if self.normal_loss_mode == "per_disk" else normal_loss
        self.add_loss_metric("NORMAL_REGULARIZATION", metric, self.base_normal)

    def set_geometry_loss_weights(self, *, distortion_scale, normal_scale):
        self.update_loss_weight("DISTORTION_REGULARIZATION", self.base_distortion*distortion_scale)
        self.update_loss_weight("NORMAL_REGULARIZATION", self.base_normal*normal_scale)

    def forward(self, render_pkg, target):
        if self.normal_loss_mode == "per_disk":
            normals = dict(alpha=render_pkg["rend_alpha"], weighted_disk_normal=render_pkg["weighted_disk_normal"],
                           unit_surface_normal=render_pkg["unit_surface_normal"])
        else:
            normals = dict(rendered=render_pkg["rend_normal"], surface=render_pkg["surf_normal"])
        return super().forward(
            render_pkg["rgb"], target,
            DISTORTION_REGULARIZATION={"distortion": render_pkg["rend_dist"]},
            NORMAL_REGULARIZATION=normals,
        )
