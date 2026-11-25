"""Geometry losses adapted from MIT-licensed NeRF-ICG revision 57ddd49."""
from Methods.FasterGS.Loss import FasterGSLoss


def distortion_loss(distortion):
    return distortion.mean()


def normal_loss(rendered, surface):
    return (1-(rendered*surface).sum(0)).mean()


class FasterGS2DLoss(FasterGSLoss):
    def __init__(self, loss_config, model):
        super().__init__(loss_config, model)
        self.base_distortion = float(loss_config.LAMBDA_DISTORTION)
        self.base_normal = float(loss_config.LAMBDA_NORMAL)
        self.add_loss_metric("DISTORTION_REGULARIZATION", distortion_loss, self.base_distortion)
        self.add_loss_metric("NORMAL_REGULARIZATION", normal_loss, self.base_normal)

    def set_geometry_loss_weights(self, *, distortion_scale, normal_scale):
        self.update_loss_weight("DISTORTION_REGULARIZATION", self.base_distortion*distortion_scale)
        self.update_loss_weight("NORMAL_REGULARIZATION", self.base_normal*normal_scale)

    def forward(self, render_pkg, target):
        return super().forward(
            render_pkg["rgb"], target,
            DISTORTION_REGULARIZATION={"distortion": render_pkg["rend_dist"]},
            NORMAL_REGULARIZATION={"rendered": render_pkg["rend_normal"], "surface": render_pkg["surf_normal"]},
        )
