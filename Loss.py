"""FasterGS/Loss.py"""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim
from Methods.FasterGS.Model import FasterGSModel


def normal_map_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Cosine loss between predicted and GT normal maps. Tensors are (3, H, W)."""
    pred = pred / (torch.norm(pred, dim=0, keepdim=True) + 1e-6)
    gt = gt / (torch.norm(gt, dim=0, keepdim=True) + 1e-6)
    loss_map = 1.0 - (pred * gt).sum(dim=0)
    if mask is None:
        return loss_map.mean()
    weight = mask.squeeze(0) if mask.dim() == 3 else mask
    return (loss_map * weight.float()).sum() / weight.sum().clamp(min=1.0)


class FasterGSLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList, model: FasterGSModel) -> None:
        super().__init__()
        self._lambda_normal = float(getattr(loss_config, 'LAMBDA_NORMAL', 0.0))
        self._lambda_albedo_prior = float(getattr(loss_config, 'LAMBDA_ALBEDO_PRIOR', 0.0))
        self._lambda_refl_prior = float(getattr(loss_config, 'LAMBDA_REFL_PRIOR', 0.0))
        self._refl_prior_scale = float(getattr(loss_config, 'REFL_PRIOR_SCALE', 0.4))
        self.add_loss_metric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric('OPACITY_REGULARIZATION', model.gaussians.opacity_regularization_loss, loss_config.LAMBDA_OPACITY_REGULARIZATION)
        self.add_loss_metric('SCALE_REGULARIZATION', model.gaussians.scale_regularization_loss, loss_config.LAMBDA_SCALE_REGULARIZATION)
        if model.ppisp is None:
            self.add_loss_metric('PPISP_REGULARIZATION', lambda: 0.0, 0.0)
        else:
            self.add_loss_metric('PPISP_REGULARIZATION', model.ppisp.model.get_regularization_loss, 1.0)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward({
            'L1_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'OPACITY_REGULARIZATION': {},
            'SCALE_REGULARIZATION': {},
            'PPISP_REGULARIZATION': {},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })

    def mesh_normal_supervision_loss(
        self,
        gaussian_normal: torch.Tensor,
        mesh_normal: torch.Tensor,
        mesh_normal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._lambda_normal <= 0.0:
            return gaussian_normal.new_zeros(())
        return self._lambda_normal * normal_map_loss(gaussian_normal, mesh_normal, mesh_normal_mask)

    def reflection_strength_prior_loss(
        self,
        pred_refl: torch.Tensor,
        target_metallic: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._lambda_refl_prior <= 0.0:
            return pred_refl.new_zeros(())
        target = target_metallic * self._refl_prior_scale
        if target_mask is None:
            return self._lambda_refl_prior * torch.nn.functional.l1_loss(pred_refl, target)
        weight = target_mask.squeeze(0) if target_mask.dim() == 3 else target_mask
        return self._lambda_refl_prior * (
            (torch.abs(pred_refl.squeeze(0) - target.squeeze(0)) * weight.float()).sum()
            / weight.sum().clamp(min=1.0)
        )

    def albedo_prior_loss(
        self,
        pred_albedo: torch.Tensor,
        target_albedo: torch.Tensor,
        albedo_mask: torch.Tensor | None,
        metallic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._lambda_albedo_prior <= 0.0:
            return pred_albedo.new_zeros(())
        weight = albedo_mask
        if weight is None:
            weight = torch.ones_like(pred_albedo[:1])
        elif weight.dim() == 2:
            weight = weight.unsqueeze(0)
        if metallic is not None:
            weight = weight * (1.0 - metallic)
        denom = weight.sum() * pred_albedo.shape[0] + 1e-6
        return self._lambda_albedo_prior * (
            (torch.abs(pred_albedo - target_albedo) * weight.float()).sum() / denom
        )
