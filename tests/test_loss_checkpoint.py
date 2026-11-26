"""From NeRFICG: PYTHONPATH=src python -m unittest discover -s src/Methods/FasterGS/tests."""
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import Framework
from Methods.FasterGS.Loss import FasterGSLoss
from Methods.FasterGS.Model import Gaussians


class LossCheckpointTest(unittest.TestCase):
    def test_disabled_ppisp_roundtrip(self):
        config = Framework.ConfigParameterList(
            TRAINING=Framework.ConfigParameterList(
                WANDB=Framework.ConfigParameterList(ACTIVATE=False),
            ),
        )
        loss_config = Framework.ConfigParameterList(
            LAMBDA_L1=0.8, LAMBDA_DSSIM=0.0,
            LAMBDA_OPACITY_REGULARIZATION=0.0, LAMBDA_SCALE_REGULARIZATION=0.0,
        )
        model = SimpleNamespace(gaussians=Gaussians(sh_degree=3, pretrained=False), ppisp=None)
        with patch.object(Framework, "config", config, create=True):
            loss = FasterGSLoss(loss_config, model)
        checkpoint = io.BytesIO()
        torch.save(loss, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=False)
        zero_metric = next(item for item in restored.loss_metrics if item.name == "PPISP_REGULARIZATION")
        self.assertEqual(zero_metric.weight, 0.0)
        self.assertEqual(zero_metric.metric_func(), 0.0)
        for objective in (loss, restored):
            image = torch.tensor([0.25, 0.75], requires_grad=True)
            value = objective(image, torch.zeros_like(image))
            torch.testing.assert_close(value, torch.tensor(0.4))
            value.backward()
            torch.testing.assert_close(image.grad, torch.full_like(image, 0.4))


if __name__ == "__main__":
    unittest.main()
