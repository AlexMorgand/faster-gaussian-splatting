"""Run from NeRFICG: PYTHONPATH=src python -m unittest discover -s src/Methods/FasterGS/tests."""
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import Framework
from Methods.FasterGS.Model import FasterGSModel, Gaussians
from Methods.FasterGS.Trainer import FasterGSTrainer
from Methods.FasterGS.disks.Loss import FasterGS2DLoss, normal_loss
from Methods.FasterGS.disks.Model import FasterGS2DModel, Gaussians2D
from Methods.FasterGS.disks.Renderer import depth_to_normal
from Methods.FasterGS.disks.Trainer import FasterGS2DTrainer


class FasterGS2DTest(unittest.TestCase):
    def setUp(self):
        self.config = Framework.ConfigParameterList(
            GLOBAL=Framework.ConfigParameterList(METHOD_TYPE="FasterGS2D", DEFAULT_DEVICE="cpu"),
            MODEL=Framework.ConfigParameterList(),
            TRAINING=Framework.ConfigParameterList(
                WANDB=Framework.ConfigParameterList(ACTIVATE=False)),
        )
        self.context = patch.object(Framework, "config", self.config, create=True)
        self.context.start()
        self.addCleanup(self.context.stop)

    def test_disk_spacing_uses_sklearn_even_when_fused_knn_is_available(self):
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [7., 0., 0.]])
        with patch("Optim.knn_utils._fused_knn", side_effect=AssertionError("unexpected fused KNN lookup")):
            distances = Gaussians2D.initial_distances(points)
        expected = torch.cdist(points, points).square().sum(dim=1).div(3).sqrt()
        torch.testing.assert_close(distances, expected)

    def test_sh_roundtrip_and_own_refusals(self):
        for degree in range(4):
            g = Gaussians2D(3, pretrained=False)
            for _ in range(degree):
                g.increase_used_sh_degree()
            restored = Gaussians2D(3, pretrained=True)
            restored.load_state_dict(g.state_dict())
            self.assertEqual((restored.active_sh_degree, restored.active_sh_bases),
                             (degree, (degree+1)**2))
        g = Gaussians2D(3, pretrained=False)
        for state in (None, {}, {"version": 1, "active_sh_degree": True, "active_sh_bases": 4},
                      {"version": 1, "active_sh_degree": 1, "active_sh_bases": 16}):
            with self.assertRaisesRegex(Framework.ModelError, "invalid saved SH execution state"):
                g.set_extra_state(state)
        with self.assertRaisesRegex(Framework.ModelError, "missing SH execution state"):
            g.load_state_dict({}, strict=False)
        with self.assertRaisesRegex(Framework.ModelError, "requires Nx2"):
            g.load_state_dict({"_scales": torch.zeros(2, 3)}, strict=False)
        for callback, text in (
            (lambda: g.reset_densification_info(True), "ABSgrad"),
            (lambda: g.setup_3d_filter(), "3D filters"),
            (lambda: g.mcmc_densification(), "MCMC"),
            (lambda: g.post_optimizer_step(True), "MCMC noise"),
        ):
            with self.assertRaisesRegex(Framework.ModelError, text):
                callback()

    def test_model_checkpoint_and_ply_keep_two_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FasterGS2DModel().build()
            g = model.gaussians
            for name, shape in (("_means", (2, 3)), ("_scales", (2, 2)),
                                ("_rotations", (2, 4)), ("_opacities", (2, 1)),
                                ("_sh_coefficients_0", (2, 1, 3)),
                                ("_sh_coefficients_rest", (2, 15, 3))):
                setattr(g, name, torch.nn.Parameter(torch.zeros(shape)))
            with torch.no_grad():
                g._rotations[:, 0] = 1
            g.increase_used_sh_degree()
            model.num_iterations_trained = 7
            path = Path(directory) / "model.pt"
            model.save(path)
            loaded = FasterGS2DModel.load(str(path))
            self.assertEqual(loaded.gaussians.raw_scales.shape, (2, 2))
            self.assertEqual(loaded.gaussians.active_sh_bases, 4)
            names = loaded.gaussians.as_ply_dict()["vertex"].dtype.names
            self.assertEqual([key for key in names if key.startswith("scale_")],
                             ["scale_0", "scale_1"])
            self.assertEqual(torch.linalg.matrix_rank(loaded.gaussians.covariances).tolist(), [2, 2])
            for key, value in model.state_dict().items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)

    def test_geometry_loss_serializes_without_retaining_render_graph(self):
        config = Framework.ConfigParameterList(
            LAMBDA_L1=.8, LAMBDA_DSSIM=0., LAMBDA_OPACITY_REGULARIZATION=0.,
            LAMBDA_SCALE_REGULARIZATION=0., LAMBDA_DISTORTION=.1, LAMBDA_NORMAL=.05,
        )
        loss = FasterGS2DLoss(config, SimpleNamespace(gaussians=Gaussians(3, False), ppisp=None))
        data = io.BytesIO()
        torch.save(loss, data)
        data.seek(0)
        loaded = torch.load(data, weights_only=False)  # Only this test's own in-memory object.
        for objective in (loss, loaded):
            rgb = torch.full((3, 3, 3), .25, requires_grad=True)
            dist = torch.ones(1, 3, 3, requires_grad=True)
            rendered = torch.zeros(3, 3, 3, requires_grad=True)
            surface = torch.ones(3, 3, 3)
            value = objective({"rgb": rgb, "rend_dist": dist, "rend_normal": rendered,
                               "surf_normal": surface}, torch.zeros_like(rgb))
            torch.testing.assert_close(value, torch.tensor(.35))
            value.backward()
            torch.testing.assert_close(rgb.grad, torch.full_like(rgb, .8/27))
            torch.testing.assert_close(dist.grad, torch.full_like(dist, .1/9))
            torch.testing.assert_close(rendered.grad, torch.full_like(rendered, -.05/9))
            self.assertFalse(hasattr(objective, "pkg"))

    def test_calibrated_normal_and_detached_alpha_policy(self):
        camera = SimpleNamespace(focal_x=11., focal_y=13., center_x=2.3, center_y=3.1)
        view = SimpleNamespace(camera=camera, rotation=torch.eye(3), position=torch.zeros(3))
        y, x = torch.meshgrid(torch.arange(7), torch.arange(9), indexing="ij")
        depth = (2/(1-.2*(x+.5-camera.center_x)/camera.focal_x
                    +.1*(y+.5-camera.center_y)/camera.focal_y))[None].requires_grad_()
        surface = depth_to_normal(view, depth)[:, 1:-1, 1:-1]
        expected = torch.tensor([.2, -.1, -1.])
        expected /= expected.norm()
        torch.testing.assert_close(surface, expected[:, None, None].expand_as(surface), atol=2e-6, rtol=2e-6)
        for alpha in (0., .25, 1.):
            rendered = expected[:, None, None].expand_as(surface).detach().requires_grad_()
            value = normal_loss(rendered, surface*alpha)
            torch.testing.assert_close(value, value.new_tensor(1-alpha), atol=2e-6, rtol=0)
            grad, = torch.autograd.grad(value, rendered)
            torch.testing.assert_close(grad, -surface.detach()*alpha/35)
        with self.assertRaisesRegex(Framework.RendererError, "at least a 3x3"):
            depth_to_normal(view, depth[:, :2])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for the fused depth adjoint")
    def test_depth_difference_adjoint(self):
        from Methods.FasterGS.disks.Renderer import _FixedCameraDifferences, _world_differences
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(17)
        for h, w in ((3, 3), (3, 11), (7, 4), (19, 37)):
            rand = lambda *shape: torch.randn(shape, device=device, generator=generator)
            z = rand(h, w).requires_grad_()
            args = (z, rand(w), rand(h), rand(3, 3).T, rand(3))
            reference = _world_differences(*args)
            fast = _FixedCameraDifferences.apply(*args)
            seeds = [rand(*v.shape).requires_grad_() for v in reference]
            for a, b in zip(reference, fast):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            expected, = torch.autograd.grad(reference, z, seeds, create_graph=True)
            actual, = torch.autograd.grad(fast, z, seeds, retain_graph=True)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=4e-6)
            differentiable, = torch.autograd.grad(fast, z, seeds, create_graph=True)
            probe = rand(h, w)
            for a, b in zip(torch.autograd.grad(expected, seeds, probe),
                            torch.autograd.grad(differentiable, seeds, probe)):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=4e-6)

    def test_schedule_and_fail_closed_training_policy(self):
        # Validate the actual methods without constructing a GPU trainer.
        trainer = FasterGS2DTrainer.__new__(FasterGS2DTrainer)
        torch.nn.Module.__init__(trainer)
        trainer.__dict__.update(deepcopy(FasterGS2DTrainer.get_default_parameters()))
        trainer.model = SimpleNamespace(ppisp=None)
        with self.assertRaisesRegex(Framework.TrainingError, "explicit finite positive"):
            trainer.validate_contract()
        trainer.NUM_ITERATIONS = 20
        trainer.validate_contract()
        for value in (True, "0.1", 0., -1., float("nan"), float("inf")):
            trainer.DENSIFICATION_GRAD_THRESHOLD = value
            with self.assertRaisesRegex(Framework.TrainingError, "explicit finite positive"):
                trainer.validate_contract()
        trainer.DENSIFICATION_GRAD_THRESHOLD = None
        for value in (True, "1", 0., float("nan")):
            trainer.DENSIFICATION_MAX_SCREEN_SIZE = value
            with self.assertRaisesRegex(Framework.TrainingError, "finite positive pixels"):
                trainer.validate_contract()
        trainer.DENSIFICATION_MAX_SCREEN_SIZE = None
        stride = trainer.DENSIFICATION_INTERVAL
        for value in (True, "100", 100.5, 0):
            trainer.DENSIFICATION_INTERVAL = value
            with self.assertRaisesRegex(Framework.TrainingError, "intervals must be positive"):
                trainer.validate_contract()
        trainer.DENSIFICATION_INTERVAL = stride
        for value in (True, 1.5, -1, 4):
            model = FasterGS2DModel()
            model.SH_DEGREE = value
            with self.assertRaisesRegex(Framework.ModelError, "SH degree 0 through 3"):
                model.build()
        for key, text in (("USE_MCMC", "MCMC"), ("DENSIFICATION_USE_ABS_GRAD", "ABSgrad")):
            setattr(trainer, key, True)
            with self.assertRaisesRegex(Framework.TrainingError, text):
                trainer.validate_contract()
            setattr(trainer, key, False)
        for iteration, yes in ((500, False), (600, True), (14900, True), (15000, False)):
            self.assertEqual(trainer._should_densify(iteration), yes)
        dataset = SimpleNamespace(default_camera=SimpleNamespace(background_color=torch.zeros(3)))
        for iteration, yes in ((0, False), (3000, True), (12000, True), (15000, False)):
            self.assertEqual(trainer._should_reset_opacities(iteration, dataset), yes)
        self.assertEqual(Gaussians.scale_dim, 3)
        self.assertEqual(FasterGSModel.gaussian_type, Gaussians)
        self.assertEqual(FasterGSTrainer.get_default_parameters().DENSIFICATION_GRAD_THRESHOLD, .0002)
        self.assertFalse(FasterGSModel.get_default_parameters().PPISP.USE)


if __name__ == "__main__":
    unittest.main()
