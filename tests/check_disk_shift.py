"""Independent pixel-translation FD, visibility and signed cancellation controls."""
import argparse
import json
from pathlib import Path

import torch

from FasterGSCudaBackend.torch_bindings.disks import diff_rasterize_disks
from check_disks import fixture, oracle, objective_weights, settings_to


def shift_checks(native=False):
    p, s = fixture()
    results = {}
    for kind in ("oblique_multi", "screen_multi"):
        params = [t.clone() for t in p]
        if kind == "screen_multi":
            params[1][:] = -6
        device_params = [t.cuda().float() for t in params] if native else params
        settings = settings_to(s, "cuda") if native else s
        shift0 = torch.tensor([[.13, -.11], [-.08, .17]], dtype=torch.float64)
        case = {}
        for name, weight in objective_weights(s).items():
            ref_shift = shift0.clone().requires_grad_()
            ref_output = oracle(params, s, ref_shift)
            reference = torch.autograd.grad((ref_output*weight).sum(), ref_shift)[0]
            shift = shift0.cuda().float().requires_grad_() if native else shift0.clone().requires_grad_()
            render = (lambda delta: diff_rasterize_disks(*device_params, settings, image_shift=delta)) if native else (
                lambda delta: oracle(params, s, delta))
            output = render(shift)
            ww = weight.to(output)
            actual = torch.autograd.grad((output*ww).sum(), shift)[0]
            torch.testing.assert_close(output.detach().cpu().double(), ref_output.detach(), atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(actual.cpu().double(), reference, atol=2e-5, rtol=2e-3)
            worst = 0.
            for index in range(shift.numel()):
                eps = .001 if native else 1e-5
                plus, minus = shift.detach().clone(), shift.detach().clone()
                plus.view(-1)[index] += eps
                minus.view(-1)[index] -= eps
                fd = ((render(plus)-render(minus))*ww).sum()/(2*eps)
                torch.testing.assert_close(fd, actual.view(-1)[index],
                                           atol=7e-4 if native else 2e-7,
                                           rtol=2e-2 if native else 2e-5)
                worst = max(worst, abs((fd-actual.view(-1)[index]).item()))
            case[name] = {"fd_checks": shift.numel(), "fd_max_abs_error": worst,
                          "gradient_oracle_max_error": (actual.cpu()-reference).abs().max().item()}
        results[kind] = case
    return results


def native_metadata_controls():
    p, s = fixture()
    s = s._replace(w2c=torch.eye(4, dtype=torch.float64), cam_position=torch.zeros(3, dtype=torch.float64),
                   bg_color=torch.zeros(3, dtype=torch.float64), width=7, height=7,
                   focal_x=10., focal_y=10., center_x=3.5, center_y=3.5)
    p = [t[:1].cuda().float() for t in p]
    p[0][:] = torch.tensor([0., 0., 2.], device="cuda")
    p[1][:] = torch.tensor(.2, device="cuda").log()
    p[2][:] = torch.tensor([1., 0., 0., 0.], device="cuda")
    p[4][:] = p[5][:] = 0
    ss = settings_to(s, "cuda")
    derivatives = []
    for pixels in ((2,), (4,), (2, 4)):
        shift = torch.zeros(1, 2, device="cuda", requires_grad=True)
        image, metadata = diff_rasterize_disks(*p, ss, image_shift=shift, return_metadata=True)
        loss = sum(image[0, 3, x] for x in pixels)
        gradient = torch.autograd.grad(loss, shift)[0]
        assert metadata[1, 0] == 1 and metadata[0, 0] > 0 and not metadata.requires_grad
        derivatives.append(gradient.detach())
    assert derivatives[0].abs().max() > .01
    torch.testing.assert_close(derivatives[0], -derivatives[1], atol=1e-7, rtol=1e-5)
    assert derivatives[2].abs().max() < 1e-7, "signed pixel contributions must cancel before norm"
    # Occluded according to the declared transmittance stop, not tile overlap.
    many = [t.repeat(5, *([1]*(t.ndim-1))) for t in p]
    many[0][:, 2] = torch.tensor([1., 1.1, 1.2, 1.3, 2.], device="cuda")
    many[3][:] = torch.logit(torch.tensor(.95, device="cuda"))
    one_pixel = ss._replace(width=1, height=1, center_x=.5, center_y=.5)
    _, metadata = diff_rasterize_disks(*many, one_pixel, return_metadata=True)
    torch.testing.assert_close(metadata[1], metadata.new_tensor([1, 1, 1, 1, 0]), atol=0, rtol=0)
    assert metadata[0, -1] > 0, "occluded negative control must still overlap the tile"
    many[0][:, 2] = -2
    _, hidden = diff_rasterize_disks(*many, one_pixel, return_metadata=True)
    assert torch.count_nonzero(hidden[1]) == 0
    for invalid in (torch.zeros(1, 3, device="cuda"), torch.zeros(2, 2, device="cuda")):
        try:
            diff_rasterize_disks(*p, ss, image_shift=invalid)
        except RuntimeError as error:
            assert "image_shift" in str(error)
        else:
            raise AssertionError("invalid shift accepted")
    return {"opposing_pixel_gradients": [x.cpu().tolist() for x in derivatives],
            "occluded_and_nohit_visibility": True, "shift_shape_guards": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {"cpu": shift_checks()}
    print(json.dumps(results), flush=True)
    if args.gpu:
        results["native"] = shift_checks(True)
        results["metadata"] = native_metadata_controls()
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results), flush=True)
    print("TRUE2D_SHIFT_PASS", flush=True)


if __name__ == "__main__":
    main()
