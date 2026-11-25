"""Fuse FasterGS2D training views into a colored TSDF mesh."""
import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import open3d as o3d
import torch

import Framework
from Implementations import Datasets, Methods


@torch.inference_mode()
def fuse(renderer, views, voxel, truncation, max_depth, sphere):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel, truncation, o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        volume_unit_resolution=8, depth_sampling_stride=1,
    )
    for view in views:
        camera = view.camera
        pose = view.w2c_numpy
        maps = renderer.render_image_inference(view)
        depth = maps['depth'][..., 0]
        valid = (depth > 0) & (depth < max_depth) & (maps['alpha'][..., 0] >= 0.5)
        if sphere is not None:
            # Test the surface sample in camera coordinates before allocating TSDF blocks.
            x, y, z = pose[:3, :3] @ sphere[:3] + pose[:3, 3]
            u = torch.arange(camera.width, device=depth.device, dtype=depth.dtype) + 0.5
            v = torch.arange(camera.height, device=depth.device, dtype=depth.dtype) + 0.5
            dx = (u - camera.center_x) / camera.focal_x * depth - x
            dy = (v[:, None] - camera.center_y) / camera.focal_y * depth - y
            valid &= dx.square() + dy.square() + (depth - z).square() <= sphere[3] ** 2
        rgb = (maps['rgb'].clamp(0, 1) * 255).round().byte().contiguous().cpu().numpy()
        z_image = depth.masked_fill(~valid, 0).contiguous().cpu().numpy()
        frame = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(z_image), depth_scale=1.0,
            depth_trunc=max_depth, convert_rgb_to_intensity=False,
        )
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height, camera.focal_x, camera.focal_y,
            camera.center_x - 0.5, camera.center_y - 0.5,
        )
        volume.integrate(frame, intrinsics, pose)
    return volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--voxel-size', type=float, required=True, help='Voxel width in scene units.')
    parser.add_argument('--max-depth', type=float, required=True, help='Camera-space depth cutoff.')
    parser.add_argument('--truncation', type=float, help='TSDF band in scene units (default: four voxels).')
    parser.add_argument('--sphere', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'R'),
                        help='Restrict fusion to a world-space foreground sphere.')
    parser.add_argument('--smooth', type=int, default=0, metavar='ITERATIONS', help='Taubin smoothing passes.')
    args = parser.parse_args()
    band = args.voxel_size * 4 if args.truncation is None else args.truncation
    if not all(math.isfinite(v) and v > 0 for v in (args.voxel_size, args.max_depth, band)):
        parser.error('voxel size, maximum depth and truncation must be positive and finite')
    if args.sphere and (not all(map(math.isfinite, args.sphere)) or args.sphere[3] <= 0):
        parser.error('sphere requires a finite center and a positive radius')
    if args.smooth < 0 or args.output.suffix.lower() != '.ply':
        parser.error('use a PLY output path and a nonnegative smoothing count')

    Framework.setup(config_path=str(args.config), require_custom_config=True)
    dataset = Datasets.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    model = Methods.get_model('FasterGS2D', checkpoint=str(args.checkpoint)).eval()
    renderer = Methods.get_renderer('FasterGS2D', model)
    renderer.DEPTH_RATIO = 0.0
    volume = fuse(renderer, dataset.train(), args.voxel_size, band, args.max_depth, args.sphere)
    # Some Open3D builds expose a configurable observation threshold with a high default.
    try:
        mesh = volume.extract_triangle_mesh(weight_threshold=3.0)
    except TypeError:
        mesh = volume.extract_triangle_mesh()
    del volume
    if not mesh.has_triangles():
        raise RuntimeError('Empty mesh. Check the bounds and depth cutoff in checkpoint coordinates.')
    if args.smooth:
        mesh = mesh.filter_smooth_taubin(args.smooth, filter_scope=o3d.geometry.FilterScope.Vertex)
    mesh.compute_vertex_normals()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / args.output.name
        if not o3d.io.write_triangle_mesh(str(staged), mesh):
            raise OSError(f'Could not write {args.output}')
        staged.replace(args.output)


if __name__ == '__main__':
    main()
