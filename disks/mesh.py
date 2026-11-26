"""Fuse FasterGS2D training views into a colored TSDF mesh."""
import argparse
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import open3d as o3d
import torch

if __name__ == '__main__' and not __package__:
    # A direct script launch needs the enclosing NeRF-ICG src directory on the import path.
    sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

import Framework
from Implementations import Datasets, Methods


@torch.inference_mode()
def fuse(renderer, views, voxel, truncation, max_depth, sphere, depth_stride=1):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel, truncation, o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        volume_unit_resolution=8, depth_sampling_stride=depth_stride,
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


def extract_surface(volume, weight):
    try:
        return volume.extract_triangle_mesh(weight_threshold=float(weight))
    except TypeError:
        # Standard legacy Open3D fixes its own extraction threshold.
        return volume.extract_triangle_mesh()


def select_foreground(renderer, views, max_depth):
    """Estimate the repeatedly framed surface from a coarse TSDF."""
    sampled = views[::3]
    voxel = max_depth / 560
    volume = fuse(renderer, sampled, voxel, 4 * voxel, max_depth, None, 2)
    mesh = extract_surface(volume, max(3, round(len(sampled) * 0.10)))
    points = np.asarray(mesh.vertices)
    votes = np.zeros(len(points), dtype=np.int32)
    for view in sampled:
        pose, camera = view.w2c_numpy, view.camera
        xyz = points @ pose[:3, :3].T + pose[:3, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (xyz[:, 0] / xyz[:, 2] * camera.focal_x + camera.center_x) / camera.width - 0.5
            v = (xyz[:, 1] / xyz[:, 2] * camera.focal_y + camera.center_y) / camera.height - 0.5
        votes += (xyz[:, 2] > 0) & (u*u + v*v < 0.30**2)

    core = points[votes >= max(3, math.ceil(0.35 * len(sampled)))]
    if len(core) < 10:
        raise ValueError('Too few repeatedly framed surface points. Set --sphere and --voxel-size.')
    center = np.median(core, axis=0)
    radius = 1.15 * np.quantile(np.linalg.norm(core - center, axis=1), 0.90)
    pixels = []
    for view in sampled:
        pose = view.w2c_numpy
        depth = pose[2, :3] @ center + pose[2, 3]
        if depth > 0:
            pixels.append(depth / view.camera.focal_x)
    if radius <= 0 or not pixels:
        raise ValueError('Degenerate foreground. Set --sphere and --voxel-size.')
    # Pad the crop for thin peripheral geometry while retaining the core resolution.
    return (*center, 1.6 * radius), max(radius / 220, 2 * np.median(pixels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--voxel-size', type=float, help='Voxel width in scene units.')
    parser.add_argument('--max-depth', type=float, required=True, help='Camera-space depth cutoff.')
    parser.add_argument('--truncation', type=float, help='TSDF band in scene units (default: four voxels).')
    region = parser.add_mutually_exclusive_group()
    region.add_argument('--auto-foreground', action='store_true', help='Infer the foreground sphere and voxel width.')
    region.add_argument('--sphere', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'R'),
                        help='Restrict fusion to a world-space foreground sphere.')
    parser.add_argument('--depth-stride', type=int, choices=(1, 2), default=1,
                        help='Depth sampling stride for allocating TSDF blocks (2 speeds up previews).')
    parser.add_argument('--smooth', type=int, default=0, metavar='ITERATIONS', help='Taubin smoothing passes.')
    args = parser.parse_args()
    if args.voxel_size is None and not args.auto_foreground:
        parser.error('--voxel-size is required unless --auto-foreground is enabled')
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (args.voxel_size, args.max_depth, args.truncation)):
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
    views = dataset.train()
    if args.auto_foreground:
        views = list(views)
        args.sphere, estimated_voxel = select_foreground(renderer, views, args.max_depth)
        args.voxel_size = args.voxel_size or estimated_voxel
        print('Foreground: --sphere', *map(float, args.sphere), '--voxel-size', args.voxel_size)
    band = args.truncation or 4 * args.voxel_size
    volume = fuse(renderer, views, args.voxel_size, band, args.max_depth, args.sphere, args.depth_stride)
    mesh = extract_surface(volume, 3)
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
