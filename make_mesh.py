"""Fuse openscanner PLY point clouds into a single 3D mesh.

Run on a desktop, not the Pi (Poisson surface reconstruction is RAM-heavy).

  pip install open3d numpy
  python make_mesh.py path/to/captures_dir -o mask.ply

Pipeline:
  1. load every *.ply in the input directory (skip concat.ply)
  2. coarse outlier strip + voxel downsample per cloud
  3. pairwise ICP register each cloud to the first (point-to-plane)
  4. merge into one cloud
  5. statistical outlier removal
  6. estimate normals (oriented towards camera origin)
  7. Poisson surface reconstruction
  8. crop low-density vertices (the "balloon" around sparse regions)
  9. write mesh + the merged cloud
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not installed. `pip install open3d` and retry.")


def load_clouds(folder, skip_names=("concat.ply",)):
    paths = sorted(glob.glob(os.path.join(folder, "*.ply")))
    paths = [p for p in paths if os.path.basename(p) not in skip_names]
    if not paths:
        sys.exit(f"no .ply files in {folder}")
    clouds = []
    for p in paths:
        pc = o3d.io.read_point_cloud(p)
        if len(pc.points) < 500:
            print(f"  skip {os.path.basename(p)} ({len(pc.points)} pts)")
            continue
        print(f"  loaded {os.path.basename(p)}: {len(pc.points)} pts")
        clouds.append(pc)
    if not clouds:
        sys.exit("all clouds were too small")
    return clouds


def preprocess(pc, voxel):
    pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    return pc


def register_to_reference(clouds, voxel):
    """Pairwise point-to-plane ICP, each cloud to the first.

    No global pose graph - assumes captures are taken from similar
    viewpoints (object roughly stationary). For larger orbits, swap in
    multiway registration with FPFH+RANSAC for the global init.
    """
    ref = clouds[0]
    fused = [ref]
    threshold = voxel * 3
    for i, src in enumerate(clouds[1:], start=1):
        result = o3d.pipelines.registration.registration_icp(
            src, ref, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        src.transform(result.transformation)
        fused.append(src)
        print(f"  ICP cloud {i}: fitness={result.fitness:.3f}  "
              f"inlier_rmse={result.inlier_rmse:.4f}")
    return fused


def merge(clouds):
    out = o3d.geometry.PointCloud()
    for c in clouds:
        out += c
    return out


def poisson_mesh(pc, depth, density_quantile):
    pc.orient_normals_towards_camera_location(camera_location=np.zeros(3))
    print(f"  Poisson depth={depth}...")
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth)
    density = np.asarray(density)
    cutoff = np.quantile(density, density_quantile)
    mesh.remove_vertices_by_mask(density < cutoff)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="folder containing N.ply files from openscanner")
    ap.add_argument("-o", "--out", default="mesh.ply", help="output mesh path (.ply or .obj)")
    ap.add_argument("--voxel", type=float, default=0.003,
                    help="voxel size in metres (default 3mm)")
    ap.add_argument("--depth", type=int, default=9,
                    help="Poisson octree depth (8=fast/coarse, 10=slow/detailed)")
    ap.add_argument("--density-quantile", type=float, default=0.05,
                    help="trim vertices below this density quantile (0..1)")
    ap.add_argument("--no-icp", action="store_true",
                    help="skip ICP, just merge as-is")
    ap.add_argument("--save-cloud", default=None,
                    help="optional path to save the merged point cloud (.ply)")
    args = ap.parse_args()

    print("Loading...")
    clouds = load_clouds(args.input_dir)

    print(f"Preprocessing (voxel={args.voxel}m)...")
    clouds = [preprocess(c, args.voxel) for c in clouds]

    if not args.no_icp and len(clouds) > 1:
        print("Registering with ICP...")
        clouds = register_to_reference(clouds, args.voxel)

    print("Merging...")
    merged = merge(clouds)
    merged = merged.voxel_down_sample(args.voxel)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    merged.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 4, max_nn=30))
    print(f"  merged cloud: {len(merged.points)} pts")

    if args.save_cloud:
        o3d.io.write_point_cloud(args.save_cloud, merged)
        print(f"wrote {args.save_cloud}")

    print("Meshing...")
    mesh = poisson_mesh(merged, args.depth, args.density_quantile)
    mesh.compute_vertex_normals()
    print(f"  mesh: {len(mesh.vertices)} verts  {len(mesh.triangles)} tris")

    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
