"""
Ball query visualization for a single point cloud (NPZ intermediates).

Usage:
  python -m test_ops.run_ball_query_3dmatch \
    --input /home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen \
    --radius 0.05 --max-neighbors 64 --num-centers 5

Outputs (in test_ops/intermediate/ball_query):
  - ball_query_source.npz  (all points, source color)
  - ball_query_sample.npz  (neighbors within radius, sample color)
  - ball_query_center.npz  (centers + sphere points to visualize radius)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure point_ops is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
POINT_OPS = ROOT / "point_ops"
if str(POINT_OPS) not in sys.path:
    sys.path.insert(0, str(POINT_OPS))

import config
from stats import report as stats_report
from ops.ball_query import BallQuery


def _find_pointcloud(path: Path) -> Path:
    if path.is_file():
        return path
    exts = [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"]
    for ext in exts:
        found = next(path.glob(f"*{ext}"), None)
        if found:
            return found
    raise FileNotFoundError(f"No point cloud file found under {path}")


def _read_xyz(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".ply":
        try:
            from plyfile import PlyData
        except Exception as exc:
            raise ImportError("plyfile is required to read binary PLY. Install with: pip install plyfile") from exc
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        x = np.asarray(v["x"], dtype=np.float32)
        y = np.asarray(v["y"], dtype=np.float32)
        z = np.asarray(v["z"], dtype=np.float32)
        return np.stack([x, y, z], axis=1)
    pts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(pts, dtype=np.float32)


def _sample_sphere_points(
    centers: np.ndarray, radius: float, points_per_center: int, rng: np.random.Generator
) -> np.ndarray:
    if centers.size == 0 or points_per_center <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    total = centers.shape[0] * points_per_center
    dirs = rng.normal(size=(total, 3)).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + 1e-9)
    sphere = dirs * float(radius)
    return np.repeat(centers, points_per_center, axis=0) + sphere


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen",
        help="Point cloud file or directory",
    )
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--max-neighbors", type=int, default=64)
    parser.add_argument("--num-centers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--sphere-points", type=int, default=200)
    parser.add_argument("--center-dup", type=int, default=30)
    args = parser.parse_args()

    if args.radius <= 0:
        raise ValueError("radius must be > 0")
    if args.max_neighbors <= 0:
        raise ValueError("max-neighbors must be > 0")

    in_path = _find_pointcloud(Path(args.input))
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    xyz = _read_xyz(in_path)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Point cloud must have shape [N, 3].")

    config.set_mode("reference")
    config.set_device("cuda" if args.device == "gpu" else "cpu")
    config.set_stats_enabled(True)

    n_pts = xyz.shape[0]
    num_centers = min(args.num_centers, n_pts)
    rng = np.random.default_rng(args.seed)
    center_indices = rng.choice(n_pts, size=num_centers, replace=False)
    centers = xyz[center_indices]

    xyz_b = xyz[None, ...]
    centers_b = centers[None, ...]
    op = BallQuery(radius=args.radius, max_neighbors=args.max_neighbors)
    idx, mask, counts = op(xyz_b, centers_b)
    if hasattr(idx, "detach"):
        idx = idx.detach()
    if hasattr(idx, "cpu"):
        idx = idx.cpu()
    if hasattr(mask, "detach"):
        mask = mask.detach()
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    idx = np.asarray(idx)[0]    # [M, K]
    mask = np.asarray(mask)[0]  # [M, K]

    valid_idx = idx[mask]
    if valid_idx.size:
        valid_idx = valid_idx[valid_idx >= 0]
        sample_indices = np.unique(valid_idx)
        samples = xyz[sample_indices]
    else:
        samples = np.zeros((0, 3), dtype=np.float32)

    # Colors
    source_color = np.array([0.0, 0.651, 0.929], dtype=np.float32)  # blue
    sample_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)       # green
    center_color = np.array([1.0, 0.706, 0.0], dtype=np.float32)      # yellow

    # Make centers appear larger and add sphere points to visualize radius.
    dup = max(int(args.center_dup), 1)
    jitter = (rng.random((centers.shape[0] * dup, 3)).astype(np.float32) - 0.5) * (
        float(args.radius) * 0.02
    )
    centers_dup = np.repeat(centers, dup, axis=0) + jitter
    sphere_pts = _sample_sphere_points(centers, float(args.radius), int(args.sphere_points), rng)
    center_points = np.vstack([centers_dup, sphere_pts])
    center_colors = np.tile(center_color[None, :], (center_points.shape[0], 1))

    out_dir = ROOT / "test_ops" / "intermediate" / "ball_query"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "ball_query_source.npz",
        points=xyz,
        colors=np.tile(source_color[None, :], (xyz.shape[0], 1)),
    )
    np.savez(
        out_dir / "ball_query_sample.npz",
        points=samples,
        colors=np.tile(sample_color[None, :], (samples.shape[0], 1)),
    )
    np.savez(
        out_dir / "ball_query_center.npz",
        points=center_points,
        colors=center_colors,
    )

    print(f"Center indices: {center_indices.tolist()}")
    print(f"Neighbor counts per center: {counts[0].tolist() if hasattr(counts, 'shape') else counts}")
    print(f"Saved BallQuery intermediates to: {out_dir}")
    print("\n=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
