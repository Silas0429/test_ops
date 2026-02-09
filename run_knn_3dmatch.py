"""
KNN visualization for a single point cloud (NPZ intermediates).

Usage:
  python -m test_ops.run_knn_3dmatch --input /home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen --k 16 --num-centers 5

Outputs (in test_ops/intermediate/knn):
  - knn_source.npz  (all points, source color)
  - knn_sample.npz  (KNN samples, sample color)
  - knn_center.npz  (center points duplicated to appear larger)
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
from ops.knn import KNN


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen",
        help="Point cloud file or directory",
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--num-centers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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

    xyz_b = xyz[None, ...]
    op = KNN(k=args.k, include_self=False)
    idx, dist2, mask = op(xyz_b)
    idx = np.asarray(idx)[0]  # [N, K]

    # Collect neighbor indices for all centers
    knn_indices = np.unique(idx[center_indices].reshape(-1))

    # Colors
    source_color = np.array([0.0, 0.651, 0.929], dtype=np.float32)  # blue
    sample_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)       # green
    center_color = np.array([1.0, 0.706, 0.0], dtype=np.float32)      # yellow

    out_dir = ROOT / "test_ops" / "intermediate" / "knn"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "knn_source.npz",
        points=xyz,
        colors=np.tile(source_color[None, :], (xyz.shape[0], 1)),
    )
    np.savez(
        out_dir / "knn_sample.npz",
        points=xyz[knn_indices],
        colors=np.tile(sample_color[None, :], (knn_indices.shape[0], 1)),
    )

    # Duplicate center points with tiny jitter to appear larger in viewers.
    centers = xyz[center_indices]
    jitter = (rng.random((centers.shape[0] * 30, 3)).astype(np.float32) - 0.5) * 1e-3
    center_pts = np.repeat(centers, 30, axis=0) + jitter
    center_colors = np.tile(center_color[None, :], (center_pts.shape[0], 1))
    np.savez(out_dir / "knn_center.npz", points=center_pts, colors=center_colors)

    print(f"Center indices: {center_indices.tolist()}")
    print(f"Saved KNN intermediates to: {out_dir}")
    print("\n=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
