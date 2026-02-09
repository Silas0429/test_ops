"""
FPS visualization for a single point cloud (PLY with colors).

Usage:
  python -m test_ops.run_fps_3dmatch --input /home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen --num-samples 2048

Outputs (in test_ops/result/fps):
  - fps_source.ply   (source only)
  - fps_samples.ply  (samples only)
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
from ops.fps import FPS


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


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have same length")
    with path.open("w", encoding="utf-8") as f:
        f.write("ply")
        f.write("format ascii 1.0")
        f.write(f"element vertex {points.shape[0]}")
        f.write("property float x")
        f.write("property float y")
        f.write("property float z")
        f.write("property uchar red")
        f.write("property uchar green")
        f.write("property uchar blue")
        f.write("end_header")
        colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen",
        help="Point cloud file or directory",
    )
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--deterministic", action="store_true")
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

    xyz_b = xyz[None, ...]
    op = FPS(num_samples=args.num_samples, deterministic=args.deterministic)
    idx = op(xyz_b)
    if hasattr(idx, "detach"):
        idx = idx.detach()
    if hasattr(idx, "cpu"):
        idx = idx.cpu()
    idx = np.asarray(idx)[0]

    src_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # gray
    samp_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red

    colors = np.tile(src_color[None, :], (xyz.shape[0], 1))
    colors[idx] = samp_color

    out_dir = ROOT / "test_ops" / "result" / "fps"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_ply(out_dir / "fps_source.ply", xyz, np.tile(src_color[None, :], (xyz.shape[0], 1)))
    _write_ply(out_dir / "fps_samples.ply", xyz[idx], np.tile(samp_color[None, :], (len(idx), 1)))

    inter_dir = ROOT / "test_ops" / "intermediate" / "fps"
    inter_dir.mkdir(parents=True, exist_ok=True)
    np.savez(inter_dir / "fps_source.npz", points=xyz, colors=np.tile(src_color[None, :], (xyz.shape[0], 1)))
    np.savez(inter_dir / "fps_samples.npz", points=xyz[idx], colors=np.tile(samp_color[None, :], (len(idx), 1)))

    print(f"Saved FPS results to: {out_dir}")
    print("\n=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
