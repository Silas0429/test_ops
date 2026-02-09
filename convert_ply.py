"""
Convert intermediate NPZ point clouds to PLY using Open3D.

Usage:
  python -m test_ops.convert_ply --input test_ops/intermediate/fps --output test_ops/result/fps

Each NPZ must contain:
  - points: (N, 3)
  - colors: (N, 3) in [0,1]
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover
    raise ImportError("open3d is required in this environment") from exc

GEDI_YELLOW = np.array([1.0, 0.706, 0.0], dtype=np.float64)
GEDI_BLUE = np.array([0.0, 0.651, 0.929], dtype=np.float64)

PRESET_COLORS = {
    "fps_source": (0.0, 0.651, 0.929),
    "fps_samples": (1.0, 0.706, 0.0),
    "knn_source": (0.0, 0.651, 0.929),
    "knn_sample": (0.0, 1.0, 0.0),
    "knn_center": (1.0, 0.706, 0.0),
    "ball_query_source": (0.0, 0.651, 0.929),
    "ball_query_sample": (0.0, 1.0, 0.0),
    "ball_query_center": (1.0, 0.706, 0.0),
}


def _apply_gedi_palette(colors: np.ndarray) -> np.ndarray:
    if colors.size == 0:
        return colors
    # Quantize to make majority-color detection stable.
    quant = np.round(colors.astype(np.float64), 3)
    uniq, counts = np.unique(quant, axis=0, return_counts=True)
    if uniq.shape[0] == 1:
        return np.tile(GEDI_YELLOW[None, :], (colors.shape[0], 1))
    base = uniq[counts.argmax()]
    mask = (quant == base).all(axis=1)
    out = np.empty_like(colors, dtype=np.float64)
    out[mask] = GEDI_BLUE
    out[~mask] = GEDI_YELLOW
    return out


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray, voxel_size: float) -> None:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if voxel_size > 0:
        pc = pc.voxel_down_sample(voxel_size)
    pc.estimate_normals()

    o3d.io.write_point_cloud(str(path), pc, write_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory containing .npz files")
    parser.add_argument("--output", required=True, help="Output directory for .ply files")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size for downsample")
    parser.add_argument("--no-voxel", action="store_true", help="Disable voxel downsample")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {in_dir}")

    for npz in npz_files:
        if npz.stem == "fps_colored":
            # Skip colored FPS output by request.
            continue
        data = np.load(npz)
        points = data["points"]
        colors = data["colors"]
        preset = PRESET_COLORS.get(npz.stem, None)
        if preset is not None:
            colors = np.tile(np.array(preset, dtype=np.float64)[None, :], (points.shape[0], 1))
        else:
            colors = _apply_gedi_palette(colors)
        out_path = out_dir / (npz.stem + ".ply")
        voxel = 0.0 if args.no_voxel else args.voxel_size
        _write_ply(out_path, points, colors, voxel_size=voxel)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
