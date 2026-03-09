"""
Minimal demo for Ball Query (pytorch3d) and stats reporting.

Run:
  python -m test_ops.run_ball_query_demo
"""

import sys
from pathlib import Path

import torch

# Ensure point_ops is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import point_ops
# Demo config (edit as needed)
DEVICE = "gpu"  # "gpu" | "cpu"
B = 1
N = 64
M = 8
RADIUS = 0.2
MAX_NEIGHBORS = 16

def main() -> None:
    point_ops.config.set_mode("reference")
    point_ops.config.set_stats_enabled(True)
    point_ops.config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # B=1, N=64, D=3 (xyz) and M queries
    xyz = torch.randn(B, N, 3, device=device)
    queries = xyz[:, :M, :].contiguous()

    op = point_ops.BallQuery(radius=RADIUS, max_neighbors=MAX_NEIGHBORS)
    idx, mask, counts = op(xyz, queries)

    print("device:", device)
    print("idx shape:", tuple(idx.shape))
    print("counts:", counts)
    print("=== stats report ===")
    print(point_ops.stats_report())


if __name__ == "__main__":
    main()
