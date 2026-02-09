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
POINT_OPS = ROOT / "point_ops"
if str(POINT_OPS) not in sys.path:
    sys.path.insert(0, str(POINT_OPS))

import config
from stats import report as stats_report
# Demo config (edit as needed)
DEVICE = "gpu"  # "gpu" | "cpu"
B = 1
N = 64
M = 8
RADIUS = 0.2
MAX_NEIGHBORS = 16

from ops.ball_query import BallQuery


def main() -> None:
    config.set_mode("reference")
    config.set_stats_enabled(True)
    config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # B=1, N=64, D=3 (xyz) and M queries
    xyz = torch.randn(B, N, 3, device=device)
    queries = xyz[:, :M, :].contiguous()

    op = BallQuery(radius=RADIUS, max_neighbors=MAX_NEIGHBORS)
    idx, mask, counts = op(xyz, queries)

    print("device:", device)
    print("idx shape:", tuple(idx.shape))
    print("counts:", counts)
    print("=== stats report ===")
    print(stats_report())


if __name__ == "__main__":
    main()
