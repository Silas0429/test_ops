"""
Demo for point_ops stats enable/disable/reset behavior.

Run:
  python -m test_ops.run_stats_control_demo

This script validates three things:
1. Stats can be disabled explicitly and no counters are collected.
2. Reset clears accumulated counters between measurement windows.
3. Two consecutive measured runs accumulate deterministic cost fields.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict

import torch

# Ensure point_ops is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import point_ops


# Demo config (edit as needed)
DEVICE = "cpu"  # "cpu" | "gpu"
B = 2
N = 256
PHASE1_NUM_SAMPLES = 32
PHASE2_NUM_SAMPLES = 48
PHASE1_DETERMINISTIC = True
PHASE2_DETERMINISTIC = False


def _parse_class_metrics(report: str, class_name: str) -> Dict[str, float]:
    for line in report.splitlines():
        if re.match(rf"^{re.escape(class_name)}(\s{{2,}}|\s*$)", line):
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) != 6:
                raise ValueError(f"Unexpected class-summary row for {class_name!r}: {line!r}")
            return {
                "calls": float(parts[1]),
                "time": float(parts[2]),
                "flops": float(parts[3]),
                "bytes_read": float(parts[4]),
                "bytes_write": float(parts[5]),
            }
    raise ValueError(f"Class {class_name!r} not found in stats report:\n{report}")


def _print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    print(f"  calls       : {metrics['calls']:.0f}")
    print(f"  time        : {metrics['time']:.6g}")
    print(f"  flops       : {metrics['flops']:.0f}")
    print(f"  bytes_read  : {metrics['bytes_read']:.0f}")
    print(f"  bytes_write : {metrics['bytes_write']:.0f}")


def _run_and_collect(op: point_ops.FPS, xyz: torch.Tensor) -> Dict[str, float]:
    _ = op(xyz)
    report = point_ops.stats_report()
    metrics = _parse_class_metrics(report, "FPS")
    if metrics["time"] <= 0:
        raise AssertionError("Expected FPS measured time to be > 0.")
    return metrics


def main() -> None:
    point_ops.config.set_mode("reference")
    point_ops.config.set_device("cuda" if DEVICE == "gpu" else "cpu")

    use_cuda = DEVICE == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    xyz = torch.arange(B * N * 3, dtype=torch.float32, device=device).reshape(B, N, 3)
    op_phase1 = point_ops.FPS(
        num_samples=PHASE1_NUM_SAMPLES,
        deterministic=PHASE1_DETERMINISTIC,
        name="stats_control/fps_phase1",
    )
    op_phase2 = point_ops.FPS(
        num_samples=PHASE2_NUM_SAMPLES,
        deterministic=PHASE2_DETERMINISTIC,
        name="stats_control/fps_phase2",
    )

    print("device:", device)
    print("input shape:", tuple(xyz.shape))
    print("phase1 config:", {"num_samples": PHASE1_NUM_SAMPLES, "deterministic": PHASE1_DETERMINISTIC})
    print("phase2 config:", {"num_samples": PHASE2_NUM_SAMPLES, "deterministic": PHASE2_DETERMINISTIC})

    try:
        _ = op_phase1(xyz)
        _ = op_phase2(xyz)
    except ImportError as exc:
        print(f"Dependency missing: {exc}")
        print("This demo requires the point_ops FPS reference path to be available.")
        return

    # Phase 0: disabled collection should produce no stats.
    point_ops.stats_set_enabled(False)
    point_ops.stats_reset()
    _ = op_phase1(xyz)
    disabled_report = point_ops.stats_report()
    print("\n=== disabled stats report ===")
    print(disabled_report)
    if disabled_report.strip() != "No stats collected.":
        raise AssertionError("Expected no stats while collection is disabled.")

    # Phase 1: first measured run.
    point_ops.stats_enable()
    point_ops.stats_reset()
    first = _run_and_collect(op_phase1, xyz)
    print("\n=== first measured run ===")
    _print_metrics("first run:", first)

    # Phase 2: second measured run after reset.
    point_ops.stats_reset()
    second = _run_and_collect(op_phase2, xyz)
    print("\n=== second measured run ===")
    _print_metrics("second run:", second)

    # Phase 3: both runs without reset between them.
    point_ops.stats_reset()
    _ = op_phase1(xyz)
    _ = op_phase2(xyz)
    cumulative = _parse_class_metrics(point_ops.stats_report(), "FPS")
    print("\n=== cumulative mixed-run window ===")
    _print_metrics("mixed-run cumulative:", cumulative)

    expected = {
        "calls": first["calls"] + second["calls"],
        "flops": first["flops"] + second["flops"],
        "bytes_read": first["bytes_read"] + second["bytes_read"],
        "bytes_write": first["bytes_write"] + second["bytes_write"],
    }

    for key, expected_value in expected.items():
        if cumulative[key] != expected_value:
            raise AssertionError(
                f"Cumulative {key} mismatch: got {cumulative[key]}, expected {expected_value}."
            )

    if cumulative["time"] <= 0:
        raise AssertionError("Expected cumulative measured time to be > 0.")

    print("\nPASS: stats enable/disable/reset behavior is consistent for FPS.")
    print("Note: 'time' is runtime-measured and is not expected to match the sum of earlier single-run windows exactly.")


if __name__ == "__main__":
    main()
