"""
Demo for point_ops stats snapshot behavior.

Run:
  python -m test_ops.run_stats_snapshot_demo

This script validates:
1. snapshot() returns the expected top-level structure.
2. Logged calls appear in both class_stats and instance_stats.
3. snapshot() is detached from internal registry state.
4. reset() clears accumulated counters.
5. disabled() affects the exported enabled flag for scoped reads.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure point_ops is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import point_ops.stats as point_stats


class DummyOp:
    pass


def _assert_zero_metrics(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if value != 0.0:
            raise AssertionError(f"Expected zero metric for {key!r}, got {value}.")


def main() -> None:
    point_stats.set_enabled(True)
    point_stats.reset()

    snap0 = point_stats.snapshot()
    print("=== initial snapshot ===")
    print(snap0)

    if sorted(snap0.keys()) != ["class_stats", "enabled", "instance_stats", "items"]:
        raise AssertionError(f"Unexpected snapshot keys: {sorted(snap0.keys())}")
    if snap0["enabled"] is not True:
        raise AssertionError("Expected stats to be enabled in the initial snapshot.")
    if snap0["class_stats"] != {}:
        raise AssertionError("Expected empty class_stats after reset.")
    if snap0["instance_stats"] != {}:
        raise AssertionError("Expected empty instance_stats after reset.")

    op = DummyOp()
    point_stats.register_instance(op, "DummyOp", local_name="snapshot_case", meta={"phase": "unit"})
    point_stats.log_call(
        op,
        {
            "calls": 2.0,
            "time": 3.5,
            "flops": 16.0,
            "bytes_read": 64.0,
            "bytes_write": 32.0,
        },
    )

    snap1 = point_stats.snapshot()
    print("\n=== populated snapshot ===")
    print(snap1)

    dummy_class = snap1["class_stats"].get("DummyOp")
    if dummy_class is None:
        raise AssertionError("Expected DummyOp to appear in class_stats.")
    if dummy_class["calls"] != 2.0 or dummy_class["time"] != 3.5:
        raise AssertionError(f"Unexpected class metrics: {dummy_class}")

    instance_names = list(snap1["instance_stats"].keys())
    if len(instance_names) != 1:
        raise AssertionError(f"Expected one instance entry, got {instance_names}")

    instance_entry = snap1["instance_stats"][instance_names[0]]
    if instance_entry["class_name"] != "DummyOp":
        raise AssertionError(f"Unexpected class_name: {instance_entry['class_name']}")
    if instance_entry["meta"] != {"phase": "unit"}:
        raise AssertionError(f"Unexpected instance meta: {instance_entry['meta']}")
    if instance_entry["metrics"]["bytes_write"] != 32.0:
        raise AssertionError(f"Unexpected instance metrics: {instance_entry['metrics']}")

    # snapshot() must be detached from registry state.
    snap1["class_stats"]["DummyOp"]["calls"] = -999.0
    snap1["instance_stats"][instance_names[0]]["metrics"]["time"] = -999.0
    snap_copy_check = point_stats.snapshot()
    if snap_copy_check["class_stats"]["DummyOp"]["calls"] != 2.0:
        raise AssertionError("snapshot() must return detached metric copies.")
    if snap_copy_check["instance_stats"][instance_names[0]]["metrics"]["time"] != 3.5:
        raise AssertionError("snapshot() must return detached instance metric copies.")

    with point_stats.disabled():
        snap_disabled = point_stats.snapshot()
        print("\n=== disabled-scope snapshot ===")
        print(snap_disabled)
        if snap_disabled["enabled"] is not False:
            raise AssertionError("Expected snapshot enabled flag to be False inside disabled() scope.")
        if snap_disabled["class_stats"]["DummyOp"]["calls"] != 2.0:
            raise AssertionError("disabled() should not erase accumulated counters.")

    point_stats.reset()
    snap2 = point_stats.snapshot()
    print("\n=== post-reset snapshot ===")
    print(snap2)

    if snap2["class_stats"] != {} or snap2["instance_stats"] != {}:
        raise AssertionError("Expected reset() to clear accumulated class/instance stats.")

    print("\nPASS: stats snapshot behavior is consistent.")


if __name__ == "__main__":
    main()
