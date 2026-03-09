# test_ops

This file is a quick project guide for the `test_ops` repository. It is intended to help a new Codex session understand how operator-level validation is organized in this workspace.

## 1. Goal

`test_ops` is the operator simulation, debugging, and visualization repository for `point_ops`.

Its main purposes are:

- run individual point operators on real or synthetic point clouds
- generate intermediate artifacts that make operator behavior easy to inspect
- convert outputs into `.ply` files for visualization
- provide lightweight demos before model-level integration

This repo sits between low-level operator implementation and full model integration.

## 2. Scope

The repository currently focuses on the implemented reference operators in `point_ops`, especially:

- FPS
- KNN
- Ball Query
- Group
- MLP sub-ops
- SVD demo

It includes both:

- 3DMatch-based scripts for running operators on real point cloud data
- small demo scripts for quick local verification

## 3. Structure

- `test_ops/`
  - `run_3dmatch.sh`
  - `run_fps_3dmatch.py`
  - `run_knn_3dmatch.py`
  - `run_ball_query_3dmatch.py`
  - `run_fps_demo.py`
  - `run_knn_demo.py`
  - `run_ball_query_demo.py`
  - `run_group_demo.py`
  - `run_mlp_demo.py`
  - `run_svd_demo.py`
  - `run_stats_control_demo.py`
  - `run_stats_snapshot_demo.py`
  - `convert_ply.py`
  - `command.md`

## 4. Main Workflow

The main 3DMatch workflow is:

1. run an operator script to produce `.npz` intermediates
2. convert those intermediates into `.ply` using `convert_ply.py`
3. inspect the `.ply` result in Open3D or another point cloud viewer

This makes it easy to verify both semantics and output geometry.

## 5. Current Status

Current state inferred from the repo:

- FPS, KNN, and Ball Query have dedicated 3DMatch runners.
- Group, MLP, and SVD currently have demo-style runners.
- Stats lifecycle and snapshot behavior now also have dedicated demo-style runners.
- `run_3dmatch.sh` is the main batch entry point for FPS / KNN / Ball Query.
- `command.md` is the current command reference and is the first file to check when script behavior changes.

## 6. Dependencies

Common environment expectations:

- PyTorch / PyTorch3D for operator execution
- Open3D or equivalent environment for point cloud conversion / visualization
- `plyfile` for reading binary PLY in the run scripts

The current workflow often uses separate environments:

- a `pytorch3d`-style environment for operator execution
- a `gedi` or Open3D-capable environment for `.ply` conversion

## 7. Inputs and Data

The main dataset used by current scripts is 3DMatch.

Typical default dataset path:

- `/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen`

The scripts usually accept either:

- a dataset directory
- or a specific point cloud file

## 8. Outputs

Typical output layout:

- intermediates:
  - `test_ops/intermediate/fps/`
  - `test_ops/intermediate/knn/`
  - `test_ops/intermediate/ball_query/`
- visualization results:
  - `test_ops/result/fps/`
  - `test_ops/result/knn/`
  - `test_ops/result/ball_query/`

Color conventions currently follow the GeDi-style palette described in `test_ops/command.md`.

## 9. How To Run

### 9.1 Main batch entry

From the workspace root:

```bash
bash test_ops/run_3dmatch.sh -n fps
bash test_ops/run_3dmatch.sh -n knn
bash test_ops/run_3dmatch.sh -n ball_query
bash test_ops/run_3dmatch.sh -n all
```

### 9.2 Single operator runners

Examples:

```bash
python -m test_ops.run_fps_3dmatch
python -m test_ops.run_knn_3dmatch
python -m test_ops.run_ball_query_3dmatch
```

### 9.3 Demos

The demo scripts are intended for direct editing and rerun. They generally do not rely on a heavy CLI.

Current demos:

- `test_ops/run_fps_demo.py`
- `test_ops/run_knn_demo.py`
- `test_ops/run_ball_query_demo.py`
- `test_ops/run_group_demo.py`
- `test_ops/run_mlp_demo.py`
- `test_ops/run_svd_demo.py`
- `test_ops/run_stats_control_demo.py`
- `test_ops/run_stats_snapshot_demo.py`

Example direct runs:

```bash
python -m test_ops.run_stats_control_demo
python -m test_ops.run_stats_snapshot_demo
```

## 10. Typical Tasks In This Repo

When working in `test_ops`, the likely tasks are:

- verify operator correctness on point cloud examples
- inspect output geometry and neighborhood selection
- compare CPU / GPU behavior
- add new demo or runner coverage when a new `point_ops` operator is added
- adjust output formatting or visualization conventions
- reproduce bugs outside the full model stack

## 11. Relationship To Other Repositories

- `point_ops` is the core dependency and the main target of validation here.
- `point_model` consumes `point_ops` at model level, after operator-level behavior is validated here.
- `registration_acc` is not directly exercised here yet, but future accelerator-aligned validation may eventually extend into this repo.

## 12. Practical Guidance For New Codex Sessions

If the user mentions:

- operator outputs
- `.ply` visualization
- 3DMatch debug
- FPS / KNN / Ball Query scripts

start in `test_ops`.

Read these files first:

- `test_ops/command.md`
- `test_ops/run_3dmatch.sh`
- the specific `run_*` script relevant to the operator in question

If behavior changes in scripts, CLI flags, or output directories, `command.md` should usually be updated as well.
