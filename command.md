# test_ops Command Reference

This document summarizes how to run the current test scripts and their options.

## 1) run_3dmatch.sh

Runs FPS, KNN and/or Ball Query end-to-end:
- Uses **pytorch3d** env to generate NPZ intermediates
- Uses **gedi** env (Open3D) to convert NPZ -> PLY

**Usage**
```
bash test_ops/run_3dmatch.sh -n fps
bash test_ops/run_3dmatch.sh -n knn
bash test_ops/run_3dmatch.sh -n ball_query
bash test_ops/run_3dmatch.sh -n all
```

**Options**
- `-n` operator to run: `fps` | `knn` | `ball_query` | `all` (default: all)
- `-d` dataset directory (default: `/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen`)
- `-s` FPS num_samples (default: `2048`)
- `-k` KNN K (default: `16`)
- `-r` BallQuery radius (default: `0.05`)
- `-m` BallQuery max neighbors (default: `64`)
- `-p` num centers for KNN/BallQuery (default: `5`)
- `-v` voxel size for conversion (default: `0.01`)
- `-V` disable voxel downsample
- `-D` enable VSCode debug (debugpy attach)
- `-P` debugpy port (default: `5678`)
- `-g` force GPU (default)
- `-c` force CPU

**Examples**
```
# FPS only, GPU, voxel 0.01
bash test_ops/run_3dmatch.sh -n fps -g

# KNN only, CPU, 10 centers, no voxel downsample
bash test_ops/run_3dmatch.sh -n knn -c -V -p 10

# BallQuery only, custom radius/neighbors/centers
bash test_ops/run_3dmatch.sh -n ball_query -r 0.05 -m 64 -p 5

# BallQuery with VSCode attach (debugpy on :5678)
bash test_ops/run_3dmatch.sh -n ball_query -D -r 0.05 -m 64 -p 5

# BallQuery with custom debug port
bash test_ops/run_3dmatch.sh -n ball_query -D -P 5679 -r 0.05 -m 64 -p 5

# Both, custom samples/K
bash test_ops/run_3dmatch.sh -n all -s 1024 -k 8
```

---

## 2) run_fps_3dmatch.py

Runs FPS on a single point cloud and saves:
- `fps_source.npz`
- `fps_samples.npz`

and (after convert_ply) produces:
- `fps_source.ply`
- `fps_samples.ply`

**Usage**
```
python -m test_ops.run_fps_3dmatch
python -m test_ops.run_fps_3dmatch --input /home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen --num-samples 2048
```

**Options**
- `--input` dataset dir or point cloud file (default: `/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen`)
- `--num-samples` FPS sample count (default: `2048`)
- `--deterministic` use deterministic start
- `--device` `gpu` | `cpu` (default: `gpu`)

**Outputs**
- intermediates: `test_ops/intermediate/fps/*.npz`
- ply (after convert): `test_ops/result/fps/*.ply`

---

## 3) run_knn_3dmatch.py

Runs KNN on a single point cloud:
- random selects multiple centers
- saves source / sample / center as NPZ

**Usage**
```
python -m test_ops.run_knn_3dmatch
python -m test_ops.run_knn_3dmatch --input /home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen --k 16 --num-centers 5
```

**Options**
- `--input` dataset dir or point cloud file (default: `/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen`)
- `--k` KNN neighbors (default: `16`)
- `--num-centers` number of random centers (default: `5`)
- `--seed` random seed (default: `0`)
- `--device` `gpu` | `cpu` (default: `gpu`)

**Outputs**
- intermediates: `test_ops/intermediate/knn/*.npz`
- ply (after convert): `test_ops/result/knn/*.ply`

---

## 4) run_ball_query_3dmatch.py

Runs Ball Query on a single point cloud:
- random selects multiple centers
- saves source / sample / center (with sphere points for radius) as NPZ

**Usage**
```
python -m test_ops.run_ball_query_3dmatch
python -m test_ops.run_ball_query_3dmatch --input /home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen --radius 0.05 --max-neighbors 64 --num-centers 5
```

**Options**
- `--input` dataset dir or point cloud file (default: `/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen`)
- `--radius` BallQuery radius (default: `0.05`)
- `--max-neighbors` max neighbors per center (default: `64`)
- `--num-centers` number of random centers (default: `5`)
- `--seed` random seed (default: `0`)
- `--device` `gpu` | `cpu` (default: `gpu`)
- `--sphere-points` points per center to draw the radius sphere (default: `200`)
- `--center-dup` center point duplication to look larger (default: `30`)

**Outputs**
- intermediates: `test_ops/intermediate/ball_query/*.npz`
- ply (after convert): `test_ops/result/ball_query/*.ply`

## 5) convert_ply.py

Converts NPZ intermediates to PLY using Open3D.

**Usage**
```
python -m test_ops.convert_ply --input test_ops/intermediate/fps --output test_ops/result/fps
python -m test_ops.convert_ply --input test_ops/intermediate/knn --output test_ops/result/knn
python -m test_ops.convert_ply --input test_ops/intermediate/ball_query --output test_ops/result/ball_query
```

**Options**
- `--input` directory with `.npz` files
- `--output` output directory for `.ply`
- `--voxel-size` voxel downsample size (default: `0.01`)
- `--no-voxel` disable voxel downsample

**Color rules**
- FPS:
  - `fps_source` -> GeDi blue
  - `fps_samples` -> GeDi yellow
- KNN:
  - `knn_source` -> GeDi blue
  - `knn_sample` -> green
  - `knn_center` -> GeDi yellow
- Ball Query:
  - `ball_query_source` -> GeDi blue
  - `ball_query_sample` -> green
  - `ball_query_center` -> GeDi yellow

---

## Notes
- `plyfile` is required to read binary PLY in the run scripts:
  - `pip install plyfile`
- Run scripts generate NPZ in `test_ops/intermediate/...` then convert using `convert_ply.py`.
- VSCode attach: run `bash test_ops/run_3dmatch.sh -n <op> -D`, then in VSCode use a Debug configuration to attach to port 5678.


---

## Demo scripts (no CLI args)

These demos are designed to be run directly without command-line parameters. Edit the
configuration constants at the top of each file to change test settings.

- `test_ops/run_fps_demo.py`
- `test_ops/run_knn_demo.py`
- `test_ops/run_mlp_demo.py`
- `test_ops/run_ball_query_demo.py`
