#!/usr/bin/env bash
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

# Usage:
#   ./run_3dmatch.sh -n fps         # run only FPS
#   ./run_3dmatch.sh -n knn         # run only KNN
#   ./run_3dmatch.sh -n ball_query  # run only Ball Query
#   ./run_3dmatch.sh -n all         # run all (default)
#
# Optional args:
#   -d DATA_DIR     dataset dir (default: /home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen)
#   -s NUM_SAMPLES  (FPS) default: 2048
#   -k K            (KNN) default: 16
#   -r RADIUS       (BallQuery) default: 0.05
#   -m MAX_NEIGHBORS (BallQuery) default: 64
#   -p NUM_CENTERS  (KNN/BallQuery) default: 5
#   -v VOXEL_SIZE   (convert) default: 0.01
#   -V              disable voxel downsample
#   -D              enable VSCode debug (debugpy attach)
#   -P PORT         debugpy listen port (default: 5678)
#   -g              use GPU (default)
#   -c              use CPU

DATA_DIR="/home/lfc/workspace/dataset/3DMatch/7-scenes-redkitchen"
NUM_SAMPLES=2048
K=16
RADIUS=0.05
MAX_NEIGHBORS=64
NUM_CENTERS=5
VOXEL_SIZE=0.01
NO_VOXEL=0
DEBUG=0
DEBUG_PORT=5678
DEVICE="gpu"
OP="all"

usage() {
  cat <<'EOF'
Usage: bash test_ops/run_3dmatch.sh [options]

Options:
  -n OP            operator: fps | knn | ball_query | all (default: all)
  -d DATA_DIR      dataset dir
  -s NUM_SAMPLES   FPS num_samples
  -k K             KNN K
  -r RADIUS        BallQuery radius
  -m MAX_NEIGHBORS BallQuery max neighbors
  -p NUM_CENTERS   num centers for KNN/BallQuery
  -v VOXEL_SIZE    voxel size for conversion
  -V               disable voxel downsample
  -D               enable VSCode debug (debugpy attach)
  -P PORT          debugpy listen port
  -g               use GPU (default)
  -c               use CPU
  -h               show this help
EOF
}

while getopts "n:d:s:k:r:m:p:v:VDgcP:h" opt; do
  case $opt in
    n) OP="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    s) NUM_SAMPLES="$OPTARG" ;;
    k) K="$OPTARG" ;;
    r) RADIUS="$OPTARG" ;;
    m) MAX_NEIGHBORS="$OPTARG" ;;
    p) NUM_CENTERS="$OPTARG" ;;
    v) VOXEL_SIZE="$OPTARG" ;;
    V) NO_VOXEL=1 ;;
    D) DEBUG=1 ;;
    P) DEBUG_PORT="$OPTARG" ;;
    g) DEVICE="gpu" ;;
    c) DEVICE="cpu" ;;
    h) usage; exit 0 ;;
    *) echo "Invalid option"; usage; exit 1 ;;
  esac
done


# Initialize conda
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "conda not found in PATH" >&2
  exit 1
fi

apply_voxel_args() {
  if [ "$NO_VOXEL" -eq 1 ]; then
    echo "--no-voxel"
  else
    echo "--voxel-size $VOXEL_SIZE"
  fi
}


run_py() {
  local module="$1"
  shift
  if [ "$DEBUG" -eq 1 ]; then
    if python - <<'PY' >/dev/null 2>&1
import debugpy
PY
    then
      if ! python - "$DEBUG_PORT" <<'PY' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
      then
        echo "[ERROR] debug port $DEBUG_PORT is already in use. Use -P to choose another port or stop the existing debug session." >&2
        exit 1
      fi
      python -m debugpy --listen "$DEBUG_PORT" --wait-for-client -m "$module" "$@"
    else
      echo "[WARN] debugpy not installed in current env; running without VSCode attach." >&2
      python -m "$module" "$@"
    fi
  else
    python -m "$module" "$@"
  fi
}

run_fps() {
  conda activate pytorch3d
  run_py test_ops.run_fps_3dmatch --input "$DATA_DIR" --num-samples "$NUM_SAMPLES" --device "$DEVICE"
  conda activate gedi
  python -m test_ops.convert_ply --input test_ops/intermediate/fps --output test_ops/result/fps $(apply_voxel_args)
}

run_knn() {
  conda activate pytorch3d
  run_py test_ops.run_knn_3dmatch --input "$DATA_DIR" --k "$K" --num-centers "$NUM_CENTERS" --device "$DEVICE"
  conda activate gedi
  python -m test_ops.convert_ply --input test_ops/intermediate/knn --output test_ops/result/knn $(apply_voxel_args)
}

run_ball_query() {
  conda activate pytorch3d
  run_py test_ops.run_ball_query_3dmatch --input "$DATA_DIR" --radius "$RADIUS" --max-neighbors "$MAX_NEIGHBORS" --num-centers "$NUM_CENTERS" --device "$DEVICE"
  conda activate gedi
  python -m test_ops.convert_ply --input test_ops/intermediate/ball_query --output test_ops/result/ball_query $(apply_voxel_args)
}

case "$OP" in
  fps) run_fps ;;
  knn) run_knn ;;
  ball_query) run_ball_query ;;
  all) run_fps; run_knn; run_ball_query ;;
  *)
    echo "Unknown operator: $OP" >&2
    usage
    exit 1
    ;;
 esac

echo "Done. Outputs are under test_ops/result/{fps,knn,ball_query}."