#!/usr/bin/env bash
set -euo pipefail

# 大学サーバーでの学習起動用ラッパー
# 使い方:
#   chmod +x scripts/train_on_server.sh
#   ./scripts/train_on_server.sh
#
# 環境変数で上書き可能:
#   DEEPLAB_ROOT, TRAIN_MODE, IMAGES_DIR, MASKS_DIR, CHECKPOINT_DIR,
#   BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, IMG_SIZE, EARLY_STOP_PATIENCE, CKPT_NAME,
#   RUN_FOREGROUND

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEEPLAB_ROOT="${DEEPLAB_ROOT:-$HOME/DeepLabV3Plus-Pytorch}"
TRAIN_MODE="${TRAIN_MODE:-smp50}" # smp50 or smp34

IMAGES_DIR="${IMAGES_DIR:-$DEEPLAB_ROOT/train/datasets/data_makassar/images}"
MASKS_DIR="${MASKS_DIR:-$DEEPLAB_ROOT/train/datasets/data_makassar/masks}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/models}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/static/outputs/train_logs}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

IMG_SIZE="${IMG_SIZE:-320}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
CKPT_NAME="${CKPT_NAME:-deeplabv3plus_road_resnet50_server.pth}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}" # 1: 前面実行, 0: nohupバックグラウンド

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$LOG_DIR/train_${TRAIN_MODE}_${timestamp}.log"

if [[ "$TRAIN_MODE" == "smp50" ]]; then
  train_script="$DEEPLAB_ROOT/train/scripts/train_road_resnet50_smp.py"
  cmd=(
    python3 "$train_script"
    --images_dir "$IMAGES_DIR"
    --masks_dir "$MASKS_DIR"
    --img_size "$IMG_SIZE"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --num_workers "$NUM_WORKERS"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --ckpt_name "$CKPT_NAME"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
  )
elif [[ "$TRAIN_MODE" == "smp34" ]]; then
  train_script="$DEEPLAB_ROOT/train/scripts/train_road_smp.py"
  cmd=(
    python3 "$train_script"
    --images_dir "$IMAGES_DIR"
    --masks_dir "$MASKS_DIR"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --num_workers "$NUM_WORKERS"
    --checkpoint_dir "$CHECKPOINT_DIR"
  )
else
  echo "[ERROR] TRAIN_MODE は smp50 か smp34 を指定してください。現在: $TRAIN_MODE" >&2
  exit 1
fi

echo "[INFO] command: ${cmd[*]}"
echo "[INFO] log: $log_path"
if [[ "$RUN_FOREGROUND" == "1" ]]; then
  echo "[INFO] running in foreground mode"
  "${cmd[@]}" 2>&1 | tee "$log_path"
else
  nohup "${cmd[@]}" >"$log_path" 2>&1 &
  pid=$!
  echo "[INFO] started. PID=$pid"
  echo "[INFO] monitor: tail -f $log_path"
fi
