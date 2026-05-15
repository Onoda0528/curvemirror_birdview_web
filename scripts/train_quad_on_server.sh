#!/usr/bin/env bash
set -euo pipefail

# 大学サーバーでの4点学習起動用ラッパー
# 使い方:
#   chmod +x scripts/train_quad_on_server.sh
#   ./scripts/train_quad_on_server.sh
#
# 環境変数で上書き可能:
#   IMAGES_DIR, LABELS_CSV, MASKS_DIR, CHECKPOINT_DIR, CKPT_NAME,
#   BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, INPUT_SIZE, VAL_RATIO, SEED, LOG_DIR,
#   RUN_FOREGROUND
#
# RGB+Maskで学習したい場合:
#   MASKS_DIR を設定する
#   例: MASKS_DIR=/path/to/masks ./scripts/train_quad_on_server.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_quad_points.py"

IMAGES_DIR="${IMAGES_DIR:-$PROJECT_ROOT/data/quadpoint/images}"
LABELS_CSV="${LABELS_CSV:-$PROJECT_ROOT/data/quadpoint/labels.csv}"
MASKS_DIR="${MASKS_DIR:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/models}"
CKPT_NAME="${CKPT_NAME:-quadpoint_best.pth}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/static/outputs/train_logs}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
INPUT_SIZE="${INPUT_SIZE:-384}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}" # 1: 前面実行, 0: nohupバックグラウンド

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$LOG_DIR/quad_train_${timestamp}.log"

cmd=(
  python3 -u "$TRAIN_SCRIPT"
  --images_dir "$IMAGES_DIR"
  --labels_csv "$LABELS_CSV"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --ckpt_name "$CKPT_NAME"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --num_workers "$NUM_WORKERS"
  --input_size "$INPUT_SIZE"
  --val_ratio "$VAL_RATIO"
  --seed "$SEED"
)

if [[ -n "$MASKS_DIR" ]]; then
  cmd+=(--masks_dir "$MASKS_DIR")
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
