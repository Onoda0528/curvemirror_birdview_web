#!/usr/bin/env bash
#SBATCH --job-name=cm_quad_train
#SBATCH --output=cm_quad_train_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# 必要なら指定:
#SBATCH --partition=gpu

set -euo pipefail

# 使い方:
#   sbatch scripts/slurm_train_quadpoint.sbatch
#
# 必要に応じてこのファイル内の変数を編集してください。
# もしくは submit 時に上書きできます:
#   sbatch --export=ALL,IMAGES_DIR=/path/to/images,LABELS_CSV=/path/to/labels.csv scripts/slurm_train_quadpoint.sbatch

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/curvemirror_birdview_web}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

IMAGES_DIR="${IMAGES_DIR:-$PROJECT_ROOT/data/quadpoint/images}"
LABELS_CSV="${LABELS_CSV:-$PROJECT_ROOT/data/quadpoint/labels.csv}"
MASKS_DIR="${MASKS_DIR:-}" # RGB+Mask の場合のみ指定
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/models}"
CKPT_NAME="${CKPT_NAME:-quadpoint_best.pth}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
INPUT_SIZE="${INPUT_SIZE:-384}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export IMAGES_DIR LABELS_CSV MASKS_DIR CHECKPOINT_DIR CKPT_NAME
export BATCH_SIZE EPOCHS LR NUM_WORKERS INPUT_SIZE VAL_RATIO SEED
export RUN_FOREGROUND=1

./scripts/train_quad_on_server.sh
