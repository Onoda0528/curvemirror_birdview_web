#!/usr/bin/env bash
#SBATCH --job-name=cm_seg_train
#SBATCH --output=cm_seg_train_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# 必要なら指定:
#SBATCH --partition=gpu

set -euo pipefail

# 使い方:
#   sbatch scripts/slurm_train_segmentation.sbatch
#
# 必要に応じてこのファイル内の変数を編集してください。
# もしくは submit 時に上書きできます:
#   sbatch --export=ALL,IMAGES_DIR=/path/to/images,MASKS_DIR=/path/to/masks scripts/slurm_train_segmentation.sbatch

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/curvemirror_birdview_web}"
DEEPLAB_ROOT="${DEEPLAB_ROOT:-$HOME/DeepLabV3Plus-Pytorch}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

IMAGES_DIR="${IMAGES_DIR:-$DEEPLAB_ROOT/train/datasets/data_makassar/images}"
MASKS_DIR="${MASKS_DIR:-$DEEPLAB_ROOT/train/datasets/data_makassar/masks}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/models}"
TRAIN_MODE="${TRAIN_MODE:-smp50}"
CKPT_NAME="${CKPT_NAME:-deeplabv3plus_road_resnet50_server.pth}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMG_SIZE="${IMG_SIZE:-320}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

export DEEPLAB_ROOT
export TRAIN_MODE IMAGES_DIR MASKS_DIR CHECKPOINT_DIR CKPT_NAME
export BATCH_SIZE EPOCHS LR NUM_WORKERS IMG_SIZE EARLY_STOP_PATIENCE
export RUN_FOREGROUND=1

./scripts/train_on_server.sh
