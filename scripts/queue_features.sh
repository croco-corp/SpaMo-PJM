#!/usr/bin/env bash
# Run both feature extraction scripts in parallel:
#   - MediaPipe Holistic (CPU, multi-process) → features/mediapipe_feat_pjm.h5
#   - Hand-crop ViT (GPU)                     → features/hand_vit_feat_pjm.h5
#
# Usage: ./scripts/queue_features.sh [--dry-run] [--num-workers N]
set -euo pipefail

DRY_RUN=""
NUM_WORKERS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN="--dry-run" ;;
        --num-workers) NUM_WORKERS="--num-workers $2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== Starting MediaPipe extraction (CPU) in background ==="
uv run python scripts/mediapipe_extract.py \
    --tars-dir /home/croco/CrocoSign/data/pjm_segments \
    --crop-params-path crop_params/crop_params.lmdb \
    --output features/mediapipe_feat_pjm.h5 \
    $NUM_WORKERS $DRY_RUN \
    2>&1 | tee "$LOG_DIR/mediapipe_extract.log" &
PID_MP=$!

echo "=== Starting Hand-crop ViT extraction (GPU) in background ==="
uv run python scripts/hand_crop_vit_extract.py \
    --tars-dir /home/croco/CrocoSign/data/pjm_segments \
    --crop-params-path crop_params/crop_params.lmdb \
    --output features/hand_vit_feat_pjm.h5 \
    $NUM_WORKERS $DRY_RUN \
    2>&1 | tee "$LOG_DIR/hand_vit_extract.log" &
PID_VIT=$!

echo "PIDs: mediapipe=$PID_MP  hand_vit=$PID_VIT"
echo "Logs: $LOG_DIR/mediapipe_extract.log  $LOG_DIR/hand_vit_extract.log"

wait $PID_MP && echo "=== MediaPipe done ===" || echo "=== MediaPipe FAILED ==="
wait $PID_VIT && echo "=== Hand-crop ViT done ===" || echo "=== Hand-crop ViT FAILED ==="

echo "=== All done ==="
