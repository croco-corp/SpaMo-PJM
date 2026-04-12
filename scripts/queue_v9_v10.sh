#!/usr/bin/env bash
# Full pipeline: feature extraction → v9 training → v10 training
#   v9: full-frame ViT + hand-crop ViT
#   v10: full-frame ViT + hand-crop ViT + MediaPipe keypoints
#
# Usage: bash scripts/queue_v9_v10.sh [--num-workers N] &

set -euo pipefail

NUM_WORKERS=${1:-12}
cd /home/croco/SpaMo-PJM
mkdir -p logs

echo "=== queue_v9_v10 started $(date -Iseconds) ==="

# Step 1: feature extraction (both scripts in parallel)
echo "--- Step 1: feature extraction (--num-workers $NUM_WORKERS) ---"
bash scripts/queue_features.sh --num-workers "$NUM_WORKERS"
echo "--- Features done $(date -Iseconds) ---"

# Step 2: v9 training (full-frame ViT + hand-crop ViT)
echo "--- Step 2: launching v9 ---"
uv run python main.py -c configs/finetune_v9_handvit.yaml -e retrieval \
    2>&1 | tee logs/run_v9.log
echo "--- v9 done $(date -Iseconds) ---"

# Step 3: v10 training (+ MediaPipe keypoints)
echo "--- Step 3: launching v10 ---"
uv run python main.py -c configs/finetune_v10_mediapipe.yaml -e retrieval \
    2>&1 | tee logs/run_v10.log
echo "--- v10 done $(date -Iseconds) ---"

echo "=== All done ==="
