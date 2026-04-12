#!/usr/bin/env bash
# Runs feature extraction, then launches v9 training (full-frame ViT + hand-crop ViT).
# Usage: bash scripts/queue_v9.sh [--num-workers N] &

set -euo pipefail

NUM_WORKERS=${1:-12}
V9_CONFIG="configs/finetune_v9_handvit.yaml"
V9_LOG="logs/run_v9.log"

cd /home/croco/SpaMo-PJM
mkdir -p logs

echo "=== queue_v9 started $(date -Iseconds) ==="
echo "Step 1: feature extraction (--num-workers $NUM_WORKERS)"

bash scripts/queue_features.sh --num-workers "$NUM_WORKERS"

echo "=== Features done $(date -Iseconds). Launching v9 training ==="
nohup uv run python main.py -c "$V9_CONFIG" -e retrieval > "$V9_LOG" 2>&1 &
V9_PID=$!
echo "v9 started (PID $V9_PID) — log: $V9_LOG"
