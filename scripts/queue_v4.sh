#!/bin/bash
# Waits for v3 training to finish, then launches v4 (contrastive-only, alignment gap).
# Usage: bash scripts/queue_v4.sh &

V3_PATTERN="main.py -c configs/finetune_v3_diff_lr.yaml"
V4_CONFIG="configs/finetune_v4_contrastive.yaml"
V4_LOG="logs/run_v4.log"

echo "=== queue_v4 started $(date -Iseconds) ==="
echo "Waiting for v3 (pattern: $V3_PATTERN) ..."

while pgrep -f "$V3_PATTERN" > /dev/null; do
    sleep 60
done

echo "v3 finished at $(date -Iseconds). Launching v4 ..."
cd /home/croco/SpaMo-PJM
nohup uv run python main.py -c "$V4_CONFIG" -e retrieval > "$V4_LOG" 2>&1 &
V4_PID=$!
echo "v4 started (PID $V4_PID) — log: $V4_LOG"
