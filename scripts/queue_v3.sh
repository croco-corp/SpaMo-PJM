#!/bin/bash
# Waits for v2 training to finish, then launches v3 (diff LR experiment).
# Usage: bash scripts/queue_v3.sh &

V2_PATTERN="main.py -c configs/finetune.yaml"
V3_CONFIG="configs/finetune_v3_diff_lr.yaml"
V3_LOG="logs/run_v3.log"

echo "=== queue_v3 started $(date -Iseconds) ==="
echo "Waiting for v2 (pattern: $V2_PATTERN) ..."

while pgrep -f "$V2_PATTERN" > /dev/null; do
    sleep 60
done

echo "v2 finished at $(date -Iseconds). Launching v3 ..."
cd /home/croco/SpaMo-PJM
nohup uv run python main.py -c "$V3_CONFIG" -e bleu > "$V3_LOG" 2>&1 &
V3_PID=$!
echo "v3 started (PID $V3_PID) — log: $V3_LOG"
