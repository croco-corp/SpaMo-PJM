#!/usr/bin/env bash
# Wait for v9 to finish, then run v11 → v12
# Usage: bash scripts/queue_v11_v12.sh &
set -euo pipefail

cd /home/croco/SpaMo-PJM
mkdir -p logs

echo "=== queue_v11_v12 started $(date -Iseconds) ==="

echo "--- Waiting for v9 to finish ---"
while pgrep -f "finetune_v9_handvit" > /dev/null; do
    sleep 60
done
echo "--- v9 done $(date -Iseconds) ---"

echo "--- Step 1: v11 (vit + mae + hand_vit) ---"
uv run python main.py -c configs/finetune_v11_triple.yaml -e retrieval \
    2>&1 | tee logs/run_v11.log
echo "--- v11 done $(date -Iseconds) ---"

echo "--- Step 2: v12 (vit + mae + hand_vit + mediapipe) ---"
uv run python main.py -c configs/finetune_v12_quad.yaml -e retrieval \
    2>&1 | tee logs/run_v12.log
echo "--- v12 done $(date -Iseconds) ---"

echo "=== All done ==="
