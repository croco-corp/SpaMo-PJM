#!/usr/bin/env bash
# Czeka na zakończenie v16, potem odpala v17 z najlepszym checkpointem v16
set -euo pipefail

cd /home/croco/SpaMo-PJM
mkdir -p logs

echo "=== queue_v17 started $(date -Iseconds) ==="

echo "--- Czekam na zakończenie v16 ---"
while pgrep -f "finetune_v16_combined" > /dev/null; do
    sleep 60
done
echo "--- v16 done $(date -Iseconds) ---"

# Znajdź najlepszy checkpoint v16
V16_CKPT=$(ls logs/*/checkpoints/*.ckpt 2>/dev/null \
    | grep "v16\|finetune_v16" \
    | sort -t= -k3 -rn \
    | head -1)

if [ -z "$V16_CKPT" ]; then
    echo "ERROR: Nie znaleziono checkpointu v16!" >&2
    exit 1
fi

echo "--- Checkpoint v16: $V16_CKPT ---"
echo "--- Step 1: v17 (generation, alpha=0.1) ---"
uv run python main.py \
    -c configs/finetune_v17_generation.yaml \
    -e bleu \
    --ckpt "$V16_CKPT" \
    2>&1 | tee logs/run_v17.log

echo "--- v17 done $(date -Iseconds) ==="
