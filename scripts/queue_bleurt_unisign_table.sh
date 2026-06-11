#!/bin/bash
# Post-hoc BLEURT-20 for Uni-Sign runs in the poster table that are missing it,
# plus a retry of the MS scratch eval (which crashed earlier on transient wandb outage).
# Run: nohup bash scripts/queue_bleurt_unisign_table.sh > queue_bleurt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."

LOG=queue_bleurt_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

UNISIGN_DIR=/home/croco/Uni-Sign

echo "=== BLEURT-20 post-hoc + MS scratch retry: $(date) ==="

source .venv/bin/activate

# 1. BLEURT-20 for the six existing phase3 evals (CSL-News / How2Sign / OpenASL × MS/SI).
for run in pjm_phase3_full_ms pjm_phase3_full_si \
           pjm_phase3_full_ms_how2sign pjm_phase3_full_si_how2sign \
           pjm_phase3_full_ms_openasl pjm_phase3_full_si_openasl; do
    d="$UNISIGN_DIR/out/eval_$run"
    if [ -f "$d/bleurt_BLEURT-20.json" ]; then
        echo "SKIP $run — bleurt already present"
        continue
    fi
    if [ ! -f "$d/test_tmp_refs.txt" ]; then
        echo "SKIP $run — no test_tmp_refs.txt"
        continue
    fi
    echo "=== BLEURT-20: $run ==="
    python scripts/bleurt_unisign.py "$d" 2>&1 | tee -a "$d/log.txt" || echo "(bleurt failed for $run, continuing)"
done

deactivate

# 2. Retry MS scratch eval (was killed by wandb GraphQL transient timeout 22:21–22:24).
US_CKPT="$UNISIGN_DIR/out/pjm_phase3_full_ms_scratch/best_checkpoint.pth"
US_OUT="$UNISIGN_DIR/out/eval_pjm_phase3_full_ms_scratch"

if [ -f "$US_OUT/test_tmp_refs.txt" ]; then
    echo "SKIP MS scratch eval — test_tmp_refs.txt already present"
else
    echo "=== Uni-Sign MS scratch eval (retry): $(date) ==="
    rm -f "$US_OUT/log.txt"
    mkdir -p "$US_OUT"
    PORT=29501
    while ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT$"; do
        PORT=$((PORT + 1))
    done
    echo "    using master port: $PORT"
    ( cd "$UNISIGN_DIR" && \
      source .venv/bin/activate && \
      torchrun --nproc_per_node=1 --master_port=$PORT fine_tuning.py \
          --eval \
          --finetune "$US_CKPT" \
          --dataset PJM \
          --task SLT \
          --pjm_split ms \
          --bertscore \
          --num_examples 5 \
          --batch-size 8 \
          --num_workers 4 \
          --output_dir "$US_OUT" \
          --wandb \
          --wandb_project uni-sign-eval \
          2>&1 | tee "$US_OUT/log.txt" ) || echo "(uni-sign eval failed)"
fi

if [ -f "$US_OUT/test_tmp_refs.txt" ] && [ ! -f "$US_OUT/bleurt_BLEURT-20.json" ]; then
    echo "=== BLEURT-20 post-hoc: pjm_phase3_full_ms_scratch ==="
    source .venv/bin/activate
    python scripts/bleurt_unisign.py "$US_OUT" 2>&1 | tee -a "$US_OUT/log.txt" || echo "(bleurt failed)"
    deactivate
fi

echo ""
echo "=== Summary ==="
for run in pjm_phase3_full_ms pjm_phase3_full_si \
           pjm_phase3_full_ms_how2sign pjm_phase3_full_si_how2sign \
           pjm_phase3_full_ms_openasl pjm_phase3_full_si_openasl \
           pjm_phase3_full_ms_scratch; do
    f="$UNISIGN_DIR/out/eval_$run/bleurt_BLEURT-20.json"
    if [ -f "$f" ]; then
        mean=$(python -c "import json; print(f'{json.load(open(\"$f\"))[\"mean\"]:.4f}')")
        echo "  $run: BLEURT-20=$mean"
    else
        echo "  $run: no bleurt"
    fi
done
echo "=== Done: $(date) ==="
