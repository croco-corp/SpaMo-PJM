#!/usr/bin/env bash
# Po zakończeniu v23 odpala kolejno: v22c (paper-faithful dual) + v22d (paper-faithful quad).
# Cel: zweryfikować czy paper config zlikwiduje gap z paper-reported BLEU 24.32.

set -e
set -o pipefail
cd /home/croco/SpaMo-PJM
source .venv/bin/activate

LOG_DIR="logs/queue_v22cd_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== START QUEUE v22c + v22d (paper-faithful, po v23) ==="

log "--- v22c: paper-faithful DUAL (ViT+MAE), DE→DE ---"
python main.py \
    -c configs/pretrain_v22c_phoenix_paper_dual.yaml \
    -n pretrain_v22c_phoenix_paper_dual \
    -e bleu \
    --tags pretrain phoenix-2014t dual de-de paper-faithful cosine-warmup \
    2>&1 | tee "$LOG_DIR/v22c_train.log"
log "v22c OK"

log "--- v22d: paper-faithful QUAD (ViT+MAE+hand_ViT+MediaPipe), DE→DE ---"
python main.py \
    -c configs/pretrain_v22d_phoenix_paper_quad.yaml \
    -n pretrain_v22d_phoenix_paper_quad \
    -e bleu \
    --tags pretrain phoenix-2014t quad de-de paper-faithful cosine-warmup \
    2>&1 | tee "$LOG_DIR/v22d_train.log"
log "v22d OK"

log "=== QUEUE v22cd DONE ==="
log "Best ckpt do raportu:"
ls -t logs/*pretrain_v22c_phoenix_paper_dual/checkpoints/*bleu4*.ckpt 2>/dev/null | head -1 | tee -a "$QLOG"
ls -t logs/*pretrain_v22d_phoenix_paper_quad/checkpoints/*bleu4*.ckpt 2>/dev/null | head -1 | tee -a "$QLOG"
