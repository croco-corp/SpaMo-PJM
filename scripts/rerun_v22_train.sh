#!/usr/bin/env bash
# Re-run tylko kroku 5 (v22 quad pretrain) z poprawnym `-e bleu`.
# Features p14t H5/npy są już wyekstraktowane — pomijamy kroki 1-4.
# Po sukcesie chainuje continue_v22_to_v23.sh.

set -e
set -o pipefail
cd /home/croco/SpaMo-PJM
source .venv/bin/activate

LOG_DIR="logs/rerun_v22_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== START RERUN v22 (poprawione -e bleu) ==="

log "--- v22 quad pretrain ---"
python main.py \
    -c configs/pretrain_v22_phoenix.yaml \
    -n pretrain_v22_phoenix \
    -e bleu \
    --tags pretrain phoenix-2014t quad de-en \
    2>&1 | tee "$LOG_DIR/train.log"
log "v22 training OK"

log "--- chain do continue (sanity + finetune v23 + cleanup) ---"
bash scripts/continue_v22_to_v23.sh

log "=== RERUN DONE ==="
