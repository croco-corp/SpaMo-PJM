#!/usr/bin/env bash
# Po zakończeniu v22cd:
#   1. Verify multilang H5 (translate już lub równolegle leci).
#   2. Pick best ckpt z v22c/v22d wg BLEU.
#   3. v23b: PJM-MS finetune EN target, paper-faithful.
#   4. v24: PJM-MS finetune PL target (native), paper-faithful.

set -e
set -o pipefail
cd /home/croco/SpaMo-PJM
source .venv/bin/activate

LOG_DIR="logs/queue_v23b_v24_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== START QUEUE v23b + v24 (po v22cd) ==="

# --- 1. Multilang sanity ---
MULTI_H5=features/texts_multilang_pjm.h5
log "--- 1. multilang H5 sanity ($MULTI_H5) ---"
if [ ! -f "$MULTI_H5" ]; then
    log "BŁĄD: brak $MULTI_H5 — translate jeszcze nie skończony lub nie wystartował"
    exit 1
fi
python -c "
import h5py
with h5py.File('$MULTI_H5') as f:
    keys = list(f.keys())
    print(f'multilang keys: {len(keys):,}')
    if len(keys) < 10000:
        print('FAIL: za mało kluczy (<10K), translation niedokończona')
        import sys; sys.exit(1)
    sample = f[keys[0]]
    print(f'sample [{keys[0]}]: pl={sample[\"pl\"][()].decode()!r}')
    print(f'  en={sample[\"en\"][()].decode()!r}')
    print(f'  fr={sample[\"fr\"][()].decode()!r}')
    print(f'  es={sample[\"es\"][()].decode()!r}')
" 2>&1 | tee "$LOG_DIR/01_multilang_check.log"

# --- 2. Pick best v22cd ckpt ---
log "--- 2. pick best ckpt z v22c/v22d ---"
BEST_CKPT=$(ls -1 logs/*pretrain_v22{c,d}_phoenix_paper_*/checkpoints/*bleu4*.ckpt 2>/dev/null \
    | awk -F'bleu4=' '{print $2 "|" $0}' \
    | sort -t. -k1,1nr -k2,2nr | head -1 | cut -d'|' -f2)
if [ -z "$BEST_CKPT" ]; then
    log "BŁĄD: brak ckpt z v22c/v22d"
    exit 1
fi
log "best v22cd ckpt: $BEST_CKPT"

# --- 3. v23b: paper-faithful PJM EN ---
log "--- 3. v23b: PJM-MS EN target, paper-faithful ---"
python main.py \
    -c configs/finetune_v23b_pjm_paper_en.yaml \
    -n finetune_v23b_pjm_paper_en \
    --ckpt "$BEST_CKPT" \
    -e bleu \
    --tags finetune pjm-ms phoenix-init quad paper-faithful en-target \
    2>&1 | tee "$LOG_DIR/03_v23b.log"
log "v23b OK"

# v24 (PJM PL native target) odpuszczone — flan-T5 jest English-centric, expected weak.
# Config configs/finetune_v24_pjm_paper_pl.yaml zostaje na dysku jako future work.

log "=== QUEUE DONE (tylko v23b — v24 PL pominięte) ==="
log "Best ckpt v23b:"
ls -t logs/*finetune_v23b_pjm_paper_en/checkpoints/*bleu4*.ckpt 2>/dev/null | head -1 | tee -a "$QLOG"
