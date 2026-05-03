#!/usr/bin/env bash
# Kolejka quad-pretrain SpaMo na PHOENIX-2014T:
#   1. smoke test (fast_dev_run, dual)
#   2. hand_ViT extraction
#   3. MediaPipe extraction
#   4. coverage sanity-check
#   5. quad pretrain (v22)
#
# Każdy krok ma osobny log; `set -e` przerywa kolejkę przy pierwszym błędzie.

set -e
set -o pipefail
cd /home/croco/SpaMo-PJM
source .venv/bin/activate

LOG_DIR="logs/queue_v22_phoenix_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== START QUEUE v22 (logdir=$LOG_DIR) ==="

log "--- 1. smoke test (fast_dev_run, dual streams) ---"
python main.py \
    -c configs/pretrain_v22_phoenix_dual_smoketest.yaml \
    -n smoketest_v22_dual \
    --fast_dev_run \
    2>&1 | tee "$LOG_DIR/01_smoketest.log"
log "smoke test OK"

log "--- 2. hand_ViT extraction (CLIP ViT-L/14 on hand crops) ---"
python scripts/hand_crop_vit_extract_p14t.py \
    --num-workers 22 \
    2>&1 | tee "$LOG_DIR/02_hand_vit.log"
log "hand_ViT extraction OK"

log "--- 3. MediaPipe extraction (pose + hands) ---"
python scripts/mediapipe_extract_p14t.py \
    --num-workers 22 \
    2>&1 | tee "$LOG_DIR/03_mediapipe.log"
log "MediaPipe extraction OK"

log "--- 4. coverage sanity-check ---"
python -c "
import h5py
hand = set(h5py.File('features/hand_vit_feat_p14t.h5', 'r').keys())
mp = set(h5py.File('features/mediapipe_feat_p14t.h5', 'r').keys())
inter = hand & mp
print(f'hand_ViT keys: {len(hand)}')
print(f'MediaPipe keys: {len(mp)}')
print(f'intersection: {len(inter)}')
print(f'hand_ViT-only: {len(hand - mp)}')
print(f'MediaPipe-only: {len(mp - hand)}')
" 2>&1 | tee "$LOG_DIR/04_coverage.log"

log "--- 5. quad pretrain v22 ---"
python main.py \
    -c configs/pretrain_v22_phoenix.yaml \
    -n pretrain_v22_phoenix \
    -e bleu \
    --tags pretrain phoenix-2014t quad de-en \
    2>&1 | tee "$LOG_DIR/05_train.log"
log "training OK"

log "=== QUEUE DONE ==="
