#!/usr/bin/env bash
# v22 i v22b już skończone (ckpt zapisane); v22b cleanup-block padł na buggy
# sanity check (złe nazwy modułów). Odpala TYLKO sanity + finetune v23 +
# cleanup, używając najlepszego ckpt v22 (DE→EN).

set -e
set -o pipefail
cd /home/croco/SpaMo-PJM
source .venv/bin/activate

LOG_DIR="logs/run_v23_only_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== START run_v23_only (sanity + v23 + cleanup) ==="

V22_CKPT=$(ls -t logs/*pretrain_v22_phoenix/checkpoints/*bleu4*.ckpt 2>/dev/null | head -1)
if [ -z "$V22_CKPT" ]; then
    log "BŁĄD: brak ckpt z v22"
    exit 1
fi
log "ckpt: $V22_CKPT"

log "--- sanity check ---"
python - <<PYEOF 2>&1 | tee "$LOG_DIR/sanity.log"
import re, sys, torch
ckpt_path = "$V22_CKPT"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt.get('state_dict', ckpt)
expected_modules = ['spatio_proj', 'spatiotemp_proj', 'aux_proj', 'kp_proj']
missing = [m for m in expected_modules if not any(m in k for k in sd.keys())]
if missing:
    print(f"FAIL: brak modułów: {missing}"); sys.exit(1)
print(f"OK: {expected_modules} obecne")
lora_keys = [k for k in sd.keys() if 'lora_A' in k or 'lora_B' in k]
if not lora_keys:
    print(f"FAIL: brak LoRA"); sys.exit(1)
print(f"OK: {len(lora_keys)} LoRA tensors")
m = re.search(r'bleu4=([\d.]+)', ckpt_path)
if m: print(f"BLEU-4 z ckpt: {m.group(1)}")
print("SANITY OK")
PYEOF

log "--- finetune v23 ---"
python main.py \
    -c configs/finetune_v23_phoenix_init_ms.yaml \
    -n finetune_v23_phoenix_init_ms \
    --ckpt "$V22_CKPT" \
    -e bleu \
    --tags finetune pjm-ms phoenix-init quad \
    2>&1 | tee "$LOG_DIR/finetune_v23.log"
log "v23 OK"

log "--- cleanup extracted frames ---"
if [ -d /home/croco/data/PHOENIX-2014-T-release-v3 ]; then
    SIZE=$(du -sh /home/croco/data/PHOENIX-2014-T-release-v3 | awk '{print $1}')
    rm -rf /home/croco/data/PHOENIX-2014-T-release-v3
    log "removed: PHOENIX-2014-T-release-v3/ ($SIZE)"
fi
rm -f /home/croco/SpaMo-PJM/features/mediapipe_feat_p14t.h5.part* 2>/dev/null || true
df -h /home/croco/SpaMo-PJM | tail -1 | tee -a "$QLOG"

log "=== DONE ==="
