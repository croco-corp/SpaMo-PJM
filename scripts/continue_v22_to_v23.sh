#!/usr/bin/env bash
# Kontynuacja po queue_v22_phoenix.sh:
#   6. sanity check najlepszego ckpt z v22 (architektura quad, key coverage, BLEU)
#   7. finetune v23 PJM-MS z initem z v22 (--ckpt → load_pretrained_weights)
#
# Uruchamiany przez watcher po zakończeniu kolejki v22; ABORT jeśli sanity fail.

set -e
set -o pipefail
cd /home/croco/SpaMo-PJM
source .venv/bin/activate

LOG_DIR="logs/continue_v22_to_v23_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== START CONTINUE v22→v23 ==="

# --- 5b. równoległy v22b: Phoenix DE→DE dla porównania z literaturą ---
log "--- 5b. v22b quad pretrain (DE→DE, native Phoenix labels) ---"
python main.py \
    -c configs/pretrain_v22b_phoenix_de_de.yaml \
    -n pretrain_v22b_phoenix_de_de \
    -e bleu \
    --tags pretrain phoenix-2014t quad de-de native-labels \
    2>&1 | tee "$LOG_DIR/05b_train_v22b.log"
log "v22b training OK"

# --- 6. sanity check ---
log "--- 6. sanity check best ckpt z v22 (DE→EN, używany do v23) ---"

V22_CKPT=$(ls -t logs/*pretrain_v22_phoenix*/checkpoints/*bleu4*.ckpt 2>/dev/null | head -1)
if [ -z "$V22_CKPT" ]; then
    log "BŁĄD: nie znaleziono ckpt z v22 w logs/*pretrain_v22_phoenix*/checkpoints/*bleu4*.ckpt"
    log "Listing logs:"
    ls -la logs/ | tail -10 | tee -a "$QLOG"
    exit 1
fi
log "kandydat: $V22_CKPT"

python - <<PYEOF 2>&1 | tee "$LOG_DIR/06_sanity.log"
import re, sys
import torch

ckpt_path = "$V22_CKPT"
print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt.get('state_dict', ckpt)

# 1) Architektura: cztery rzutowania strumieni muszą być obecne (nazwy zgodne z t5_slt.py)
expected_modules = ['spatio_proj', 'spatiotemp_proj', 'aux_proj', 'kp_proj']
missing = [m for m in expected_modules if not any(m in k for k in sd.keys())]
if missing:
    print(f"FAIL: brak modułów stream w ckpt: {missing}")
    sys.exit(1)
print(f"OK: wszystkie 4 strumienie obecne ({expected_modules})")

# 2) T5 + LoRA — sprawdź że są LoRA adaptery
lora_keys = [k for k in sd.keys() if 'lora_A' in k or 'lora_B' in k]
if not lora_keys:
    print(f"FAIL: brak LoRA adapterów")
    sys.exit(1)
print(f"OK: {len(lora_keys)} LoRA tensors")

# 3) Metryka z nazwy pliku
m = re.search(r'bleu4=([\d.]+)', ckpt_path)
if m:
    bleu = float(m.group(1))
    print(f"BLEU-4 z nazwy ckpt: {bleu}")
    if bleu < 0.1:
        print(f"OSTRZEŻENIE: BLEU bardzo niskie ({bleu}) — phoenix transfer może być słaby, ale finetune i tak ruszy")

# 4) Liczba parametrów
total = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
print(f"Total params: {total/1e9:.2f}B")

print("SANITY OK — proceeding to finetune v23")
PYEOF

if [ $? -ne 0 ]; then
    log "BŁĄD: sanity check failed"
    exit 1
fi

log "sanity OK, ckpt: $V22_CKPT"

# --- 7. finetune v23 ---
log "--- 7. finetune v23 PJM-MS z initem z v22 ---"
python main.py \
    -c configs/finetune_v23_phoenix_init_ms.yaml \
    -n finetune_v23_phoenix_init_ms \
    --ckpt "$V22_CKPT" \
    -e bleu \
    --tags finetune pjm-ms phoenix-init quad \
    2>&1 | tee "$LOG_DIR/07_finetune_v23.log"
log "finetune v23 OK"

# --- 8. cleanup tylko rozpakowanych klatek (tar + zipy zachowane jako backup) ---
log "--- 8. cleanup extracted frames (tar i zipy zostają) ---"
if [ -d /home/croco/data/PHOENIX-2014-T-release-v3 ]; then
    SIZE=$(du -sh /home/croco/data/PHOENIX-2014-T-release-v3 | awk '{print $1}')
    rm -rf /home/croco/data/PHOENIX-2014-T-release-v3
    log "removed: data/PHOENIX-2014-T-release-v3/ ($SIZE freed)"
else
    log "frames już sprzątnięte"
fi
# Orphan MediaPipe part files (z ewentualnej awarii)
rm -f /home/croco/SpaMo-PJM/features/mediapipe_feat_p14t.h5.part* 2>/dev/null || true

df -h /home/croco/SpaMo-PJM | tail -1 | tee -a "$QLOG"

log "=== CONTINUE DONE ==="
log "Zachowane artefakty:"
log "  CKPTY:    logs/<ts>_pretrain_v22_phoenix/checkpoints/*.ckpt    (v22 best by BLEU4)"
log "            logs/<ts>_finetune_v23_phoenix_init_ms/checkpoints/*.ckpt   (v23 best)"
log "  CECHY:    features/p14t/{spatial,motion}/{train,dev,test}/*.npy"
log "            features/{hand_vit,mediapipe}_feat_p14t.h5"
log "  BACKUP:   data/phoenix-2014-T.v3.tar.gz, data/{spa,mo}_feat_p14t.zip"
