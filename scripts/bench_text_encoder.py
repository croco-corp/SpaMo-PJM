"""
Benchmark T5-XL encoder forward vs lighter alternatives.
Measures wall time per batch to estimate fraction of training step.

Usage:
    uv run python scripts/bench_text_encoder.py
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

DEVICE = "cuda:0"
BATCH_SIZE = 8
N_WARMUP = 5
N_ITER = 20
CACHE_DIR = "/home/croco/.cache/huggingface/hub"

SAMPLE_TEXTS = [
    "translate the sign language video to english",
    "the cat sat on the mat near the window",
    "she is signing about her family and daily life",
    "this is a sentence about weather and seasons",
    "he showed me how to sign the word for bread",
    "the interpreter stood next to the speaker on stage",
    "we practiced the alphabet and numbers together",
    "the video shows a sequence of signs for common words",
] * (BATCH_SIZE // 8 + 1)
TEXTS = SAMPLE_TEXTS[:BATCH_SIZE]


def time_encoder(name, encode_fn, n_warmup=N_WARMUP, n_iter=N_ITER):
    # warmup
    for _ in range(n_warmup):
        encode_fn()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        encode_fn()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_call = elapsed / n_iter * 1000
    print(f"  {name:50s}  {ms_per_call:7.1f} ms/batch")
    return ms_per_call


def masked_mean(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


print(f"Batch size: {BATCH_SIZE}, device: {DEVICE}\n")
results = {}

# ── T5-XL encoder (current) ────────────────────────────────────────────────
print("Loading FlanT5-XL...")
t5_tok = AutoTokenizer.from_pretrained("google/flan-t5-xl", cache_dir=CACHE_DIR)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-xl", cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16
).to(DEVICE).eval()

enc_inputs = t5_tok(TEXTS, padding="longest", return_tensors="pt").to(DEVICE)

@torch.no_grad()
def t5_fwd():
    out = t5_model.encoder(**enc_inputs).last_hidden_state.float()
    return masked_mean(out, enc_inputs.attention_mask)

results["FlanT5-XL encoder (current)"] = time_encoder("FlanT5-XL encoder (current)", t5_fwd)
del t5_model
torch.cuda.empty_cache()

# ── mBERT ──────────────────────────────────────────────────────────────────
print("\nLoading mBERT...")
bert_name = "bert-base-multilingual-cased"
bert_tok = AutoTokenizer.from_pretrained(bert_name, cache_dir=CACHE_DIR)
bert_model = AutoModel.from_pretrained(
    bert_name, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16
).to(DEVICE).eval()

bert_inputs = bert_tok(TEXTS, padding="longest", return_tensors="pt").to(DEVICE)

@torch.no_grad()
def bert_fwd():
    out = bert_model(**bert_inputs).last_hidden_state.float()
    return masked_mean(out, bert_inputs.attention_mask)

results["mBERT (bert-base-multilingual-cased)"] = time_encoder("mBERT (bert-base-multilingual-cased)", bert_fwd)
del bert_model
torch.cuda.empty_cache()

# ── Summary ────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────")
t5_ms = results["FlanT5-XL encoder (current)"]
for name, ms in results.items():
    speedup = t5_ms / ms
    print(f"  {name:50s}  {ms:7.1f} ms  ({speedup:.1f}x)")
print()
print("Note: training step also includes visual forward + loss + backward.")
print("Typical visual forward (ViT+proj+pool) ≈ 50-150ms on bf16.")
print(f"T5 encoder alone = {t5_ms:.0f}ms → estimate {t5_ms/(t5_ms+100)*100:.0f}% of forward time.")
