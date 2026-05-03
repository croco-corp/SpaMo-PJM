# Raport runów modeli PJM: SpaMo-PJM + Uni-Sign

Stan na 23 kwietnia 2026. Raport dokumentuje treningi dwóch modeli:

- **SpaMo-PJM** — główne repo (FlanT5-XL + LoRA + multi-stream fusion), runy v3–v21 (11–18 kwietnia 2026).
- **Uni-Sign (PJM fork)** — alternatywny framework pose-only + mT5, runy 19–22 kwietnia 2026, uruchomione po SpaMo.

Raport nie zawiera rekomendacji — jest rozpisem settingów i konkretnych zmian między kolejnymi runami.

---

## 1. Kontekst

Zadanie: Gloss-Free Sign Language Translation (SLT) z Polskiego Języka Migowego na tekst.

- Dane: PJM Korpus (Uniwersytet Warszawski, publikowany jako `croco-corp/pjm-segments` na HuggingFace), 730 wideo, 151 migających.
- SpaMo-PJM: FlanT5-XL z LoRA, joint fusion do 4 strumieni wizualnych, tłumaczy na język angielski (`texts_eng.h5`).
- Uni-Sign PJM fork: mT5 + pose-only, dodaje klasę `S2T_Dataset_PJM` (Uni-Sign/datasets.py:670), bez strumienia RGB.

---

## 2. Splity danych

Używane są dwie rodziny splitów (poza skopem tego raportu leży pipeline ich generowania w sąsiednim repo CrocoSign):

- **SI (single-migający)** — każdy migający trafia w całości do jednego podzbioru (train/val/test), bez rozdzielania tej samej osoby między zbiory; w dev/test pojawiają się osoby nieznane modelowi. Rodzina SI ma dwa warianty:
  - `split_train.csv` (vanilla SI) — 11 448 próbek, 51 migających; bazowy split bez filtrów
  - `split_train_filtered.csv` (filtered SI) — 10 063 próbki, 51 migających; usunięte krótkie/niskiej jakości transkrypcje
- **MS (multi-migających)** — wszystkie osoby są obecne proporcjonalnie we wszystkich podzbiorach (styl PHOENIX-2014T):
  - `split_train_ms.csv` — 23 941 próbek, 151 migających

Uni-Sign stosuje tę samą kategoryzację:

- **SI** w Uni-Sign (~1412 próbek dev) — czyta splity rodziny SI z `../CrocoSign/data/`
- **MS** w Uni-Sign (~375 próbek dev) — czyta `../CrocoSign/data/split_*_ms.csv`

---

## 3. SpaMo-PJM — runy v3–v21

### 3.1 Architektura

- Model: `spamo.t5_slt.FlanT5SLT` (FlanT5-XL + LoRA, projekcja przez `inter_hidden=768`, `fusion_mode=joint`).
- LoRA na ogół: `r=16`, `α=32`, `dropout=0.1` (wyjątek v21: `r=8`, `α=16`).
- Dwa LR-y: `lr` (cały model poza fusion) i `fusion_lr` (warstwy fuzji od zera, 10× niższy).
- Zawsze `max_frame_len=512`, `max_txt_len=64`, `precision=bf16`, `gradient_clip_val=1.0`, `fusion_dropout=0.3`.
- Prompt stały: `"Translate the given sentence into English."`.

### 3.2 Strumienie

| Oznaczenie | Klucz w configu | Plik HDF5 | Wymiar |
|---|---|---|---|
| spatial | `visual_features_path` | `features/vit_feat_pjm.h5` | 2048 (`input_size`) |
| motion | `motion_features_path` | `features/mae_feat_pjm.h5` *lub* `hand_vit_feat_pjm.h5` | 1024 (MAE) / 2048 (hand_ViT) — `motion_input_size` |
| aux | `aux_features_path` | `features/hand_vit_feat_pjm.h5` | 2048 (`aux_input_size`) |
| keypoint | `keypoint_features_path` | `features/mediapipe_feat_pjm.h5` | 258 (`keypoint_dim`) — MediaPipe Holistic: 33 pose×4 + 21 L-hand×3 + 21 R-hand×3 |

Uwaga: w v9/v10 strumień „motion" nie wskazuje na VideoMAE, tylko na hand-crop ViT (`motion_features_path: hand_vit_feat_pjm.h5`). Dopiero od v11 VideoMAE wraca jako osobny „motion", a hand_ViT przechodzi do slotu „aux".

### 3.3 Tabela zbiorcza v3–v21

| Wersja | Streamy | `combined_loss` | α | LR (main / fusion) | Batch (acc) | Epochs max | Scheduler | Monitor | Beam | LoRA r/α | Split |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | dual (spatial+motion) | — | — | — | — | 5 (partial) | — | — | — | — | `split_train` |
| v3 | dual (ViT + MAE) | true | 1.0 | 6e-4 / 6e-5 | 8 (acc=2) | 60 | — | val/bleu4 | 5 | 16/32 | `split_train` |
| v4 | dual (ViT + MAE) | **false** | 1.0 | 6e-4 / 6e-5 | 8 (acc=2) | 60 | — | **val/alignment_gap** | 5 | 16/32 | `split_train` |
| v7 | dual (ViT + MAE) | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=2) | 60 | — | val/alignment_gap | 5 | 16/32 | **`split_train_filtered`** |
| v8 | dual (ViT + MAE) | false | 1.0 | 6e-4 / 6e-5 | 8 (**acc=1**) | 60 | — | val/alignment_gap | 5 | 16/32 | `split_train_filtered` |
| v9 | dual (ViT + **hand_ViT jako motion**) | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | 60 | — | val/alignment_gap | 5 | 16/32 | `split_train_filtered` |
| v10 | triple (ViT + hand_ViT jako motion + **MediaPipe**) | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | 60 | — | val/alignment_gap | 5 | 16/32 | `split_train_filtered` |
| v11 | triple (ViT + **hand_ViT jako aux** + MAE) | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | 60 | — | val/alignment_gap | **1** | 16/32 | `split_train_filtered` |
| v12 | **quad** (ViT + hand_ViT + MAE + MediaPipe) | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | 60 | — | val/alignment_gap | 1 | 16/32 | `split_train_filtered` |
| v13 | quad | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | **30** | — | val/alignment_gap | 1 | 16/32 | `split_train_filtered` |
| v14 | quad | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | **20** | — | val/alignment_gap | 1 | 16/32 | `split_train_filtered` |
| v15 | quad | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | 60 | **Plateau p=5** | val/alignment_gap | 1 | 16/32 | `split_train_filtered` |
| v15b | quad | false | 1.0 | 6e-4 / 6e-5 | 8 (acc=1) | 60 | Plateau p=5 **mode=max** | val/alignment_gap | 1 | 16/32 | `split_train_filtered` |
| v16 | quad | **true** | **0.5** | **3e-4 / 3e-5** | 8 (acc=1) | 60 | Plateau p=5 **mode=min** | **val/loss** | **4** | 16/32 | `split_train_filtered` |
| v17 | quad | true | **0.1** | **1e-4 / 1e-5** | 8 (acc=1) | 60 | Plateau p=5 mode=min | val/loss | 4 | 16/32 | `split_train_filtered` |
| v18 | quad | true | **0.0** (`cross_modal_align=false`, `queue=0`) | **3e-4 / 3e-5** | 8 (acc=1) | 60 | Plateau p=5 mode=min | val/loss | 4 | 16/32 | `split_train_filtered` |
| v19 | **dual (ViT + MAE, Phoenix-compat)** | true | 0.0 | 1e-4 / 1e-5 | 8 (acc=1) | 60 | Plateau p=5 mode=min | val/loss | 4 | 16/32 | `split_train_filtered` |
| v20 | quad | true | 0.1 | 1e-4 / 1e-5 | **4 (acc=2)** | 60 | Plateau p=5 mode=min | val/loss | 4 | 16/32 | **`split_train_ms`** |
| v21 | quad | true | **0.3** | 1e-4 / 1e-5 | 4 (acc=2) | 60 | Plateau p=5 **mode=max** | **val/bleu4** | 4 | **8/16** | `split_train_ms` |

Wyniki (best metric; dla runów contrastive-only podany `alignment_gap`, dla generation — `val/loss` lub `BLEU-4`):

| Wersja | Epok przebytych | Status | Best metryka |
|---|---|---|---|
| baseline | 5 | partial | BLEU-4 = 1.031 |
| v3 | 5 | partial | BLEU-4 = 0.985 |
| v4 (3 próby) | 1–2 | partial | alignment_gap ≈ 0.01–0.09 |
| v7 | 2 | partial | alignment_gap = 0.081 |
| v8 | 9 | partial | alignment_gap = 0.105 |
| v9 | 12 | partial | alignment_gap = 0.105 |
| v10 | 6 | partial | alignment_gap = 0.126 |
| v11 | 9 | partial | alignment_gap = 0.122 |
| v12 | 7 | partial | alignment_gap = **0.134** (najlepszy contrastive) |
| v13 | 10 | partial | alignment_gap = 0.128 |
| v14 | 8 | partial | alignment_gap = 0.127 |
| v15b | 10 | partial | alignment_gap = 0.124 |
| **v16** | 32 | **completed** | val/loss = 5.709 |
| v17a | 7 | crashed (DataLoader worker) | val/loss = 7.143 |
| v17b | 3 | partial | val/loss = 2.474 |
| v18a/b | 2–3 | partial | val/loss ≈ 2.40–2.42 |
| v19 | 8 | interrupted (manual) | val/loss = 2.642 |
| v20a | 7 | partial | val/loss = 2.210 |
| **v20b** | 10 | partial | **BLEU-4 = 3.016** (najlepszy SpaMo) |
| v21 | 11 | partial | BLEU-4 = 2.607 |

Checkpointy w `checkpoints/`:

- `epoch=00019-step=0044800-bleu4=2.28.ckpt`
- `pheonix-spamo-pretrained.ckpt` (inicjalizacja dla v19)

### 3.4 Sekcje per-run

---

#### baseline (`2026-04-11T14-44-23_pjm-baseline-v1`)

- Data startu: 11 kwietnia 2026
- Cel: pierwsze uruchomienie SpaMo-PJM na czystym pipeline FlanT5-XL
- **Zmiany vs poprzedni run**: pierwszy run PJM, punkt odniesienia
- Wynik: BLEU-4 = 1.031, 5 epok (partial)

---

#### v3 — `finetune_v3_diff_lr.yaml`

- Config: `configs/finetune_v3_diff_lr.yaml`
- Settings:
  - Streamy: dual (ViT + VideoMAE)
  - Loss: `combined_loss=true`, `alpha=1.0`, `cross_modal_align=true`
  - LR: `lr=6e-4`, `fusion_lr=6e-5` (10× niższy dla warstw fuzji)
  - Batch=8, `accumulate_grad=2`, `max_epochs=60`, `beam_size=5`
  - LoRA r=16/α=32
  - Monitor: `val/bleu4`
- Split: `split_train.csv` (vanilla)
- **Zmiana vs baseline**:
  - Wprowadzone dwa osobne LR-y (main vs fusion)
  - `combined_loss` z alpha=1.0 (waga contrastive równa ~CE)
- Wynik: BLEU-4 = 0.985, 5 epok (partial)
- Artefakty: `logs/2026-04-12T09-12-44_finetune_v3_diff_lr/`

---

#### v4 — `finetune_v4_contrastive.yaml`

- Config: `configs/finetune_v4_contrastive.yaml`
- Settings: jak v3, różnice:
  - `combined_loss=false` (czysty contrastive, bez CE T5)
  - `monitor=val/alignment_gap` (zmiana metryki referencyjnej)
- **Zmiany vs v3**:
  - Wyłączona strata generacyjna (`combined_loss: true → false`)
  - Monitor `val/bleu4 → val/alignment_gap`
- Wynik: 3 próby (12 kwietnia 12:53, 13:21, 13:43), wszystkie partial, alignment_gap ≈ 0.01–0.09
- Artefakty: `logs/2026-04-12T12-{39,53}..13-{21,43}*_finetune_v4_contrastive/`

---

#### v5, v6

Brak plików konfiguracyjnych w `configs/` — numeracja przeskoczyła na v7 (prawdopodobnie nieudane wewnętrzne próby nie dopięte do osobnego YAML-a).

---

#### v7 — `finetune_v7_filtered.yaml`

- Config: `configs/finetune_v7_filtered.yaml`
- Settings: jak v4
- Split: **`split_train_filtered.csv`**
- **Zmiany vs v4**:
  - Przełączenie na filtrowany split (`split_train.csv → split_train_filtered.csv`)
- Wynik: alignment_gap = 0.081, 2 epoki (partial)
- Artefakty: `logs/2026-04-12T14-11-19_finetune_v7_filtered/`

---

#### v8 — `finetune_v8_bigbatch.yaml`

- Config: `configs/finetune_v8_bigbatch.yaml`
- Settings: jak v7
- **Zmiany vs v7**:
  - `accumulate_grad_batches: 2 → 1` (faktyczny batch z 16 do 8)
  - Dodany jawny `queue_size=4096` dla kolejki negatywów contrastive
- Wynik: alignment_gap = 0.105, 9 epok (partial)
- Artefakty: `logs/2026-04-12T14-{28,51}*_finetune_v8_bigbatch/`

---

#### v9 — `finetune_v9_handvit.yaml`

- Config: `configs/finetune_v9_handvit.yaml`
- Streamy: dual — **spatial=ViT, motion=hand_ViT** (VideoMAE zastąpiony hand-crop ViT)
- **Zmiany vs v8**:
  - `motion_features_path: mae_feat_pjm.h5 → hand_vit_feat_pjm.h5`
  - `motion_input_size: 1024 → 2048` (dopasowanie do wymiaru hand_ViT)
- Wynik: alignment_gap = 0.105, 12 epok (partial)
- Artefakty: `logs/2026-04-13T{11,14,15}*_finetune_v9_handvit/`

---

#### v10 — `finetune_v10_mediapipe.yaml`

- Config: `configs/finetune_v10_mediapipe.yaml`
- Streamy: triple — spatial=ViT + motion=hand_ViT + **keypoint=MediaPipe**
- **Zmiany vs v9**:
  - Dodany trzeci strumień: `keypoint_features_path: features/mediapipe_feat_pjm.h5`, `keypoint_dim=258`
  - VideoMAE nadal nieużywany (motion = hand_ViT)
- Wynik: alignment_gap = 0.126, 6 epok (partial)
- Artefakty: `logs/2026-04-13T16-40-32_finetune_v10_mediapipe/`

---

#### v11 — `finetune_v11_triple.yaml`

- Config: `configs/finetune_v11_triple.yaml`
- Streamy: triple — spatial=ViT + **aux=hand_ViT + motion=VideoMAE** (MediaPipe wyłączone)
- **Zmiany vs v10**:
  - Przeorganizowane sloty: hand_ViT z slotu „motion" przeniesiony do „aux" (`aux_input_size=2048`), VideoMAE wraca jako „motion" (`motion_input_size=1024`)
  - `keypoint_dim: 258 → 0` (MediaPipe wyłączone)
  - `beam_size: 5 → 1` (szybsza walidacja w trybie contrastive)
- Wynik: alignment_gap = 0.122, 9 epok (partial)
- Artefakty: `logs/2026-04-13T{16-41,17-17}*_finetune_v11_triple/`

---

#### v12 — `finetune_v12_quad.yaml`

- Config: `configs/finetune_v12_quad.yaml`
- Streamy: **quad** — ViT + hand_ViT (aux) + VideoMAE (motion) + MediaPipe (keypoint)
- **Zmiany vs v11**:
  - Ponownie włączony keypoint stream: `keypoint_dim: 0 → 258`
- Wynik: alignment_gap = 0.134 (najlepszy wynik contrastive), 7 epok (partial)
- Artefakty: `logs/2026-04-13T18-02-10_finetune_v12_quad/`

---

#### v13 — `finetune_v13_quad_tuned.yaml`

- Config: `configs/finetune_v13_quad_tuned.yaml`
- **Zmiany vs v12**:
  - `max_epochs: 60 → 30`
  - `check_val_every_n_epoch: 2 → 1` (częstsza walidacja)
- Wynik: alignment_gap = 0.128, 10 epok (partial)
- Artefakty: `logs/2026-04-13T19-09-00_finetune_v13_quad_tuned/`

---

#### v14 — `finetune_v14_quad_20ep.yaml`

- Config: `configs/finetune_v14_quad_20ep.yaml`
- **Zmiany vs v13**:
  - `max_epochs: 30 → 20`
- Wynik: alignment_gap = 0.127, 8 epok (partial)
- Artefakty: `logs/2026-04-13T22-02-09_finetune_v14_quad_20ep/`

---

#### v15 — `finetune_v15_plateau.yaml`

- Config: `configs/finetune_v15_plateau.yaml`
- **Zmiany vs v14**:
  - Wprowadzony scheduler **ReduceLROnPlateau**, `lr_patience=5` (`max_epochs` wraca do 60)
  - `logging_interval: step → epoch`
- Wynik: (ten run szybko iterowany do v15b, patrz niżej)
- Artefakty: `logs/2026-04-13T22-45-32_finetune_v15_plateau/`

---

#### v15b — `finetune_v15b_plateau_p5.yaml`

- Config: `configs/finetune_v15b_plateau_p5.yaml`
- **Zmiany vs v15**:
  - Jawne `lr_scheduler_mode=max` (monitor = `val/alignment_gap`, który się maksymalizuje)
- Wynik: alignment_gap = 0.124, 10 epok (partial)
- Artefakty: `logs/2026-04-13T23-{04,25}*_finetune_v15b_plateau_p5/`

---

#### v16 — `finetune_v16_combined.yaml`

- Config: `configs/finetune_v16_combined.yaml`
- Settings: quad, `combined_loss=true`, `alpha=0.5`, `beam_size=4`, Plateau p=5 mode=min, monitor `val/loss`, lr=3e-4
- **Zmiany vs v15b** (istotne przejście z contrastive-only na combined):
  - `combined_loss: false → true`, `alpha: 1.0 → 0.5` (pół-pół CE i contrastive)
  - `lr: 6e-4 → 3e-4` (niższy LR dla fazy generacyjnej)
  - `lr_scheduler_mode: max → min` (teraz minimalizujemy `val/loss`)
  - `monitor: val/alignment_gap → val/loss`
  - `beam_size: 1 → 4` (rzeczywiste generowanie zamiast samej miary alignmentu)
- Wynik: val/loss = 5.709, **32 epoki, jedyny completed run w SpaMo**
- Artefakty: `logs/2026-04-14T00-28-26_finetune_v16_combined/`

---

#### v17 — `finetune_v17_generation.yaml`

- Config: `configs/finetune_v17_generation.yaml`
- **Zmiany vs v16**:
  - `lr: 3e-4 → 1e-4`
  - `alpha: 0.5 → 0.1` (wzmocnienie udziału CE)
- Wynik: v17a (7 ep, crash workera, `val/loss=7.143`); v17b (3 ep, partial, `val/loss=2.474`); eval na train (`2026-04-15T08-31-11_eval_train_v17`) `val/loss=2.49`
- Artefakty: `logs/2026-04-14T{20-58,23-25}*_finetune_v17_generation/`, `logs/2026-04-15T08-31-11_eval_train_v17/`

---

#### v18 — `finetune_v18_pure_generation.yaml`

- Config: `configs/finetune_v18_pure_generation.yaml`
- **Zmiany vs v17**:
  - `alpha: 0.1 → 0.0` (czysta strata generacyjna CE)
  - `cross_modal_align: true → false`
  - `queue_size: 4096 → 0` (kolejka contrastive wyłączona)
  - `lr: 1e-4 → 3e-4` (przy pełnej CE podniesiony LR)
- Wynik: v18a (2 ep, partial, val/loss ≈ 2.42); v18b (3 ep, partial, val/loss ≈ 2.40)
- Artefakty: `logs/2026-04-15T{17-33,23-24}*_finetune_v18_pure_generation/`

---

#### v19 — `finetune_v19_phoenix_finetune.yaml`

- Config: `configs/finetune_v19_phoenix_finetune.yaml`
- Settings: **dual** (ViT + MAE), compatible z `pheonix-spamo-pretrained.ckpt`
- **Zmiany vs v18**:
  - Architektura z powrotem do dual: `keypoint_dim: 258 → 0`, `aux_input_size: 2048 → 0` (brak MediaPipe i hand_ViT)
  - Inicjalizacja: checkpoint `pheonix-spamo-pretrained.ckpt` zamiast scratchu
  - `lr: 3e-4 → 1e-4` (LR zjechany, bo finetuning z checkpointu)
- Wynik: val/loss = 2.642, 8 epok (manual interrupt)
- Artefakty: `logs/2026-04-16T10-15-16_finetune_v19_phoenix_finetune/`

---

#### v20 — `finetune_v20_ms_generation.yaml`

- Config: `configs/finetune_v20_ms_generation.yaml`
- Settings: quad (powrót), combined z `alpha=0.1`, `batch_size=4`, `accumulate_grad=2`, Plateau p=5 mode=min, lr=1e-4, monitor `val/loss`
- **Zmiany vs v19**:
  - Architektura wraca do quad (MediaPipe i hand_ViT ponownie włączone)
  - `combined_loss`: efektywnie włączony contrastive (alpha=0.1 zamiast 0.0), `cross_modal_align: false → true`, `queue_size: 0 → 4096`
  - **Split**: `split_train_filtered.csv → split_train_ms.csv` (dataset multi-migających)
  - `batch_size: 8 → 4` (mniejszy ze względu na szerszy dataset), `accumulate_grad: 1 → 2` (efektywny batch dalej 8)
- Wynik:
  - v20a (`2026-04-16T15-07-{05,11}`): 7 ep, val/loss = 2.210, partial
  - **v20b** (`2026-04-17T18-06-37_finetune_v20b_ms_bleu`): 10 ep, **BLEU-4 = 3.016** (najlepszy SpaMo)
- Artefakty: `logs/2026-04-16T14-33-45_finetune_v20_ms_generation/`, `logs/2026-04-16T15-07-*_finetune_v20_ms_generation/`, `logs/2026-04-17T18-06-37_finetune_v20b_ms_bleu/`, `logs/v20_per_speaker_bleu4.csv`

---

#### v21 — `finetune_v21_ms_generation_with_roberta.yaml`

- Config: `configs/finetune_v21_ms_generation_with_roberta.yaml`
- Settings: quad, combined z `alpha=0.3`, `batch_size=4`, acc=2, Plateau p=5 **mode=max**, lr=1e-4, monitor `val/bleu4`, LoRA r=8/α=16, `use_frozen_text_encoder=true`
- **Zmiany vs v20b**:
  - `LoRA r=16 → 8`, `α=32 → 16` (redukcja liczby parametrów LoRA)
  - `alpha: 0.1 → 0.3` (większy udział straty contrastive w combined)
  - `use_frozen_text_encoder: false → true` — zamrożony encoder RoBERTa jako źródło reprezentacji tekstowych dla strat contrastive
  - `lr_scheduler_mode: min → max` (powiązane z zmianą monitora)
  - `monitor: val/loss → val/bleu4`
- Wynik: BLEU-4 = 2.607, 11 epok (partial)
- Artefakty: `logs/2026-04-18T01-{11,45}*_finetune_v21_ms_generation_with_roberta/`

---

## 4. Uni-Sign PJM — runy (19–22 kwietnia 2026)

### 4.1 Upstream i modyfikacje pod PJM

- Upstream: Uni-Sign (Li et al., ICLR 2025, arXiv:2501.15187) — wspólny framework SLT z fuzją pozy (body + hands + face) przez GCN, tłumaczenie mT5.
- Fork dodaje:
  - `S2T_Dataset_PJM` w `Uni-Sign/datasets.py:670` — ładuje PJM z `../CrocoSign/data/split_*.csv`
  - Teksty z `../CrocoSign/data/texts_eng.h5` (angielskie tłumaczenia)
  - Pozy PJM w `Uni-Sign/dataset/PJM/pose_format/` (pickle; 1 próbka pominięta z braku pozy)
  - `rgb_support=False` — tryb pose-only (RGB wyłączone)

### 4.2 Fazy treningu i warianty

- **Phase 2 (LoRA)** — 10 epok, LoRA na mT5, zamrożone wizualne, mniejszy batch, szybki
- **Phase 3 (full fine-tuning)** — 15 (MS) lub 30 (SI) epok, odmrożone całe siatki, batch 8, `weight_decay=0.01`, LR=3e-4
- Warianty splitu:
  - **SI** (single-interaction, ~1412 dev): szerszy, mniej przefiltrowany zbiór
  - **MS** (multi-migających, ~375 dev): filtrowane splity `split_*_ms.csv`

### 4.3 Inicjalizacje

Trzy źródła wag testowane jako punkt wyjścia:

- **CSL-News** — `out/csl_news_stage2/csl_stage2_weight.pth` (upstreamowy benchmark)
- **OpenASL** — `pretrained_weight/openasl_pose_only_slt.pth` (ASL pose-only)
- **How2Sign** — checkpoint z How2Sign (ablacja)

### 4.4 Tabela zbiorcza

| Log | Faza | Split | Init | Epok | Czas wall-clock | Status | BLEU-4 | BERTScore | Output |
|---|---|---|---|---|---|---|---|---|---|
| `train_pjm.log` | — | — | — | — | 19 kwi 13:51 | ERROR (nohup fail) | — | — | — |
| `eval_queue.log` / `eval_queue_failed1.log` | eval | PJM | — | — | 19 kwi | partial | — | — | — |
| `phase2_queue.log` | Phase 2 (LoRA) | PJM-MS | CSL-News | 10 | ~1 h (20 kwi 00:11→01:14) | completed | — (loss ≈ 9.97) | — | `out/pjm_phase2_lora_ms/` |
| `phase3_queue.log` | Phase 3 (full) | PJM-MS | CSL-News | 15 | 5 h 33 min (20 kwi 01:22→06:56) | completed | 2.60 % | 86.78 | `out/pjm_phase3_full_ms/` |
| `phase3_si.log` | Phase 3 (full) | PJM-SI | CSL-News | 30 | 8 h 09 min (20 kwi 09:13→17:23) | completed | 1.29 % | 86.59 | `out/pjm_phase3_full_si/` |
| `eval_si_best.log` | eval | PJM-SI | best ckpt z phase3_si | — | 20 kwi 19:17→19:29 | completed | 1.32 % | 86.40 | `out/pjm_phase3_si_best_eval/` |
| **`phase3_ms_openasl.log`** | Phase 3 (full) | PJM-MS | **OpenASL** | 15 | 5 h 32 min (20 kwi 19:32 → 21 kwi 01:05) | completed | **7.40 %** | **88.51** | `out/pjm_phase3_full_ms_openasl/` |
| `queue_si_openasl.log` | Phase 3 (full) | PJM-SI | OpenASL | 30 | ~21 h (21–22 kwi) | completed | 4.82 % | 87.81 | `out/pjm_phase3_full_si_openasl/` |
| `queue_followup.log` | Phase 3 (full) | PJM-SI+MS | How2Sign | 30 / 15 | do 22 kwi 00:04 | completed | 4.82 % (SI), ~2.6 % (MS) | 87.81 | `out/pjm_phase3_full_{si,ms}_how2sign/` |
| — | eval-only | PJM-MS | OpenASL | 0 | — | completed | 2.28 % | — | `out/pjm_openasl_zeroshot_ms/` |
| — | eval-only | PJM-SI | OpenASL | 0 | — | completed | 0.39 % | — | `out/pjm_openasl_zeroshot_si/` |

### 4.5 Sekcje per-run

---

#### train_pjm.log (19 kwietnia)

- Pierwsza próba uruchomienia treningu PJM — proces nie wystartował (błąd nohup).
- **Zmiana**: brak — stan „zero" przed pierwszym udanym runem.
- Wynik: ERROR.

---

#### eval_queue.log / eval_queue_failed1.log (19 kwietnia)

- Eval-only uruchomienie, przed startem pełnych treningów.
- **Zmiana**: —
- Wynik: partial, log nie zawiera końcowych metryk PJM.

---

#### phase2_queue.log — LoRA MS (20 kwietnia, 00:11–01:14)

- Faza: **Phase 2 (LoRA)**
- Settings: batch=16, LoRA rank=16, lr=3e-4, 1496 kroków/epokę, 10 epok
- Init: CSL-News
- Split: PJM-MS (filtered multi-migających)
- **Zmiany vs train_pjm.log**: pierwszy poprawnie wystartowany trening; wprowadzony podział na fazę 2 (LoRA)
- Wynik: completed, loss: 9.94 → 9.97 (brak BLEU raportowanego per epoka w tej fazie)
- Output: `out/pjm_phase2_lora_ms/`, W&B run id `5sxqesks`

---

#### phase3_queue.log — Full MS CSL-News (20 kwietnia, 01:22–06:56)

- Faza: **Phase 3 (full fine-tune)**
- Settings: batch=8, lr=3e-4, weight_decay=0.01, 15 epok
- Init: CSL-News stage-2
- Split: PJM-MS
- **Zmiany vs phase2_queue**:
  - Przejście z LoRA do pełnego fine-tuningu (`fine_tuning.py` z odmrożoną całą siecią)
  - `max_epochs: 10 → 15`
  - `batch_size: 16 → 8`
  - Metryki generacyjne (BLEU/BERTScore) naliczane per epoka
- Wynik: completed, **BLEU-4 = 2.60 %**, BERTScore = 86.78 (dev plateau ok. 5. epoki)
- Output: `out/pjm_phase3_full_ms/`, W&B `hztqml76`

---

#### phase3_si.log — Full SI CSL-News (20 kwietnia, 09:13–17:23)

- Faza: Phase 3 (full fine-tune)
- Settings: batch=8, lr=3e-4, weight_decay=0.01, 30 epok
- Init: CSL-News
- Split: PJM-SI (~1412 dev)
- **Zmiany vs phase3_queue**:
  - **Split**: MS → SI (~4× więcej danych treningowych, szerszy dev)
  - `max_epochs: 15 → 30`
- Wynik: completed, BLEU-4 = 1.29 %, BERTScore = 86.59 (konwergencja wolniejsza, BLEU niższy mimo większej ilości danych)
- Output: `out/pjm_phase3_full_si/`, W&B `q7wnyztr`

---

#### eval_si_best.log (20 kwietnia, 19:17–19:29)

- Eval checkpointu „best" z phase3_si na dev SI.
- **Zmiana vs phase3_si**: bez treningu — tylko ewaluacja
- Wynik: BLEU-4 = 1.32 %, BERTScore = 86.40
- Output: `out/pjm_phase3_si_best_eval/`

---

#### phase3_ms_openasl.log — Full MS OpenASL (20 kwi 19:32 → 21 kwi 01:05)

- Faza: Phase 3 (full fine-tune)
- Settings: batch=8, lr=3e-4, weight_decay=0.01, 15 epok
- Init: **OpenASL** (`pretrained_weight/openasl_pose_only_slt.pth`)
- Split: PJM-MS
- **Zmiany vs phase3_queue** (ten sam config treningowy, inna inicjalizacja):
  - Init: CSL-News → OpenASL
- Wynik: completed, **BLEU-4 = 7.40 %**, BERTScore = 88.51 (epoka 14 szczyt: BLEU-4 = 8.18 %) — najlepszy wynik w całym ekosystemie PJM
- Output: `out/pjm_phase3_full_ms_openasl/`, W&B `0aahubtw`

---

#### queue_si_openasl.log — Full SI OpenASL (21–22 kwietnia)

- Settings: batch=8, lr=3e-4, weight_decay=0.01, 30 epok
- Init: OpenASL
- Split: PJM-SI
- **Zmiany vs phase3_si**:
  - Init: CSL-News → OpenASL
- Wynik: completed, BLEU-4 = 4.82 %, BERTScore = 87.81 (~3.7× skok vs CSL init)
- Output: `out/pjm_phase3_full_si_openasl/`

---

#### queue_followup.log — How2Sign ablacja (21–22 kwietnia do 00:04)

- Settings: batch=8, lr=3e-4, weight_decay=0.01
- Init: **How2Sign** (finalna ablacja inicjalizacji)
- Runy: PJM-SI (30 ep) i PJM-MS (15 ep)
- **Zmiany vs queue_si_openasl / phase3_ms_openasl**:
  - Init: OpenASL → How2Sign
- Wynik:
  - SI: BLEU-4 = 4.82 %, BERTScore = 87.81 (porównywalny do OpenASL SI)
  - MS: BLEU-4 ≈ 2.6 % (zbliżony do CSL init — transfer z How2Sign nie daje boosta na MS)
- Output: `out/pjm_phase3_full_si_how2sign/`, `out/pjm_phase3_full_ms_how2sign/`

---

#### Zero-shot OpenASL (bez treningu)

- Init: OpenASL, brak fine-tuningu
- Wynik:
  - MS: BLEU-4 = 2.28 %
  - SI: BLEU-4 = 0.39 %
- Output: `out/pjm_openasl_zeroshot_ms/`, `out/pjm_openasl_zeroshot_si/`

### 4.6 W&B

Projekt: `uni-sign-pjm`, 23 zalogowane runy. Metryki: BLEU-1/2/3/4, BERTScore, CHRF, length_ratio per epoka. Przykładowe run ID: `5sxqesks` (phase2_lora_ms), `hztqml76` (phase3_full_ms), `q7wnyztr` (phase3_full_si), `0aahubtw` (phase3_full_ms_openasl).

---

## 5. Linia czasu

```
2026-04-11  baseline                          (SpaMo-PJM)
2026-04-12  v3 → v4 → v7 → v8                 (SpaMo-PJM, iteracje loss i splitu)
2026-04-13  v9 → v10 → v11 → v12 → v13 → v14
            → v15 → v15b                      (SpaMo-PJM, modalności i scheduler)
2026-04-14  v16                               (SpaMo-PJM, pierwszy combined loss)
2026-04-14/15 v17 → v18                       (SpaMo-PJM, przejście do generation)
2026-04-16  v19 → v20a                        (SpaMo-PJM, phoenix finetune → MS split)
2026-04-17  v20b                              (SpaMo-PJM, BLEU-4 monitor)
2026-04-18  v21                               (SpaMo-PJM, frozen RoBERTa)
2026-04-19  Uni-Sign: train_pjm + eval_queue  (pierwsze próby, błąd nohup)
2026-04-20  Uni-Sign: phase2_queue, phase3_queue, phase3_si, eval_si_best, phase3_ms_openasl
2026-04-21  Uni-Sign: queue_si_openasl, queue_followup (How2Sign SI+MS)
2026-04-22  Uni-Sign: queue_followup zakończone (00:04)
```

---

## 6. Najlepsze wyniki (BLEU-4)

| Pozycja | Repo | Run | BLEU-4 | Uwagi |
|---|---|---|---|---|
| 1 | Uni-Sign | `pjm_phase3_full_ms_openasl` | **7.40 %** | OpenASL init, MS, 15 ep |
| 2 | Uni-Sign | `queue_si_openasl` | 4.82 % | OpenASL init, SI, 30 ep |
| 3 | Uni-Sign | `phase3_full_si_how2sign` | 4.82 % | How2Sign init, SI, 30 ep |
| 4 | SpaMo-PJM | v20b | 3.016 % | MS split + BLEU-4 monitor, 10 ep |
| 5 | SpaMo-PJM | v21 | 2.607 % | MS split + frozen RoBERTa, 11 ep |
| 6 | Uni-Sign | `phase3_full_ms` (CSL) | 2.60 % | CSL-News init, MS, 15 ep |
| 7 | Uni-Sign | zero-shot MS (OpenASL) | 2.28 % | brak fine-tuningu |
| 8 | Uni-Sign | `phase3_full_si` (CSL) | 1.29 % | CSL-News init, SI, 30 ep |
| 9 | SpaMo-PJM | baseline | 1.031 % | FlanT5-XL, 5 ep |
| 10 | SpaMo-PJM | v3 | 0.985 % | diff_lr, 5 ep |
| 11 | Uni-Sign | zero-shot SI (OpenASL) | 0.39 % | brak fine-tuningu |

Dla ścisłości: runy SpaMo v4–v15b raportują tylko `alignment_gap` (0.081–0.134), nie BLEU-4, więc nie występują w powyższym rankingu. Runy v16–v19 raportują głównie `val/loss` (bez BLEU w plikach logów); najlepszy z nich to v16 (val/loss = 5.709, 32 epoki).
