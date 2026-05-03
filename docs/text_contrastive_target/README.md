# Text contrastive target — `embed_tokens` vs `encoder.last_hidden_state`

## TL;DR

Fork's `visual_textual_align` używał **pełnego forwarda T5 encodera** jako targetu kontrastywnego. Powrót do **embed_tokens lookup** (jak upstream / paper) likwiduje kilkujednostkowy regres BLEU na każdym pretraingu/finetuningu z aktywnym contrastive (alpha > 0).

Dotyczy: pretrain v22c (Phoenix paper-faithful BLEU 14.28 vs paper 24.32, vs upstream 25.40), v23/v24 finetune'y, każdy run z `combined_loss: true` i `cross_modal_align: true`.

## Co było źle

`SpaMo-PJM/spamo/t5_slt.py` (commit 324fdb7 "Add contrastive training pipeline" przesunął target):

```python
# WAS (broken)
enc_out = self.t5_model.encoder(
    input_ids=output_tokens.input_ids,
    attention_mask=output_tokens.attention_mask,
).last_hidden_state.float()
text_embeds = (enc_out * mask).sum(1) / mask.sum(1).clamp(min=1)
```

Upstream (`SpaMo/spamo/t5_slt.py:415`) — paper-faithful, NAACL 2025:

```python
text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
text_embeds = text_embeds.mean(1)
```

## Matematyka różnicy

Niech $E \in \mathbb{R}^{V \times d}$ to tablica embedding tokenów T5 (vocab size × d_model), a $H_l(\cdot)$ to wyjście $l$-tej warstwy encodera (24 warstw × self-attention + FFN dla T5-XL).

| | Upstream (poprawne) | Fork pre-fix (wadliwe) |
|---|---|---|
| Target $t$ | $\frac{1}{L}\sum_j E[i_j]$ | $\frac{1}{L}\sum_j H_{24}\!\big(E[i]\big)_j$ |
| Statyczność | $E$ stabilne (mała aktualizacja przez gen. loss) | $H_{24}$ przesuwa się każdą epokę przez LoRA |
| Compute / step | $O(L \cdot d)$ — lookup | $O(L^2 \cdot d \cdot 24)$ — pełny encoder |
| Konkurencja gradientów | brak | jest, jeśli bez `no_grad` |

### Moving target → "chasing tail"

Generation loss aktualizuje T5 encoder każdy step. Wtedy target przesuwa się:

$$t_{k+1} \;=\; t_k + \nabla_\theta t \cdot \Delta\theta_{enc} + O(\|\Delta\theta\|^2)$$

Visual side właśnie nauczył się trafiać $t_k$, target uciekł do $t_{k+1}$. Konwergencja:
- alignment cosine plateau na ~0.3 (zamiast 0.6+ w upstream)
- contrastive loss oscyluje, nie schodzi konsekwentnie
- visual representation pokrywa zamulony fragment przestrzeni encoder feature space, niedopasowany do żadnego stabilnego sygnału tekstowego

W upstream `embed_tokens` to **lookup** w tablicy, nie funkcja parametrów encodera (transformer warstw). $\nabla_\theta t = 0$ dla parametrów warstw → zero feedback loop → stabilny target → stabilna konwergencja.

### Computational cost

Fork robił pełny forward T5 encodera **co batch** w treningu, tylko po to, żeby policzyć target. T5-XL encoder = 24 warstwy × ~80M params = ~2B params per pass. Spowolnienie ~2× per step bez korzyści.

## Skutki empiryczne

| Run | BLEU dev | Paper / upstream baseline |
|---|---|---|
| v22c Phoenix paper-faithful (fork pre-fix) | 14.28 | 24.32 (paper), 25.40 (upstream replikacja) |
| Wszystkie pretrain'y forka po commicie 324fdb7 | systematycznie niżej niż upstream | — |

Empirycznie: fork's contrastive nigdy nie zbiegał tak jak na configu paperowym, niezależnie od `alpha`, `queue_size`, `lr`, ani liczby streamów. Po Switch'u na `embed_tokens` to powinno wrócić do paper-level BLEU.

## Fix

Plik: `spamo/t5_slt.py`, funkcja `visual_textual_align`, gałąź `else` (gdy `use_frozen_text_encoder=False` — domyślne).

```python
else:
    tok_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids).float()
    text_embeds = (tok_embeds * mask).sum(1) / mask.sum(1).clamp(min=1)
```

`with torch.no_grad()` zostaje (fork-decision: frozen text-side dla stabilności), masked-mean zachowane (better handling padding niż unmasked `mean(1)` — drobne ulepszenie vs upstream, neutralne dla treningu).

`use_frozen_text_encoder=True` (RoBERTa) — niezmieniona, to inny eksperyment ablacyjny (zbadany v21, BLEU 2.61 < v20b 3.02 — ścieżka odrzucona, zostawiona dla rozdziału ablacji).

## Verification po fix

Następny pretrain z `cross_modal_align: true, combined_loss: true, alpha: 1.0` — oczekiwane:

1. `train/contra_loss` wyraźnie schodzi (epoch 1 ~3.5, epoch 10 ~1.5; nie oscyluje)
2. `val/cosine_sim_mean` rośnie do >0.5 w 10-20 epokach (vs ~0.3 plateau pre-fix)
3. `val/bleu4` na Phoenix-2014T docelowo zbliża się do paper 24.32 / upstream 25.40 (vs 14.28 pre-fix)
4. Step time ~2× szybszy (brak 24-warstwowego forward dla target)
