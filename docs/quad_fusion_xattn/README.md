# Quad-stream fusion via cross-attention z zero-init residual

**Status:** design — nie zaimplementowane.
**Cel:** włączać `aux_proj` (hand_ViT) i `kp_proj` (MediaPipe) do trenowanego dual-ckpt bez destabilizacji wytrenowanego forwarda.

## Problem

Aktualny fork (joint mode) konkatenuje aux/keypoint do parts wraz z V (visual) i M (motion) przed `temporal_encoder`:

```python
parts = [V_visual, M_motion]
if self.aux_input_size > 0:
    parts.append(aux_outputs[i, :aux_length[i], :])
if self.keypoint_dim > 0:
    parts.append(kp_outputs[i, :kp_length[i], :])
joint_outputs = pad_sequence([torch.cat(parts, dim=0) for ...])
```

Empirycznie (v25 quad continuation z upstream dual ckpt 5.36):

- init val BLEU 0.91 (vs upstream 5.36) — degeneracja na samym start
- random `aux_proj` / `kp_proj` produkują noise → temporal_encoder + fusion_proj fuzują noise z dobrymi V/M → fusion_proj output to noise prefix
- T5 trenowany na dual prefix length dostaje quad prefix → attention pattern decoder cross-attention nie pasuje

Dwa źródła problemu:
1. **Random projektory** dla nowych streamów psują forward.
2. **Zmieniona długość prefix** — T5 nie była trenowana na quad-length sekwencjach.

## Rozwiązania rozważone

| Aspekt | A) Freeze warmup callback | B) Zero-init last linear projektora | C) Learnable gate $\alpha$ | **E) Cross-attn + zero-init out** |
|---|---|---|---|---|
| Prefix length | zmieniony | zmieniony | zmieniony | **stały (= dual)** |
| Init = identity z dual ckpt | nie | częściowo (zero tokeny) | częściowo (gate=0) | **tak (zero residual)** |
| T5 attention pattern | inny | inny | inny | **identyczny z trenowanym** |
| Gradient flow w nowe streamy | nagle po unfreeze | przez wszystkie wagi | przez skalar | **przez $W_O$, kontrolowane** |
| Standard literaturowy | brak | ResNet trick, lokalne | sporadycznie | **Flamingo, BLIP-2, LLaVA, DiT** |
| Zmiana kodu | callback + freeze logic | 1 linia | 3 linie + nowy param | ~20 linii (init + forward) |

A/B/C wszystkie cierpią na ten sam fundamentalny problem: **dłuższy prefix dla T5**. Ckpt 5.36 nauczył się attention dla prefixu `[V, M]`. Każda konkatenacja-based fuzja zmienia długość → drift nawet z perfect aux features.

## Wybór: E) Cross-attention z zero-init out projection

**Pomysł**: aux/kp NIE wydłużają sekwencji, tylko wstrzykują informację do V/M tokenów przez cross-attention. T5 widzi prefix dokładnie tej długości, na którą była trenowana.

```
V_enriched = V + CrossAttn(query=V, key=concat[A,K], value=concat[A,K])
```

gdzie wewnątrz `CrossAttn`, ostatnia projekcja `W_O` jest **zero-init**:

```python
nn.init.zeros_(self.aux_xattn.proj.weight)
nn.init.zeros_(self.aux_xattn.proj.bias)
```

Następnie joint mode używa `[V_enriched, M]` — dokładnie dual layout, forward identyczny z trenowanym ckpt na step 0.

### Matematyka

Niech $V \in \mathbb{R}^{T_V \times d}$, $A \in \mathbb{R}^{T_A \times d}$, $K \in \mathbb{R}^{T_K \times d}$. Cross-attention z zero-init out:

$$\text{XAttn}(V, [A;K]) = W_O \cdot \text{softmax}\!\left(\frac{(VW_Q)(C W_K)^\top}{\sqrt{d}}\right) (CW_V)$$

gdzie $C = [A;K] \in \mathbb{R}^{(T_A+T_K) \times d}$ i $W_O \leftarrow 0$.

**Step 0** (zero-init):
$$W_O = 0 \;\Rightarrow\; \text{XAttn} = 0 \;\Rightarrow\; V_{enriched} = V$$

Forward przez fusion path identyczny z dual → fusion_proj output identyczny z dual → T5 prefix identyczny z trenowanym → init BLEU = 5.36.

**Step k > 0**: gradient z generation loss płynie przez $V_{enriched} \to V$ + przez $W_O$. $W_O$ stopniowo opuszcza zero, aux/kp wlewają się do V. Pozostałe wagi ($W_Q, W_K, W_V$) random-init ale ich wpływ "blokowany" przez $W_O \approx 0$ na początku — bezpieczna ścieżka eksploracji gradient.

### Dlaczego to standard

- **Flamingo (DeepMind 2022)** — gated cross-attention layers między warstwami LLM, $\tanh(0)=0$ init bramki: LLM identyczny z frozen baseline na step 0
- **BLIP-2 (Salesforce 2023)** — Q-Former + cross-attention do LLM zamiast konkatenacji visual tokenów
- **LLaVA-1.5+** — projekcje visual jako "soft prompts", nie tokens (dla większych modeli)
- **DiT (Peebles & Xie 2022)** — adaLN-zero w block: zero-init parametry bramek, blok identity na step 0
- **ResNet zero-init last conv w bloku** (He et al.) — gradient buduje skip-connection sygnał stabilnie

Pattern: każde rozszerzenie trained modelu o nowe wejście / nowy moduł powinno startować jako **identity transformation**, gradient potem buduje non-zero contribution. Cross-attn z zero-init $W_O$ to konkretna realizacja.

## Plan implementacji

`spamo/mm_projector.py` ma już klasę `CrossAttention` (linia ~56+). Zmiany w `spamo/t5_slt.py`:

### 1. `__init__` (po istniejącym `aux_proj` / `kp_proj`)

```python
if self.aux_input_size > 0 or self.keypoint_dim > 0:
    from spamo.mm_projector import CrossAttention
    self.aux_xattn = CrossAttention(
        dim=self.inter_hidden,
        num_heads=8,
        qkv_bias=True,
    )
    # zero-init out projection — V_enriched = V on step 0
    nn.init.zeros_(self.aux_xattn.proj.weight)
    nn.init.zeros_(self.aux_xattn.proj.bias)
```

(Konkretne nazwy `proj.weight`/`proj.bias` zależą od struktury istniejącej `CrossAttention` — jeśli różne, target the actual output linear layer.)

### 2. Forward joint mode (zastąpić appending aux/kp do `parts`)

Aktualnie:
```python
parts = [V_per_sample, M_per_sample]
if self.aux_input_size > 0 and aux_outputs is not None:
    parts.append(aux_outputs[i, :aux_length[i], :])
if self.keypoint_dim > 0 and ...:
    parts.append(kp_outputs[i, :kp_length[i], :])
```

Po zmianie:
```python
# Zbierz aux/kp jako context (poza parts — nie wydłużają sekwencji)
context_streams = []
if self.aux_input_size > 0 and aux_outputs is not None:
    context_streams.append(aux_outputs)         # [B, T_A, d]
if self.keypoint_dim > 0 and kp_outputs is not None:
    context_streams.append(kp_outputs)          # [B, T_K, d]

if context_streams:
    context = torch.cat(context_streams, dim=1)  # [B, T_A+T_K, d]
    # V is built from per-sample visual_outputs; apply cross-attn batched
    V_enriched = V_batched + self.aux_xattn(V_batched, context, context)
    # use V_enriched in parts construction
else:
    V_enriched = V_batched

parts_per_sample = [V_enriched[i, :v_len[i]], M[i, :m_len[i]]]  # length = dual
```

Dokładny shape handling musi pasować do per-sample `pad_sequence` workflow już obecnego — możliwe że łatwiej zaaplikować cross-attn na batched `V_padded` przed pętlą po samples.

### 3. Masking

Cross-attention musi maskować padding w `context` (różne `T_A`, `T_K` per sample). Standard: attention_mask z `[B, 1, T_V, T_A+T_K]` shape, $-\infty$ na padding. Wyklucza padding tokeny aux/kp z attention pool.

## Oczekiwane behavior po implementacji

1. **init val BLEU = 5.36** (identyczne z dual ckpt — zero-init residual gwarantuje identity forward)
2. **Step 1-1000**: gradient buduje $W_O \neq 0$, aux/kp stopniowo wlewa się do V tokenów
3. **Cel długoterminowy**: BLEU > 5.36 — aux (hand_ViT) i kp (MediaPipe) dostarczają informacji nieobecnych w V (full ViT) i M (MAE motion), co powinno poprawić translation jakość
4. **Gradient stability**: brak nagłej eksplozji loss przy starcie (vs v25 init 0.91 = 5× spadek)

## Stosunek do innych rozwiązań w pracy

Ten design działa **niezależnie** od fixów dropout (refactor B w `mm_projector` / `tconv` / `t5_slt`) i text_embeds (`embed_tokens.mean` zamiast `encoder.last_hidden`). Wszystkie trzy są ortogonalne i wymagane:

- **Dropout fix** — żeby ckpt 1:1 ładował się do forka niezależnie od `fusion_dropout`
- **text_embeds fix** — żeby contrastive miał stabilny target podczas treningu
- **Cross-attn quad fusion** — żeby quad continuation startował z BLEU dual i mógł rosnąć
