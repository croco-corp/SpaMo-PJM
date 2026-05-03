## 4. Results *(Wojt — main responsibility)*

> **Headline number to put in big type at the top of this section:**
> *“Best PJM SLT model so far: **[ARCH]** — BLEU‑4 = **__**, ROUGE‑L = **__** on MS test.”*

### 4.1 Main results table

Architectures × splits × transfer learning. **One table to rule them all.**

> Numbers in **bold** = best per column; numbers in *italic* = second‑best.
> All scores reported on the **test** set (single run, beam = 5).
> ↑ = higher is better.

| Architecture | Pretraining | Split | BLEU‑1 ↑ | BLEU‑2 ↑ | BLEU‑3 ↑ | BLEU‑4 ↑ | ROUGE‑L ↑ | (BLEURT ↑) |
|---|---|---|---|---|---|---|---|---|
| SpaMo (baseline) | — (from scratch) | MS | _ | _ | _ | _ | _ | _ |
| SpaMo (baseline) | — (from scratch) | SI | _ | _ | _ | _ | _ | _ |
| SpaMo (baseline) | **PHOENIX‑2014T** (transfer) | MS | _ | _ | _ | _ | _ | _ |
| SpaMo (baseline) | **PHOENIX‑2014T** (transfer) | SI | _ | _ | _ | _ | _ | _ |
| Modified SpaMo (ours) | — (from scratch) | MS | _ | _ | _ | _ | _ | _ |
| Modified SpaMo (ours) | — (from scratch) | SI | _ | _ | _ | _ | _ | _ |
| Modified SpaMo (ours) | **PHOENIX‑2014T** (transfer) | MS | _ | _ | _ | _ | _ | _ |
| Modified SpaMo (ours) | **PHOENIX‑2014T** (transfer) | SI | _ | _ | _ | _ | _ | _ |
| Uni‑Sign | — (from scratch) | MS | _ | _ | _ | _ | _ | _ |
| Uni‑Sign | — (from scratch) | SI | _ | _ | _ | _ | _ | _ |
| Uni‑Sign | CSL‑News (transfer) | MS | _ | _ | _ | _ | _ | _ |
| Uni‑Sign | CSL‑News (transfer) | SI | _ | _ | _ | _ | _ | _ |
| **Uni‑Sign** | **OpenASL (transfer) ← best** | **MS** | _ | _ | _ | _ | _ | _ |
| **Uni‑Sign** | **OpenASL (transfer) ← best** | **SI** | _ | _ | _ | _ | _ | _ |

> If the *baseline* SpaMo / Modified SpaMo also have a transfer‑learning variant in W&B, just add two more rows. Keep the row order: **architecture → pretraining → split** so the eye scans top‑to‑bottom by model.

### 4.2 Companion charts (two small charts beside the table)

**Chart A — “Architecture comparison, MS test, BLEU‑4”**
Grouped bar: x = architecture, y = BLEU‑4, one bar per *(architecture, pretraining)* pair — i.e. SpaMo: scratch / Phoenix; Mod‑SpaMo: scratch / Phoenix; Uni‑Sign: scratch / CSL‑News / OpenASL. Use a hatch pattern or shade for the **OpenASL bar** so the “best transfer source” jumps out.

**Chart B — “SI penalty”**
For each architecture, plot Δ(BLEU‑4) = MS − SI as a horizontal bar. Smaller bar = better signer generalisation.
Optional: same plot but on ROUGE‑L for cross‑metric robustness.

### 4.3 Qualitative examples box

A 3‑row mini‑table the audience can actually read at the poster:

| Reference | SpaMo prediction | Mod‑SpaMo prediction | Uni‑Sign (transfer) prediction |
|---|---|---|---|
| _ | _ | _ | _ |
| _ | _ | _ | _ |
| _ | _ | _ | _ |

Pick: 1 easy example, 1 medium, 1 failure case. This is the bit that walks people through the poster.

---

## 5. Discussion *(Wojt — main responsibility)*

Three short paragraphs — one bullet each, ≤30 words.

### 5.1 SI vs MS — the speaker‑independence cost
- Across all architectures, BLEU‑4 drops by **≈ Δ** going from MS to SI.
- Confirms PJM SLT generalisation to unseen signers is the open problem, not in‑domain accuracy.
- Architecture with the smallest gap: **[___]**.

### 5.2 Transfer learning: which source language helps PJM the most?
- We compared three transfer sources: **PHOENIX‑2014T** (German Sign Language, weather domain — for SpaMo / Mod‑SpaMo), **CSL‑News** (Chinese SL, news, 1985 h) and **OpenASL** (American SL, large open‑domain — for Uni‑Sign).
- **OpenASL was the strongest source** for Uni‑Sign on PJM, beating both CSL‑News and from‑scratch by Δ BLEU‑4 = **__**. Hypothesis: OpenASL’s open‑domain, multi‑signer composition is a closer distributional match to PJM than the news‑only / weather‑only alternatives.
- Effect is **larger on SI** than on MS — pretrained low‑level visual features generalise across signers, not across language semantics.
- Caveat: full fine‑tune (Phase 3) is needed; LoRA (Phase 2) on top of CSL‑News collapsed to degenerate output (“I, I, I, …”) — *include this as a brief failure note, it is honest and informative.*

### 5.3 Architecture takeaway
- Pose‑first encoders (Uni‑Sign STGCN) **beat / match** RGB‑only fusion (SpaMo) on PJM, despite less mature pose extraction for Polish signers.
- Modified SpaMo improves over baseline SpaMo on **[which metric / which split]** by **Δ**, suggesting [your modification] is the right direction.
- Open question for next work: **[one concrete future direction — e.g. pose+RGB hybrid, larger PJM continued pretraining, or scaling signer count in the dataset]**.

---

## 6. References / footer (small text along the bottom)

- Uni‑Sign — Li et al., *ICLR 2025* (arXiv:2501.15187).
- SpaMo — [CITE].
- mT5 — Xue et al., *NAACL 2021*.
- BLEU — Papineni et al., 2002 · ROUGE — Lin, 2004 · BLEURT — Sellam et al., 2020.
- This work — undergraduate engineering thesis, [Affiliation], 2026.
- QR code → repo / dataset card.

---

## Appendix A — How to fill the tables from W&B (notes for Wojt)

- **Use test scores, not dev.** Final test eval in Uni‑Sign lands in `log.txt` as `final_test_*` (see PHASE3 report). For the older runs without that flag, take the test eval logged in the last epoch of W&B.
- For each (architecture × pretraining × split) cell, **one run** in W&B → copy BLEU‑1..4 and ROUGE‑L. If you logged multiple seeds, pick the **median run** and put the std in the caption (“±” notation), not in the cell.
- BLEURT — only fill it where you actually have it; **leave “—” elsewhere** rather than “0”. A missing number is more honest than a fake one.
- Round all BLEU/ROUGE to **2 decimals**, BLEURT to **3**. Consistency reads better on a poster than precision.
- Bold = best per column, italic = 2nd best. Compute this *after* all cells are filled, not row‑by‑row.

## Appendix B — Quick checklist before printing

- [ ] All `_` placeholders in §2, §3, §4, §5 replaced with real numbers.
- [ ] Architecture diagram exported at ≥ 300 dpi (or as vector SVG/PDF).
- [ ] Bar charts use the **same colour per architecture** across §4.1, A and B.
- [ ] Qualitative examples checked for offensive / private content.
- [ ] Footer references match in‑text [CITE] markers.
- [ ] Single sentence at the very top of §4 contains the headline number.
- [ ] QR code resolves and points to the correct repo.