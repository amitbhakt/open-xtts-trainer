# Colab run 4: Phases 6–7 with real weights (2026-10-24)

Summarized from the evidence zip the run produced (`docs/evidence/phase6`, `docs/evidence/phase7` on the
Colab machine). The raw files stay out of the repository; the numbers below are copied from them.

The run used the code uploaded after 1838b4c (the run-4 planner fix), before source pinning, on a single
Colab GPU. Every training run here is a
smoke test on the 8-clip sample set (6 LJSpeech clips, 2 JFK clips). It shows that each stage loads the real
weights, trains, validates and saves its best checkpoint. It says nothing about quality.

## Phase 6: data preparation

**Sample set: prepared and validated with zero edits.** JFK (11.0 s) now gives 2 clips covering 89% of the
audio (`min_s` 3.0): "And so, my fellow Americans: ask not" and "what your country can do for you, ask what
you can do for your country." Run 3 got one clip here. Validation with re-alignment passed, with the
short-clip warning only (0/8 clips under 1.5 s).

**Shifted-cut check, word scores (MMS_FA, `test_mms_jfk_alignment_and_shifted_cut`).** With `min_s` 2.5 the
test prepares 3 clips, then moves the first clip's end by ±1 s and re-aligns every clip as validation does.

| Clip 1 ("And so, my fellow Americans: ask") | Lowest word score | Words below 0.2 | Validation |
|---|---|---|---|
| As cut | 0.866 | 0 | passes |
| End moved +1 s | 0.012 ('ask') | 5 | `alignment_mismatch` |
| End moved −1 s | 0.018 ('my') | 5 | `alignment_mismatch` |

Across all three unshifted clips, the lowest re-aligned word score is 0.54 ('what'). The word 'for', which
scored 0.12 when the whole 11 s file was aligned, scores 0.66 once it is re-aligned within its own clip.
Unexplained speech is 0.0 s on every clean clip (0.04 s on the +1 s clip).

What this supports: the rejection thresholds (hard floor 0.05; at most one word below 0.2) sit well inside
the gap between clean clips (≥ 0.54) and clips cut 1 s wrong (≤ 0.02, five weak words). This is one
recording, so it supports the thresholds without proving them for other speakers or languages.

## Phase 7: training smoke runs

**Stage 2, GPT** (6 train / 2 eval clips, 2 epochs, effective batch 2, fp32). Eval loss went from 3.200
before training to 2.972 at the best step (9); text CE 0.022, mel CE 2.950. The GPT loaded strictly from
`xtts_base/model.pth`, and the export completed.

**Stage 3, HiFi-GAN** (60 epochs, batch 4, discriminator warm-up 40 steps, CUDA mixed precision).
Validation mel L1 went from 1.553 (stock decoder) to a best of 1.350 at step 106, 13% lower, and then rose to
1.367 by step 120. So the best-by-validation checkpoint matters on small data. The first four measurements
(steps 2–8) are exactly 1.553001582622528, so the generator did not change for about 8 steps. A likely
cause is the fp16 loss scaler skipping its first steps on overflow; this has not been checked.

**Stage 1, DVAE** (20 epochs, batch 4, normal and frozen-codebook modes, from `xtts_base/dvae.pth`). Code drift
is the share of validation code positions where the fine-tuned DVAE picks a different code than the stock one.

| Mode | Val reconstruction (stock 0.243) | Active codes (stock 136) | Perplexity (stock 114.6) | Code drift at best step |
|---|---|---|---|---|
| Normal | 0.105 | 141 | 114.3 | 81.5% |
| Frozen codebook | 0.116 | 147 | 123.2 | 81.5% |

Neither mode collapsed the codebook. In both, though, 81% of codes changed for the same audio, and drift rose
steadily with training (normal: 3.5% at step 2, 46% at step 20, 81% at step 40). Freezing the quantizer did not
reduce drift on real speech, because the trainable encoder moves to different codes (synthetic data gave
47%). A GPT trained on stock codes would see mostly different tokens, so after any Stage 1 run, Stage 2 must
be retrained. Stage 1 stays off by default.
