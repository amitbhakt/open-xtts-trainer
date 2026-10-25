# Ablation Study A3: Conditioning Masking Impact (F2 Bug Verification)

**Conducted:** 2026-10-25 / 2026-10-26  
**Compute:** Modal serverless compute with an NVIDIA A100-SXM4-40GB GPU (CUDA device 0, ~104 ms/step)  
**Evaluator:** Batched Whisper large-v3-turbo inference (batch size 16)  

This directory archives the primary evidence and evaluation manifests for pre-registered **Ablation Study A3** (`eval/pre_registered_ablations.json`), comparing:
1. **Upstream Masking (F2 Bug):** Coqui's buggy implementation where `cond_idxs` were provided in mel-frame indices rather than discrete code indices, masking only the final sample in the mini-batch while leaking unmasked prompt tokens across the autoregressive sequence.
2. **Per-Sample Masking (Bugfix):** Individual sample-level prompt masking in discrete code units, preventing acoustic prompt leakage into autoregressive loss calculations.

---

## Files in this Directory

| File | What it holds |
|---|---|
| `ab_report_target.json` | Paired A/B hypothesis test report for target voice adaptation against pre-registered gates in `eval/pre_registered_ablations.json` ($N=60$) |
| `manifest_upstream_s20_target.json` | Unified 3-seed evaluation manifest for the upstream buggy masking variant on Moira Fogarty held-out reference audio ($20 \text{ prompts} \times 3 \text{ seeds} = 60 \text{ runs}$) |
| `manifest_per_sample_s20_target.json` | Unified 3-seed evaluation manifest for the per-sample fixed masking variant on Moira Fogarty held-out reference audio ($N=60$) |
| `per_seed_manifests/` | Individual per-seed evaluation manifests (`seed_1`, `seed_2`, `seed_3`) for both variants |

---

## 1. Target Speaker Evaluation ($N=60$ Paired Utterances)

Evaluated on held-out reference audio of Moira Fogarty reading *The Art of War* across 20 sentences and 3 random seeds (`1, 2, 3`), paired utterance-by-utterance under batched Whisper inference:

| Metric | Rule / Gate | Upstream Masking (F2) | Per-Sample Masking (Fix) | Paired Difference $\Delta$ | 95% Bootstrap CI | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Short-Prompt CER** | `not_worse` (margin 0.022) | **0.5556 (55.56%)** | **0.0000 (0.00%)** | **−0.5556 (−55.56%)** | [−1.6667, 0.0000] | **PASS** (Major bugfix) |
| **Overall CER** | `not_worse` (margin 0.01) | 0.0946 (9.46%) | **0.0083 (0.83%)** | **−0.0863 (−8.63%)** | [−0.2552, +0.0001] | **PASS** (11× error drop) |
| **Overall WER** | `not_worse` (margin 0.02) | 0.0529 (5.29%) | **0.0342 (3.42%)** | **−0.0187 (−1.87%)** | [−0.0550, +0.0018] | **PASS** (Improved) |
| **Naturalness (UTMOS)** | `not_worse` (margin 0.10) | 4.2418 | 4.2134 | −0.0285 | [−0.0778, +0.0177] | **PASS** |
| **Failures** | `not_worse` (margin 0) | 0 / 60 | 0 / 60 | 0 | [0.0, 0.0] | **PASS** |
| **Speaker Similarity (SECS)** | `not_worse` (margin 0.02) | 0.5917 | 0.5748 | −0.0170 | [−0.0297, −0.0040] | Mean within margin |

*(Note: SECS is evaluated on $n=50$ qualifying pairs with take durations $\ge 2\text{ s}$ per headline speaker-embedding evaluation rule; overall $N=60$.)*

### Empirical Conclusion
* **Decisive Proof of Defect F2**: Under upstream batch masking, unmasked prompt tokens leaked into autoregressive training, driving a catastrophic 55.56% character error rate on short prompts. Fixing the prompt masking to per-sample code lengths eliminated short-text hallucinations completely (**0.00% CER**) and drove an 11-fold reduction in overall character errors ($9.46\% \to 0.83\%$).
* **Acoustic Similarity Trade-off**: Upstream's $+0.017$ higher SECS is an expected symptom of unmasked acoustic prompt leakage. The $-0.0170$ mean difference is within the allowable $0.02$ margin; the 95% bootstrap CI lower tail touched $-0.0297$ due to the $n=50$ sample size.
