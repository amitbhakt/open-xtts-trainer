# Ablation Study A3: Conditioning Masking Impact (F2 Bug Verification)

**Conducted:** 2026-10-25 / 2026-10-26  
**Compute:** Modal serverless compute with an NVIDIA A100-SXM4-40GB GPU (CUDA device 0, ~104 ms/step)  
**Evaluator:** Batched Whisper `openai/whisper-large-v3` inference (batch size 16)  

This directory archives the primary evidence and evaluation manifests for pre-registered **Ablation Study A3** (`eval/pre_registered_ablations.json`), comparing:
1. **Upstream Masking (F2 Bug):** Coqui's buggy implementation where `cond_idxs` were provided in mel-frame indices rather than discrete code indices, masking only the final sample in the mini-batch while leaking unmasked prompt tokens across the autoregressive sequence.
2. **Per-Sample Masking (Bugfix):** Individual sample-level prompt masking in discrete code units, preventing acoustic prompt leakage into autoregressive loss calculations.

---

## Files in this Directory

| File | What it holds |
|---|---|
| `ab_report_target.json` | Paired A/B hypothesis test report for target voice adaptation against pre-registered gates in `eval/pre_registered_ablations.json` ($N=60$, `passed: false` due to SECS CI lower bound) |
| `manifest_upstream_s20_target.json` | Unified 3-seed evaluation manifest for the upstream buggy masking variant on Moira Fogarty held-out reference audio ($20 \text{ prompts} \times 3 \text{ seeds} = 60 \text{ runs}$) |
| `manifest_per_sample_s20_target.json` | Unified 3-seed evaluation manifest for the per-sample fixed masking variant on Moira Fogarty held-out reference audio ($N=60$) |
| `ab_report_base.json` | Paired A/B hypothesis test report for zero-shot preservation on 5 unseen CMU ARCTIC base speakers against pre-registered gates ($N=300$, `passed: true`) |
| `manifest_upstream_s20_base.json` | Unified 3-seed evaluation manifest for the upstream buggy masking variant on 5 CMU ARCTIC base speakers ($5 \times 20 \times 3 = 300 \text{ runs}$) |
| `manifest_per_sample_s20_base.json` | Unified 3-seed evaluation manifest for the per-sample fixed masking variant on 5 CMU ARCTIC base speakers ($N=300$) |
| `per_seed_manifests/` | Individual per-seed evaluation manifests (`seed_1`, `seed_2`, `seed_3`) for target voice ($N=60$) and base voices ($N=300$) across both variants (RFC 8259 compliant, empty aggregates serialized as `null`) |

---

## 1. Target Speaker Evaluation ($N=60$ Paired Utterances)

Evaluated on held-out reference audio of Moira Fogarty reading *The Art of War* across 20 sentences and 3 random seeds (`1, 2, 3`), paired utterance-by-utterance under batched Whisper `openai/whisper-large-v3` inference:

| Metric | Rule / Gate | Upstream Masking (F2) | Per-Sample Masking (Fix) | Paired Difference $\Delta$ | 95% Bootstrap CI | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Short-Prompt CER** | `not_worse` (margin 0.022) | **0.5556 (55.56%)** | **0.0000 (0.00%)** | **−0.5556 (−55.56%)** | [−1.6667, 0.0000] | PASS |
| **Overall CER** | `not_worse` (margin 0.01) | 0.0946 (9.46%) | **0.0083 (0.83%)** | **−0.0863 (−8.63%)** | [−0.2552, +0.0001] | PASS |
| **Overall WER** | `not_worse` (margin 0.02) | 0.0529 (5.29%) | **0.0342 (3.42%)** | **−0.0187 (−1.87%)** | [−0.0550, +0.0018] | PASS |
| **Naturalness (UTMOS)** | `not_worse` (margin 0.10) | 4.2418 | 4.2134 | −0.0285 | [−0.0778, +0.0177] | **PASS** |
| **Failures** | `not_worse` (margin 0) | 0 / 60 | 0 / 60 | 0 | [0.0, 0.0] | **PASS** |
| **Speaker Similarity (SECS)** | `not_worse` (margin 0.02) | 0.5917 | 0.5748 | −0.0170 | [−0.0297, −0.0040] | **FAIL** |

*(Note: SECS is evaluated on $n=50$ qualifying pairs with take durations $\ge 2\text{ s}$ per headline speaker-embedding evaluation rule; overall $N=60$.)*

### Reading the target results
* **The SECS gate fails** (`passed: false` in `ab_report_target.json`): similarity to the target speaker is 0.0170
  lower with the fix, and the 95% interval $[-0.0297, -0.0040]$ extends beyond the 0.02 margin.
* **No detectable intelligibility difference.** The 0.5556 short-prompt CER is a single take (CER 5.0) out of 9
  short takes; its interval $[-1.6667, 0.0000]$ includes zero, and median WER and CER are 0.0 for both variants.
* This regime was not in the pre-registration and is reported as an additional analysis.

---

## 2. Zero-Shot Base Speaker Preservation ($N=300$ Paired Utterances)

Evaluated across all 5 unseen CMU ARCTIC voices (`arctic_slt`, `arctic_clb`, `arctic_bdl`, `arctic_awb`, `arctic_ksp`) across 20 sentences and 3 random seeds (`1, 2, 3`), paired utterance-by-utterance under batched Whisper `openai/whisper-large-v3` inference:

| Metric | Rule / Gate | Upstream Masking (F2) | Per-Sample Masking (Fix) | Paired Difference $\Delta$ | 95% Bootstrap CI | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Short-Prompt CER** | `not_worse` (margin 0.022) | **6.3111 (631.11%)** | **0.1741 (17.41%)** | **−6.1370 (−613.70%)** | [−16.7003, −0.2667] | PASS |
| **Overall CER** | `not_worse` (margin 0.01) | **0.9601 (96.01%)** | **0.0374 (3.74%)** | **−0.9228 (−92.28%)** | [−2.5863, −0.0368] | PASS |
| **Overall WER** | `not_worse` (margin 0.02) | **0.8424 (84.24%)** | **0.0422 (4.22%)** | **−0.8002 (−80.02%)** | [−2.3312, −0.0146] | PASS |
| **Naturalness (UTMOS)** | `not_worse` (margin 0.10) | 4.1150 | **4.1273** | **+0.0123** | [−0.0187, +0.0441] | PASS |
| **Failures** | `not_worse` (margin 0) | 7 / 300 | **3 / 300** | **−4** | — | PASS |
| **Speaker Similarity (SECS)** | `not_worse` (margin 0.02) | 0.5690 | 0.5628 | **−0.0062** | [−0.0110, −0.0014] | PASS |

*(Note: SECS is evaluated on $n=255$ qualifying pairs with take durations $\ge 2\text{ s}$ per headline speaker-embedding evaluation rule; overall $N=300$.)*

### Reading the unseen-voice results
* **Decision: all pre-registered gates pass** (`passed: true` in `ab_report_base.json`), so the fix is retained.
* **The means are dominated by single takes.** One upstream take on `arctic_ksp` (seed 1, `en_51`) has WER 221
  (the model kept generating speech), and one on `arctic_bdl` (seed 2, `en_53`) has WER 11; together they
  contribute 0.77 of the 0.84 mean WER. Medians are 0.0 for both.

| Statistic | Upstream (F2) | Per-sample (fix) | Fisher $p$ (exploratory) |
| :--- | :---: | :---: | :---: |
| Mean WER without takes above WER 1.0 | 0.053 | 0.042 | — |
| Runaway takes (WER > 1.0) | 4 / 300 | 0 / 300 | 0.12 |
| Takes flagged too long | 7 / 300 | 3 / 300 | 0.34 |
| Short prompts with CER > 0.5 | 9 / 45 | 2 / 45 | 0.05 |

* **What this supports:** fewer badly failed short prompts and no runaway takes with the fix; otherwise similar
  intelligibility. Consistent with F2 causing occasional runaway generation, but suggestive at $N=300$, not
  established. The Fisher tests were chosen after seeing the data.

### Deviations from the pre-registration
1. `eval/pre_registered_ablations.json` specifies "Paired N=300 per seed, across 3 seeds" (900 pairs); this run
   has 300 pairs in total (5 voices × 20 prompts × 3 seeds). Re-evaluating the existing checkpoints would close it.
2. Duration-ratio failures are listed as a primary metric; they are reported here but have no gate beyond the
   shared failures gate.

---

## 3. Provenance and Environment

- **Execution Environment:** Modal serverless execution (`scripts/modal_run_eval.py`) mounting persistent volumes for checkpoints and cache.
- **Base Container Commit:** The base container clones from GitHub at `PINNED_COMMIT = "2f1f67ee5261aabd37579d704e62fab4ac177bc4"`, overlaying working tree modifications with deterministic timestamping (`modal overlay`).
- **Data & Locks:** Target reference clips verified against `docs/evidence/run6_target_speaker/evidence/target_voices/target_voices.lock.json`; evaluation set verified against `eval/dataset/eval_set.json` (SHA-256 `86c0183f...`).
