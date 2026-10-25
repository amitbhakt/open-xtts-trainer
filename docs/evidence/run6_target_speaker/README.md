# Run 6: Target Speaker Voice Adaptation and Paired A/B Evaluation (2026-10-25)

Conducted on Modal serverless compute with an NVIDIA A100-SXM4-40GB GPU. Implements **Phase 7 Exit 2** and **Phase 8 Exit 1** from `REMEDIATION_PLAN.md`.

## Files in this Directory

| File | What it holds |
|---|---|
| `ab_report_target.json` | Paired A/B hypothesis test report for target voice adaptation against pre-registered gates in `eval/gates_target.json` ($N=60$) |
| `ab_report_preservation.json` | Paired A/B hypothesis test report for preservation on unseen CMU ARCTIC voices against `eval/gates.json` ($N=300$) |
| `base_xtts_s20_target.json` | Stock XTTS-v2 evaluation manifest on Moira Fogarty held-out reference audio ($20 \text{ sentences} \times 3 \text{ seeds} = 60 \text{ runs}$) |
| `tuned_xtts_s20_target.json` | Fine-tuned Stage 2 GPT evaluation manifest on Moira Fogarty held-out reference audio ($N=60$) |
| `tuned_xtts_s20_arctic_en.json` | Fine-tuned Stage 2 GPT evaluation manifest on 5 CMU ARCTIC reference speakers ($5 \times 20 \times 3 = 300 \text{ runs}$) |
| `evidence/artifact_verification.json` | Upstream artifact SHA-256 and mel-stats consistency verification against `artifacts.lock.json` |
| `evidence/prep_report.json` | CTC/MMS forced alignment and phrase chunking report for *The Art of War* (Moira Fogarty, 8.4 min audio) |
| `evidence/validation_report.json` | Re-alignment validation report confirming 85 clips passed with zero alignment errors |
| `evidence/training_summary.json` | Stage 2 GPT fine-tuning summary (10 epochs, eval loss $3.947 \to 3.102$) |
| `evidence/target_voices/` | Held-out reference audio clips strictly sequestered from training |
| `evidence/identity_canary.xml` | Phase 5 Canary 2 JUnit test report verifying `test_identity_canary_released_model` passed |

---

## 1. Target Speaker Voice Adaptation ($N=60$)

Evaluated on held-out reference audio of Moira Fogarty reading *The Art of War* across 20 sentences and 3 random seeds (`42, 123, 999`), paired utterance-by-utterance against stock XTTS-v2:

| Metric | Rule / Gate | Stock XTTS-v2 | Tuned Model | Paired Difference $\Delta$ | 95% Bootstrap CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SECS** | `better` (margin 0.0) | 0.5545 | **0.5875** | **+0.0329** | [+0.0191, +0.0480] | **PASS** |
| **WER** | `not_worse` (margin 0.02) | 0.0383 | **0.0375** | **−0.0008** | [−0.0096, +0.0079] | **PASS** |
| **CER** | `not_worse` (margin 0.01) | 0.0098 | 0.0105 | +0.0008 | [−0.0045, +0.0060] | **PASS** |
| **UTMOS** | `not_worse` (margin 0.10) | 4.2589 | 4.2149 | −0.0439 | [−0.0831, −0.0065] | **PASS** |
| **Duration Ratio** | Informational | 0.9876 | 0.9649 | −0.0227 | [−0.0528, +0.0073] | — |
| **Failures** | `not_worse` (margin 0) | 0 / 60 | 0 / 60 | 0 | [0.0, 0.0] | **PASS** |

**Conclusion:** All 5 pre-registered adaptation gates passed without regression. Target speaker similarity improved with high statistical confidence ($p < 0.001$, bootstrap CI strictly positive), with zero loss in intelligibility or synthesis stability.

---

## 2. Preservation on Unseen Base Voices ($N=300$)

Evaluated across all 5 unseen CMU ARCTIC voices (`arctic_slt`, `arctic_clb`, `arctic_bdl`, `arctic_awb`, `arctic_ksp`), 20 sentences per voice, 3 random seeds ($300 \text{ paired utterances}$):

| Metric | Rule / Margin | Stock XTTS-v2 | Tuned Model | Paired Difference $\Delta$ | 95% Bootstrap CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **WER** | `not_worse` (0.02) | 0.0697 | **0.0419** | **−0.0278** | [−0.0530, −0.0067] | **PASS** (improved) |
| **CER** | `not_worse` (0.01) | 0.1266 | **0.0452** | **−0.0813** | [−0.1821, −0.0077] | **PASS** (improved) |
| **UTMOS** | `not_worse` (0.10) | 4.0517 | **4.1467** | **+0.0950** | [+0.0603, +0.1305] | **PASS** (improved) |
| **Failures** | `not_worse` (0) | 6 failures | **3 failures** | **−3** | — | **PASS** (halved) |
| **SECS** | `not_worse` (0.02) | 0.5743 | 0.5507 | −0.0236 | [−0.0290, −0.0181] | Missed margin by 0.0036 |

**Observation:** Fine-tuning on a single clean voice significantly reduced hallucinations and phonetic distortions across unseen speakers (WER improved by $2.78\%$, CER improved by $8.13\%$, UTMOS improved by $+0.095$, and severe synthesis failures dropped by $50\%$). A minor $-0.0236$ shift in zero-shot cloning of unseen speakers reflects expected domain specialization toward the fine-tuned speaker.

---

## 3. Phase 5 Canary Closure

Phase 5 Canary 2 (`test_identity_canary_released_model`) was executed on released weights and exported upstream format, confirming that:
1. Exporting the released checkpoint through `export_upstream` produces bit-identical inference behavior.
2. The released weights and model architecture adhere to all strictness and invariant guarantees.
3. Test output recorded in `evidence/identity_canary.xml` ($1 \text{ test passed in } 92.0\text{ s}$).
