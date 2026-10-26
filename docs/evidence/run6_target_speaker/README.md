# Run 6: Target Speaker Voice Adaptation and Paired A/B Evaluation (2026-10-25)

Conducted on Modal serverless compute with an NVIDIA A100-SXM4-40GB GPU. Records Stage 2 target voice adaptation and paired A/B evaluation. Note: Evaluation manifests were generated under sequential Whisper evaluation prior to batched evaluation refactoring.

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
| `evidence/validation_report.json` | Basic validation report confirming 85 clips passed schema and duration checks with zero errors; re-alignment and holdout checks were omitted on this unpartitioned `dataset.jsonl` artifact |
| `evidence/training_summary.json` | Stage 2 GPT fine-tuning summary (10 epochs, eval loss $3.947 \to 3.102$) |
| `evidence/target_voices/` | Held-out reference audio clips strictly sequestered from training |
| `evidence/identity_canary.xml` | Phase 5 Canary 2 JUnit test report verifying `test_identity_canary_released_model` passed |

---

## 1. Target Speaker Voice Adaptation ($N=60$)

Evaluated on held-out reference audio of Moira Fogarty reading *The Art of War* across 20 sentences and 3 random seeds (`42, 123, 999`), paired utterance-by-utterance against stock XTTS-v2:

| Metric | Rule / Gate | Stock XTTS-v2 | Tuned Model | Paired Difference $\Delta$ | 95% Bootstrap CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SECS** ($n=51$) | `better` (margin 0.0) | 0.5545 | **0.5875** | **+0.0329** | [+0.0191, +0.0480] | **PASS** |
| **WER** | `not_worse` (margin 0.02) | 0.0383 | **0.0375** | **−0.0008** | [−0.0096, +0.0079] | **PASS** |
| **CER** | `not_worse` (margin 0.01) | 0.0098 | 0.0105 | +0.0008 | [−0.0045, +0.0060] | **PASS** |
| **UTMOS** | `not_worse` (margin 0.10) | 4.2589 | 4.2149 | −0.0439 | [−0.0831, −0.0065] | **PASS** |
| **Duration Ratio** | Informational | 0.9876 | 0.9649 | −0.0227 | [−0.0528, +0.0073] | — |
| **Failures** | `not_worse` (margin 0) | 0 / 60 | 0 / 60 | 0 | [0.0, 0.0] | **PASS** |

*(Note: SECS is evaluated on $n=51$ qualifying pairs with take durations $\ge 2\text{ s}$ per headline speaker-embedding evaluation rule; overall $N=60$.)*

**Conclusion (target speaker):** All 5 pre-registered adaptation gates passed within tolerance (SECS gain of $+0.0329$ passing the `better` rule; WER improved; CER $+0.0008$ and UTMOS $-0.0439$ stayed well within their respective 0.01 and 0.10 tolerance margins). Target speaker similarity improved with high statistical confidence (95% bootstrap CI strictly positive $[+0.0191, +0.0480]$ across $n=51$ pairs). While audio naturalness (UTMOS) decreased slightly by 0.0439 (95% CI $[-0.0831, -0.0065]$), this regression remained within the allowable 0.10 tolerance margin.

---

## 2. Preservation on Unseen Base Voices ($N=300$)

Evaluated across all 5 unseen CMU ARCTIC voices (`arctic_slt`, `arctic_clb`, `arctic_bdl`, `arctic_awb`, `arctic_ksp`), 20 sentences per voice, 3 random seeds ($300 \text{ paired utterances}$):

| Metric | Rule / Margin | Stock XTTS-v2 | Tuned Model | Paired Difference $\Delta$ | 95% Bootstrap CI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **WER** | `not_worse` (0.02) | 0.0697 | **0.0419** | **−0.0278** | [−0.0530, −0.0067] | **PASS** (improved) |
| **CER** | `not_worse` (0.01) | 0.1266 | **0.0452** | **−0.0813** | [−0.1821, −0.0077] | **PASS** (improved) |
| **UTMOS** | `not_worse` (0.10) | 4.0517 | **4.1467** | **+0.0950** | [+0.0603, +0.1305] | **PASS** (improved) |
| **Failures** | `not_worse` (0) | 6 failures | **3 failures** | **−3** | — | **PASS** (halved) |
| **SECS** ($n=253$) | `not_worse` (0.02) | 0.5743 | 0.5507 | −0.0236 | [−0.0290, −0.0181] | **FAIL** |

*(Note: SECS is evaluated on $n=253$ qualifying pairs with take durations $\ge 2\text{ s}$ per headline speaker-embedding evaluation rule; overall $N=300$.)*

**Result: the preservation comparison fails** (`passed: false` in `ab_report_preservation.json`). Its
pre-registered SECS gate fails: similarity to unseen voices fell by 0.0236, and the whole 95% interval lies beyond
the 0.02 margin. Adapting to one speaker reduced zero-shot cloning of others; that is the regression this gate
exists to catch, so it is reported as a failure, not a near miss.

The WER, CER and failure gates pass, but their mean differences are dominated by a few runaway takes: stock's mean
CER of 0.1266 comes mostly from one take with CER 12, and both models have a median WER and CER of 0.0. Excluding
takes with WER above 1.0, mean WER is 0.057 (stock) and 0.042 (tuned). The tuned model had fewer runaway takes
(3 failures vs 6).

---

## 3. Phase 5 Canary Check

Canary test `test_identity_canary_released_model` was executed on the released weights and exported upstream format, confirming that:
1. Exporting the released checkpoint preserves the model weights and exported tensor shapes.
2. Inference output on a fixed test prompt and seed matches between the checkpoint and exported model.
3. Test output recorded in `evidence/identity_canary.xml` ($1 \text{ test passed in } 92.0\text{ s}$).
