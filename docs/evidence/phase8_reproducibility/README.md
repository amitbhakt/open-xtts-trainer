# Phase 8 Baseline Statistical Agreement & Cross-Revision Verification (Exit Criterion 2)

**Conducted:** 2026-10-26  
**Compute:** Modal serverless compute with an NVIDIA A100-SXM4-40GB GPU (`ap-F820RbKzKYLK3kf3JiU0dB`)  
**Total Runtime:** 57.6 minutes across all 5 evaluation languages  
**Evaluator:** Batched Whisper `openai/whisper-large-v3` (batch size 16), ECAPA2 (`speechbrain/spkrec-ecapa-voxceleb`), UTMOS (`sarulab-speech/utmos-strong-learner-epoch68`)  
**Scope:** Full multilingual baseline re-evaluation across 5 languages (`en`, `es`, `fr`, `de`, `hi`), 12 sentences $\times$ 5 CMU ARCTIC voices $\times$ 3 random seeds (`42, 123, 999`) = **900 utterances total**.

---

## 1. Goal and Verification Protocol

As specified in the baseline verification protocol (§5 Phase 8 Exit Criterion 2, as documented in `eval/baselines/base_xtts_s12_index.json` and `docs/evidence/run5/README.md`):
> **Phase 8 Exit Criterion 2:** Re-run the baseline evaluation harness and confirm all published baseline metrics reproduce within their 95% confidence interval bounds:
> $$|\mu_1 - \mu_2| \le 1.96 \times \sqrt{\mathrm{SE}_1^2 + \mathrm{SE}_2^2}$$

This evaluation is a cross-revision statistical comparison measuring agreement within pre-registered 95% confidence bounds on the current revision against the published baseline (`base_xtts_s12_index.json`). Because the code fingerprint and batch size differ from the original publication run (which used batch size 1 and commit `1c33371`), the strict same-input comparator reports `reproducible: false` due to input metadata mismatches. This test verifies that:
1. Model synthesis and metric calculation achieve statistical agreement within 95% confidence bounds across independent environment boots.
2. Changes to codebase infrastructure, lint rules, and evaluation batching did not introduce numerical drift or regressions.
3. Every baseline metric reported in `eval/baselines/base_xtts_s12_index.json` agrees within its 95% confidence bounds on the current revision.

---

## What this rerun does and does not show

* **Means agree within the 95% tolerance for every language**, so the published baseline numbers stand.
* **It is not bit-identical.** 852 of 900 takes produced identical audio (English 172, Spanish 170, French 171,
  German 167, Hindi 172, each of 180); 48 differ. Runs 1–2 were bit-identical within one environment; this rerun
  used a later commit and a new GPU session.
* **One Spanish take explains the Spanish difference.** In the rerun, an `arctic_ksp` take (seed 999) became a
  repetition loop: 8.96 s → 17.17 s, the sentence spoken twice, WER 20.8. That single take moves Spanish mean WER
  from 0.099 to 0.213 and CER from 0.083 to 0.197; the median is 0.0 in both runs. Without that take the two runs
  match: WER 0.100 vs 0.098, CER 0.083 vs 0.082. The tolerance (0.249) is wide
  because the same kind of runaway takes inflate the standard error, so this comparison could barely fail; read
  "AGREES" for Spanish WER and CER as "not distinguishable at this sample size", not as reproduction.

## 2. Files in this Directory

| File | Description |
|---|---|
| `reproducibility_report.json` | Master report recording parameters, timestamp, per-language comparisons, absolute differences, tolerances, and overall verdict (`all_metrics_agree: true`). |
| `compare_en.json` | Statistical comparison for English (`en`) across 6 metrics (WER, CER, SECS, SECS_ALL, UTMOS, Duration Ratio). |
| `compare_es.json` | Statistical comparison for Spanish (`es`) across 6 metrics. |
| `compare_fr.json` | Statistical comparison for French (`fr`) across 6 metrics. |
| `compare_de.json` | Statistical comparison for German (`de`) across 6 metrics. |
| `compare_hi.json` | Statistical comparison for Hindi (`hi`) across 6 metrics. |
| `repro_base_s12_en.json` | Fresh evaluation manifest for English ($N=180$). |
| `repro_base_s12_es.json` | Fresh evaluation manifest for Spanish ($N=180$). |
| `repro_base_s12_fr.json` | Fresh evaluation manifest for French ($N=180$). |
| `repro_base_s12_de.json` | Fresh evaluation manifest for German ($N=180$). |
| `repro_base_s12_hi.json` | Fresh evaluation manifest for Hindi ($N=180$). |

---

## 3. Results Summary (All 5 Languages)

| Language | Metric | Published Baseline (Run 1) | Reproducibility Re-run (Run 2) | Absolute Difference $|\Delta|$ | 95% CI Tolerance | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **English (`en`)** | WER | 0.0603 | 0.0617 | 0.0014 | 0.0260 | **AGREES** |
| | CER | 0.0203 | 0.0197 | 0.0006 | 0.0131 | **AGREES** |
| | SECS ($n=150$) | 0.5751 | 0.5744 | 0.0008 | 0.0237 | **AGREES** |
| | SECS_ALL ($N=180$) | 0.5367 | 0.5356 | 0.0011 | 0.0276 | **AGREES** |
| | UTMOS | 4.0361 | 4.0392 | 0.0032 | 0.0814 | **AGREES** |
| | Duration Ratio | 1.0605 | 1.0585 | 0.0020 | 0.0682 | **AGREES** |
| **Spanish (`es`)** | WER | 0.0990 | 0.2129 | 0.1139 | 0.2490 | **AGREES** |
| | CER | 0.0828 | 0.1973 | 0.1145 | 0.2359 | **AGREES** |
| | SECS ($N=180$) | 0.5112 | 0.5110 | 0.0002 | 0.0078 | **AGREES** |
| | SECS_ALL ($N=180$) | 0.5112 | 0.5110 | 0.0002 | 0.0078 | **AGREES** |
| | UTMOS | 3.2872 | 3.2886 | 0.0014 | 0.0786 | **AGREES** |
| | Duration Ratio | 1.0782 | 1.0868 | 0.0086 | 0.0940 | **AGREES** |
| **French (`fr`)** | WER | 0.0429 | 0.0437 | 0.0009 | 0.0168 | **AGREES** |
| | CER | 0.0314 | 0.0318 | 0.0003 | 0.0121 | **AGREES** |
| | SECS ($N=180$) | 0.4498 | 0.4496 | 0.0002 | 0.0101 | **AGREES** |
| | SECS_ALL ($N=180$) | 0.4498 | 0.4496 | 0.0002 | 0.0101 | **AGREES** |
| | UTMOS | 3.3023 | 3.3074 | 0.0051 | 0.0775 | **AGREES** |
| | Duration Ratio | 0.9386 | 0.9368 | 0.0017 | 0.0719 | **AGREES** |
| **German (`de`)** | WER | 0.0526 | 0.0526 | 0.0000 | 0.0247 | **AGREES** |
| | CER | 0.0316 | 0.0316 | 0.0001 | 0.0155 | **AGREES** |
| | SECS ($N=180$) | 0.5121 | 0.5115 | 0.0007 | 0.0124 | **AGREES** |
| | SECS_ALL ($N=180$) | 0.5121 | 0.5115 | 0.0007 | 0.0124 | **AGREES** |
| | UTMOS | 3.4387 | 3.4448 | 0.0060 | 0.0556 | **AGREES** |
| | Duration Ratio | 1.0621 | 1.0636 | 0.0015 | 0.0368 | **AGREES** |
| **Hindi (`hi`)** | WER | 0.1005 | 0.0992 | 0.0013 | 0.0157 | **AGREES** |
| | CER | 0.0470 | 0.0475 | 0.0005 | 0.0093 | **AGREES** |
| | SECS ($N=180$) | 0.5419 | 0.5405 | 0.0013 | 0.0152 | **AGREES** |
| | SECS_ALL ($N=180$) | 0.5419 | 0.5405 | 0.0013 | 0.0152 | **AGREES** |
| | UTMOS | 3.4541 | 3.4559 | 0.0017 | 0.0529 | **AGREES** |
| | Duration Ratio | 1.2908 | 1.2916 | 0.0008 | 0.0284 | **AGREES** |

---

## 4. Technical Notes

1. **Overall Verdict & Scope:** 30 out of 30 statistical tests pass ($100\%$ pooled statistical agreement within pre-registered $95\%$ confidence bounds). However, this pooled agreement does not rule out language-specific regressions: notably, Spanish (`es`) WER (0.0990 vs 0.2129) and CER (0.0828 vs 0.1973) more than doubled in the rerun, accompanied by six `too_long` duration failures, though remaining within the wide conservative confidence interval ($\pm 0.2490$ and $\pm 0.2359$).
2. **ASR Batch Calibration:** Whisper Large-v3 evaluation batch size was set to `batch_size=16` in `xtts_trainer/asr.py`. At `batch_size=32`, float32 attention states exceeded the 40 GB VRAM limit of the A100 GPU (38.64 GiB allocated). At `batch_size=16`, peak VRAM stayed below 18 GiB while delivering identical deterministic transcriptions.
3. **Cross-Revision Input Metadata & Provenance:** Because this run evaluates the current code revision with batched ASR (`batch_size=16` vs. published `batch_size=1`), metadata keys `code_fingerprint` and `batch_size` differ from the published baseline manifest (commit `1c33371`). Consequently, the strict same-input comparator flags `reproducible: false` due to input metadata differences. The results demonstrate pooled statistical agreement across the suite rather than identical same-input reproduction, and do not rule out language-specific regressions.
