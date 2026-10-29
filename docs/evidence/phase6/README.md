# Phase 6 Multi-Speaker Alignment Quality Threshold Calibration

Implements and empirically grounds **`docs/recipe_design.md` §2 (Finding F5)** and **§6 (Phase 6 Data Preparation)** as an in-sample threshold calibration (`[E5]`). Held-out validation on unseen voices and languages has since run under a pre-registered protocol (`../phase6_heldout/README.md`): clean acceptance is validated out-of-sample for English (39/40 utterances, 59/60 segments); French and German had too few clips to validate; detection is reliable only for large corruptions.

---

## 1. Overview & Calibration Summary

- **Execution Date**: 2026-10-26 18:50 UTC
- **Compute Hardware**: NVIDIA A100-SXM4-40GB GPU (Modal app `ap-aLtghQDigN7bOJJjqUVrSn`)
- **Software Environment**: PyTorch 2.1.2 + CUDA 12.1 (exploratory run; locked environment `torch==2.14.1` required for held-out validation)
- **Scope**: Multi-speaker calibration across **8 baseline voices** and 2 languages (English, Castilian Spanish) across 5 accent profiles.
- **Evaluation Scale**: 8 clean baseline recordings + 88 controlled corruption tests (11 mutations per speaker) = **96 total alignment tests**.
- **Clean Baseline Acceptance**: **8/8 (100.0%)** clean recordings passed. Calibrated `max_weak_words = 3` to accommodate non-native accented speech (`arctic_ksp`, 3 weak words scoring down to $0.062$), while the studio-clean native clips (four ARCTIC voices, LJSpeech) and the Spanish audiobook have a minimum per-word alignment score $\ge 0.54$; JFK's 1961 recording has one weak word (minimum word score 0.122). The run itself used `max_unexplained_speech_s = 0.20` (`quality_config` in the JSON); the operating 0.10 s limit was applied afterwards by recomputing from the recorded unexplained speech (all clean clips measure $0.00\text{ s}$, so their verdicts are the same at either limit).
- **Controlled Corruption Characterization**: Characterizes empirical detection boundaries across cut severity, boundary position, unsupported transcript tokens, and trailing speech. At the 0.10 s limit (by recomputation), Spanish trailing spillover ($0.14\text{s}$) is caught. Held-out results: `../phase6_heldout/README.md`.

---

## 2. Directory Contents & Cryptographic Verification

| File | SHA-256 Checksum | Description |
|---|---|---|
| [`multilingual_alignment_validation.json`](multilingual_alignment_validation.json) | `7ebcd27c9fbd7eccaff0033d929009f5e91021984d24f2d57f69d5c13f45ac49` | Complete JSON report with metadata, per-speaker metrics, and test-by-test mutation outcomes. |
| [`ALIGNMENT_VALIDATION_METHODOLOGY.md`](ALIGNMENT_VALIDATION_METHODOLOGY.md) | `c7b65f059a6a08eac71d96c89b98f7ac8cb5179d3b26328a3e53feb42f481e7f` | Detailed methodology, threshold arithmetic, comparison operators, and failure-mode diagnostic analysis. |

---

## 3. Evaluated Speakers & Source Audio Provenance

All source recordings are cryptographically pinned in [`scripts/validate_alignment_multilingual.py`](../../scripts/validate_alignment_multilingual.py) and [`docs/evidence/source_pins.md`](../source_pins.md):

| Speaker ID | Language & Accent | Source / Origin | SHA-256 | License | Clean Status |
|---|---|---|---|---|:---:|
| `jfk_english_male` | English (US male, 1960s vintage) | JFK Inaugural Address excerpt | `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715` | Public Domain | **PASS** |
| `arctic_slt_english_female` | English (US female, studio clean) | CMU ARCTIC SLT (`eval/reference_voices/`) | `36895e2c7df0e354c059319fe32c612635b552a25ffc2692d56a8d9ee1d74e82` | Permissive (CMU) | **PASS** |
| `arctic_ksp_english_indian_accent` | English (Indian accented male) | CMU ARCTIC KSP (`eval/reference_voices/`) | `e374c94b05fe476b8563f0ac3086d4c601ead7d80cd42e5512a0d1c7ac424fd8` | Permissive (CMU) | **PASS** |
| `arctic_bdl_english_male` | English (US male, studio clean) | CMU ARCTIC BDL (`eval/reference_voices/`) | `886f17023bc3f0880043252b3f1471ccc35f286500512d939b7f1a91a6d8e627` | Permissive (CMU) | **PASS** |
| `arctic_clb_english_female` | English (US female, studio clean) | CMU ARCTIC CLB (`eval/reference_voices/`) | `3764871a513fd11309c58b36b27d291d428502daf0495159ca153c4e5e162e5b` | Permissive (CMU) | **PASS** |
| `arctic_awb_english_scottish` | English (Scottish accented male) | CMU ARCTIC AWB (`eval/reference_voices/`) | `63a3e6103b00fcd168df79a4d6ad6fda0e8b1b64eda6d9ce32eddd078c9c9b94` | Permissive (CMU) | **PASS** |
| `ljspeech_female` | English (US female, gold standard) | LJSpeech LJ001-0001 (`LJSpeech-1.1`) | `4d2bd83ff0f7fe33491b03193abdf623f1b4cc2d8103972b33638d3607d86dc5` | Public Domain | **PASS** |
| `spanish_cuentos_male` | Castilian Spanish (male, audiobook) | LibriVox Cuentos de Hadas intro | `28d18af59b6dcfad0d9f8121a948ecc377723d40639af7f0e64eba4fc3d7e30e` | Public Domain | **PASS** |

---

## 4. Summary of Detection Performance

### Observed Rejection Rates by Mutation Type & Severity
- **Edge Truncation**:
  - 100 ms tail cut: 3/8 rejected (37.5%)
  - 250 ms tail cut: 5/8 rejected (62.5%)
  - 500 ms tail cut: 2/8 rejected (25.0%)
  - 1,000 ms tail cut: 7/8 rejected (87.5%)
  - 500 ms head cut: 7/8 rejected (87.5%)
  - 1,000 ms head cut: 8/8 rejected (100.0%)
- **Transcript Word Insertions**: 17/24 rejected (70.8%).
- **Speech Spillover**:
  - Sub-threshold ($+0.10\text{s}$ appended): 8/8 passed (sub-threshold tolerance working as designed).
  - Cross-threshold ($+0.50\text{s}$ appended): 8/8 passed (evaded rejection because inter-utterance pause meant active speech portion measured $80 \text{--} 140\text{ ms}$, below the $200\text{ ms}$ threshold).
  - These counts were measured with `max_unexplained_speech_s = 0.20`, the setting at run time (`multilingual_alignment_validation.json`, `quality_config`). At the operating 0.10 s limit, recomputed from the recorded `unexplained_speech_s` (no rerun; only this rule depends on the limit, and no other mutation measured between 0.10 and 0.20 s): sub-threshold 8/8 passed, cross-threshold 7/8 passed (Spanish, 0.14 s, rejected), so 58/88 corruptions are classified correctly instead of 57/88. The sub-threshold case is not on the 0.10 s boundary: the 0.10 s appended fell inside the inter-utterance pause, and every speaker measured 0.00 s of unexplained speech.
