# Run 5: Full Multilingual Baseline and Stage 2/3 Verification (2026-10-25)

Conducted on Modal serverless compute with an NVIDIA A100-SXM4-40GB GPU. All baseline manifests and master index are committed to `eval/baselines/`.

| File | What it holds |
|---|---|
| `base_xtts_s12_index.json` | Master index containing cryptographic SHA-256 digests and summaries across all 5 languages |
| `base_xtts_s12_en.json` | English baseline manifest (180 utterances: 12 sentences × 5 ARCTIC voices × 3 seeds) |
| `base_xtts_s12_es.json` | Spanish baseline manifest (180 utterances: 12 sentences × 5 ARCTIC voices × 3 seeds) |
| `base_xtts_s12_fr.json` | French baseline manifest (180 utterances: 12 sentences × 5 ARCTIC voices × 3 seeds) |
| `base_xtts_s12_de.json` | German baseline manifest (180 utterances: 12 sentences × 5 ARCTIC voices × 3 seeds) |
| `base_xtts_s12_hi.json` | Hindi baseline manifest (180 utterances: 12 sentences × 5 ARCTIC voices × 3 seeds) |

---

## 1. Resolution of Finding P1 (Checkpoint Provenance)

Previous Colab runs 1–2 suffered from an in-place checkpoint mutation bug in upstream `coqui-ai-TTS`'s `load_checkpoint` (which caught `UnpicklingError` and silently re-saved `model.pth`, mutating its SHA-256 from `c7ea2000…` to `3077a05b…`). 

Run 5 resolved this finding (P1):
- Implemented `patch_xtts_prevent_checkpoint_mutation()` in `xtts_trainer/__init__.py` and called it prior to model initialization in `run_eval_baseline.py`.
- Re-verified all weights against `artifacts.lock.json` after every stage.
- `base_xtts_s12_index.json` strictly records `model.pth` as `c7ea20001c6a0a841c77e252d8409f6a74fb423e79b3206a0771ba5989776187`.

---

## 2. Phase 1: Multilingual Baseline Results (900 Utterances Total)

Evaluated across 5 CMU ARCTIC reference speakers (`arctic_slt`, `arctic_clb`, `arctic_bdl`, `arctic_awb`, `arctic_ksp`), 12 sentences per language, and 3 random seeds (`42, 123, 999`) under Metrics Version 2:

| Language | WER (Whisper large-v3) | CER | SECS (ECAPA2, ≥ 2 s) | UTMOS | Duration Ratio | Utterances / Failures |
|---|---|---|---|---|---|---|
| **English (`en`)** | **0.021 ± 0.007** | **0.013 ± 0.004** | **0.575 ± 0.017** | **4.036 ± 0.057** | **1.060 ± 0.048** | 180 / 0 |
| **Spanish (`es`)** | **0.099 ± 0.075** | **0.082 ± 0.049** | **0.511 ± 0.006** | **3.287 ± 0.055** | **1.078 ± 0.065** | 180 / 0 |
| **French (`fr`)** | **0.042 ± 0.012** | **0.031 ± 0.009** | **0.450 ± 0.007** | **3.302 ± 0.055** | **0.938 ± 0.051** | 180 / 0 |
| **German (`de`)** | **0.052 ± 0.017** | **0.031 ± 0.011** | **0.512 ± 0.009** | **3.438 ± 0.039** | **1.062 ± 0.026** | 180 / 0 |
| **Hindi (`hi`)** | **0.100 ± 0.011** | **0.047 ± 0.007** | **0.542 ± 0.011** | **3.454 ± 0.037** | **1.290 ± 0.020** | 180 / 0 |

Zero silent dropouts (`is_empty = 0`), zero runaway audio loops (`is_too_long = 0`), and zero `NaN` values across all 900 evaluation clips.

---

## 3. Phase 7: Training Smoke Verification

1. **Stage 2 (GPT Backbone):**
   - 3 epochs completed on 8-clip sample set with per-sample prompt masking (Fix F2) and registered tokenizer (Fix F4).
   - Eval loss dropped from **3.1999 → 2.9746**.
   - Checkpoint exported cleanly to `runs/gpt_smoke/export/`.
2. **Stage 3 (HiFi-GAN Vocoder):**
   - Latents extracted in teacher-forced eval mode with true transcript alignment.
   - 30 steps trained with discriminator warm-up.
   - Validation Mel L1 improved from **1.5452 (starting decoder) → 1.4547 (step 30)**.
