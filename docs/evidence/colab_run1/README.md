# Colab run 1: Phase 0 + Phase 1 pilot (2026-10-24, Tesla T4)

Produced by `notebooks/phase0_phase1_colab.ipynb` from an uploaded working-tree zip, so the manifests have no
git commit (`commit_sha: unknown`). The code was the local, uncommitted state as of Phase 2.

## What this run proves (still valid)

| File | Proves |
|---|---|
| `artifact_verification.json` | All 6 upstream artifacts match `artifacts.lock.json` (hash and size), and `mel_stats` is identical across the packaged file, the Hugging Face file and the tensor inside `model.pth` (max difference 0.0). Also copied to `docs/evidence/artifact_verification.json`. |
| `environment_colab.json` | Python 3.13.16, torch 2.11.0+cu130, torchaudio 2.11.0, transformers 5.20.2, CUDA 13.0, Tesla T4; `coqui-tts` at the pinned commit `03339e5b`. Full `pip freeze` included. |
| `pilot_en_reproducibility.json` | Two identical runs agree exactly (difference 0.0 on every metric). Different seeds do change the audio (47/50 durations differ between seeds 42 and 123), so the pipeline is deterministic per seed. |

## What is superseded

`smoke.json`, `pilot_en_run1.json`, `pilot_en_run2.json` and `reference_voices.lock.json` were measured on eval
set v1.1.0 with these known limitations, fixed afterwards:
- `--subset_size 10` took the first 10 English sentences, all in the easy "harvard" category, so WER sat at its
  floor (0.005; all 4 errors were "March" heard as "marched"). Subsets are now stratified across categories.
- The eval set had no short (1-3 word) utterances, so short-text hallucination could not be measured.
- The reference lock records no license file (the CMU ARCTIC "packed" tarballs omit `COPYING`) and no transcripts.

Pilot numbers on that (easy, English-only) subset, for reference only:
WER 0.005 ± 0.005, CER 0.002 ± 0.002, SECS (ECAPA2) 0.548 ± 0.018, UTMOS 4.12 ± 0.05, duration ratio 1.04,
0 failures; per-voice SECS from 0.43 (slt) to 0.62 (awb). About 1.45 s synthesis per utterance on a T4.
