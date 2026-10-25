---
language:
- en
- multilingual
license: mpl-2.0
tags:
- text-to-speech
- tts
- xtts
- xtts-v2
- voice-cloning
- pytorch
- speech-synthesis
metrics:
- wer
- cer
pipeline_tag: text-to-speech
library_name: coqui-tts
---

# Open-XTTS-Trainer

> **Reproducible 3-Component Training Suite for XTTS v2**  
> A modular framework bringing together training and fine-tuning workflows for all three XTTS v2 components: **Discrete VAE (DVAE)**, **Autoregressive GPT Backbone (with Vocabulary Extension for new scripts)**, and the **Latent HiFi-GAN Vocoder**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23573888.svg)](https://zenodo.org/records/23573888)
<a href="https://huggingface.co/amitbhakt/open-xtts-trainer" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amitbhakt%2Fopen--xtts--trainer-yellow" alt="Hugging Face"></a>
<a href="https://colab.research.google.com/github/amitbhakt/open-xtts-trainer/blob/main/notebooks/open_xtts_train_demo.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

---

### Quick Links
- 🚀 **Interactive Demo:** <a href="https://colab.research.google.com/github/amitbhakt/open-xtts-trainer/blob/main/notebooks/open_xtts_train_demo.ipynb" target="_blank" rel="noopener noreferrer">Open in Google Colab (Run End-to-End Verification)</a>
- 🤗 **Hugging Face Hub:** <a href="https://huggingface.co/amitbhakt/open-xtts-trainer" target="_blank" rel="noopener noreferrer">amitbhakt/open-xtts-trainer</a>

---

## Scope & Contribution

The XTTS v2 neural architecture consists of three distinct components:
1. **Stage 1 — Discrete VAE (DVAE):** Compresses raw mel-spectrograms into discrete audio tokens.
2. **Stage 2 — Autoregressive GPT Backbone:** Generates speech tokens conditioned on text and speaker latents (enhanced here with dynamic vocabulary extension for unsupported scripts).
3. **Stage 3 — Latent HiFi-GAN Vocoder:** Synthesizes audio waveforms conditioned on GPT acoustic latents and speaker embeddings.

### The Problem Addressed
When Coqui AI shut down, the community was left without official training recipes for the **Discrete VAE** and the **latent-conditioned HiFi-GAN vocoder**. Additionally, expanding XTTS v2 to non-Latin scripts (e.g. Arabic, Indic, Thai, Dhivehi) frequently caused embedding out-of-bounds crashes, while repository abandonment led to `torchcodec` dependency issues on modern Python builds.

**The contribution of this repository is unifying, repairing, and open-sourcing the separate component-training workflows into one accessible, reproducible, dependency-rot-free suite.**

## Status and measured results

Every number in this README comes from the evaluation harness (`eval/`, `scripts/run_eval_baseline.py`) and
links to the manifest that produced it. Training losses are not results: the earlier convergence table was
removed because no manifest backed it ([docs/recipe_design.md](https://github.com/amitbhakt/open-xtts-trainer/blob/main/docs/recipe_design.md) §6, phase 8).

**Measured: stock XTTS-v2 baseline**, full multilingual baseline across 5 languages, 12 sentences × 5 CMU ARCTIC voices × 3 seeds (900 utterances total; values: mean ± 95% CI half-width; speaker similarity uses ECAPA2 on clips ≥ 2 s, English n = 150; [index](https://github.com/amitbhakt/open-xtts-trainer/blob/main/eval/baselines/base_xtts_s12_index.json), [run notes](https://github.com/amitbhakt/open-xtts-trainer/blob/main/docs/evidence/run5/README.md)):

| Language | WER (Whisper large-v3) | CER | Speaker similarity, ECAPA2 (clips ≥ 2 s) | UTMOS | Duration ratio |
|---|---|---|---|---|---|
| **English (`en`)** | 0.060 ± 0.018 | 0.020 ± 0.009 | 0.575 ± 0.017 | 4.036 ± 0.057 | 1.061 ± 0.048 |
| **Spanish (`es`)** | 0.099 ± 0.075 | 0.083 ± 0.049 | 0.511 ± 0.006 | 3.287 ± 0.055 | 1.078 ± 0.065 |
| **French (`fr`)** | 0.043 ± 0.012 | 0.031 ± 0.009 | 0.450 ± 0.007 | 3.302 ± 0.055 | 0.939 ± 0.051 |
| **German (`de`)** | 0.053 ± 0.017 | 0.032 ± 0.011 | 0.512 ± 0.009 | 3.439 ± 0.039 | 1.062 ± 0.026 |
| **Hindi (`hi`)** | 0.101 ± 0.011 | 0.047 ± 0.007 | 0.542 ± 0.011 | 3.454 ± 0.037 | 1.291 ± 0.020 |

**Not yet measured: full language adaptation.** The three stages run end to end and are covered by tests. Full language adaptation across non-Latin scripts will be evaluated in subsequent runs.

**Measured: Target Speaker Voice Adaptation (Stage 2 GPT Fine-Tuning)**
We evaluate target speaker voice adaptation on Moira Fogarty reading *The Art of War* (8.4 minutes audio, 83 records in `dataset_train.jsonl` [75 train clips, 8 eval clips], 2 strictly held-out reference clips). Paired A/B evaluation against pre-registered quality gates ([eval/gates_target.json](https://github.com/amitbhakt/open-xtts-trainer/blob/main/eval/gates_target.json)) on held-out reference audio across $N=60$ ($20 \text{ sentences} \times 3 \text{ seeds}$):
- **Speaker Similarity (SECS):** $0.5545 \to 0.5875$ (**$+0.0329$ gain**, 95% CI $[+0.0191, +0.0480]$, $n=51$ qualifying pairs $\ge 2\text{ s}$, gate: `better`, **PASS**)
- **Word Error Rate (WER):** $0.0383 \to 0.0375$ ($-0.0008$, 95% CI $[-0.0096, +0.0079]$, gate: `not_worse 0.02`, **PASS**)
- **Character Error Rate (CER):** $0.0098 \to 0.0105$ ($+0.0008$, 95% CI $[-0.0045, +0.0060]$, gate: `not_worse 0.01`, **PASS**)
- **Audio Naturalness (UTMOS):** $4.2589 \to 4.2149$ ($-0.0439$, 95% CI $[-0.0831, -0.0065]$, gate: `not_worse 0.10`, **PASS**)
- **Failures:** 0 failures across all 60 evaluations (gate: `not_worse 0`, **PASS**)

All 5 pre-registered adaptation gates passed with statistically significant speaker similarity gain and no intelligibility regression beyond the gate tolerance (CER $+0.0008$, within the 0.01 tolerance). Details and run manifests are in [docs/evidence/run6_target_speaker/](https://github.com/amitbhakt/open-xtts-trainer/tree/main/docs/evidence/run6_target_speaker/).

**Licenses:**
- XTTS-v2 weights: Coqui Public Model License, non-commercial only.
- MMS_FA aligner and ECAPA2: CC BY-NC 4.0.
- CMU ARCTIC reference voices: their own license (downloaded and verified by [scripts/fetch_reference_voices.py](https://github.com/amitbhakt/open-xtts-trainer/blob/main/scripts/fetch_reference_voices.py); upstream source at [festvox.org/cmu_arctic](http://festvox.org/cmu_arctic/)).

> [!WARNING]
> **DVAE Codebook Retraining Invariant:**  
> Fine-tuning or retraining the Discrete VAE (Stage 1) alters discrete codebook token assignments, which **invalidates pretrained GPT backbone checkpoints**. Only train Stage 1 if you intend to train the Stage 2 GPT backbone from scratch or adapt the audio tokenizer to a fundamentally new sample rate / codec architecture. For standard voice cloning or adding new language scripts, keep the base DVAE frozen and train Stage 2 and Stage 3.

---

## Component Workflows

```
┌─────────────────────────────────┐
│     Stage 1: Discrete VAE       │──► Mel-Spectrogram Compression & Audio Tokenizer
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Stage 2: GPT Alignment      │──► Text & Vocab Extension ──► Predicts DVAE Codes
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Stage 3: Latent HiFi-GAN      │──► Converts GPT Latents + Speaker Embeddings ──► Waveform
└─────────────────────────────────┘
```

| Goal | Required Workflows |
| :--- | :--- |
| **Voice adaptation / Speaker cloning** | Run **Stage 2 (GPT)** + **Stage 3 (HiFi-GAN)** |
| **New script / language (e.g. Arabic, Hindi, Dhivehi)** | Run **Stage 2 (with Vocab Extender)** + **Stage 3 (HiFi-GAN)** |
| **Custom audio codec / Sample rate change (e.g. 48kHz)** | Run **Stage 1 (DVAE from scratch)** + **Stage 2** + **Stage 3** |

---

## Installation

Requires **FFmpeg** (system package). Coqui-TTS reads audio through `torchcodec`, which loads FFmpeg's shared
libraries: `sudo apt-get install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS. Colab already has it.

```bash
git clone https://github.com/amitbhakt/open-xtts-trainer.git
cd open-xtts-trainer
uv sync --frozen        # exactly the versions in uv.lock; or: pip install -e .
python scripts/fetch_base_artifacts.py --dest xtts_base   # XTTS-v2 files, hash-verified
```

---

## Quickstart

### 1. Preparing Speech Data

Transcripts are never guessed from fixed-length cuts. Public sources take their text from the corpus's own
metadata; recordings are transcribed (Whisper large-v3-turbo, or your transcript), forced-aligned with
MMS_FA, and cut only at pauses between words, so every clip's text is exactly the words spoken in it.
The aligner weights (1.26 GB, CC BY-NC 4.0) download on first use.

```bash
# Sample set: 6 LJSpeech clips (text from metadata.csv) + JFK, aligned and cut at pauses
python -m xtts_trainer.speech_data --source sample --output_dir sample_audio

# Continuous English speech, one reader (LibriVox, public domain, files SHA-256 pinned), Whisper-transcribed
# and aligned: art_of_war_short is 8.4 min, art_of_war is 72 min (--max_minutes caps each file)
python -m xtts_trainer.speech_data --source art_of_war_short --output_dir sample_audio

# Your own recording: language and speaker are required, never defaulted
python -m xtts_trainer.speech_data --input_file my_voice.wav --lang en --speaker_name me --output_dir my_data
```

Each run writes `prep_report.json` (segments, duration histogram, characters per second, alignment scores,
everything dropped and why) and `validation_report.json`, and fails unless the result passes validation.
Both write `sample_audio/dataset.jsonl`, the one dataset format every stage reads (schema: [xtts_trainer/data/schema.py](https://github.com/amitbhakt/open-xtts-trainer/blob/main/xtts_trainer/data/schema.py)).
To convert an existing pipe-separated metadata file, declare its columns explicitly:

```bash
python scripts/convert_metadata_csv.py my_data/metadata.csv --columns audio,text,speaker,lang --transcript_source human
```

Every stage validates the dataset before training and refuses to start on any error (missing or mismatched audio,
empty or `[UNK]` text, unregistered language, text/audio length mismatch, single-clip speakers, overlap with the
eval set). Run the same check yourself first:

```bash
python -m xtts_trainer.data.validate sample_audio/dataset.jsonl --vocab xtts_base/vocab.json
```

### 2. Stage 2: Vocabulary Extender & GPT Backbone Fine-Tuning

1. **Extend Tokenizer for a New Language Script:**
   ```bash
   python xtts_trainer/stage2_gpt/vocab_extender.py \
       --target_text /path/to/target_corpus.txt \
       --base_vocab xtts_base/vocab.json \
       --output_vocab xtts_base/vocab_extended.json \
       --languages bn el
   ```

2. **Fine-tune the Autoregressive GPT Backbone** (a thin wrapper over upstream `GPTTrainer`):
   ```bash
   python -m xtts_trainer.stage2_gpt.train \
       --dataset my_data/dataset.jsonl \
       --base_dir xtts_base \
       --vocab_path xtts_base/vocab_extended.json --languages bn \
       --replay en_data/dataset.jsonl \
       --output_dir checkpoints/gpt \
       --epochs 10 --batch_size 4 --grad_accum_steps 8
   ```
   Upstream runs the loop; the wrapper fixes what upstream gets wrong:
   - every sample's own voice prompt is excluded from the loss, not just the last sample's (F2);
   - new languages get a registered tokenizer in training *and* inference (F4);
   - base weights and the DVAE load strictly, and new text rows start at the mean of the existing rows (F3, F6);
   - a clip that fails to load stops the run instead of being silently replaced.

   The best checkpoint by eval loss is exported to `checkpoints/gpt/export/`, a complete checkpoint directory
   (`model.pth`, `config.json`, `vocab.json`, `dvae.pth`). Load it with
   `xtts_trainer.stage2_gpt.export.load_for_inference`. Replay data must be in base languages; upstream
   samples languages uniformly, so with one replay language half of the training draws are replay.

### 3. Stage 3: Latent HiFi-GAN Vocoder

1. **Extract True GPT Latents from Paired Audio and Ground-Truth Transcripts:**
   ```bash
   python xtts_trainer/stage3_hifigan/extract_latents.py \
       --dataset sample_audio/dataset.jsonl \
       --checkpoint_dir xtts_base \
       --output_dir extracted_latents \
       --device cuda
   ```

2. **Fine-Tune the HiFi-GAN Vocoder** (optional; keep it only if it beats the stock decoder in the Phase 8 A/B):
   ```bash
   python -m xtts_trainer.stage3_hifigan.train \
       --dataset sample_audio/dataset.jsonl \
       --latents_dir extracted_latents \
       --checkpoint_dir xtts_base \
       --output_dir checkpoints/hifigan \
       --batch_size 8 --epochs 25 --disc_warmup_steps 1000
   ```
   - **Validation:** held-out clips are scored as whole-clip resynthesis (log-mel L1). The starting decoder is
     scored first, and that number is the one to beat.
   - **Best checkpoint:** `checkpoints/hifigan/hifigan.pth` keeps the best by validation error and records
     whether it beats the starting decoder. Audio samples go to `samples/step_N/`.
   - **Alignment:** decoder input is interpolated over each whole clip before segments are cut, so audio and
     latents line up exactly.
   - **Warm-up:** a run that would never leave the discriminator warm-up is refused (or pass `--mel_only`).
   - **Options:** `--scl_weight` (Speaker Consistency Loss) and `--full_band_mel`.
   - **Resume** with `--resume`. Mixed precision is on for CUDA.

3. **Synthesize Waveforms with Trained Vocoder:**
   ```bash
   python -m xtts_trainer.stage3_hifigan.inference \
       --checkpoint checkpoints/hifigan/hifigan.pth \
       --latent extracted_latents/speech_001.pt \
       --output synthesized_voice.wav
   ```

### 4. Stage 1: Discrete VAE (DVAE) Training

Off by default: every training step moves the codebook ([docs/recipe_design.md](https://github.com/amitbhakt/open-xtts-trainer/blob/main/docs/recipe_design.md) F1), so a trained DVAE no longer
matches the pretrained GPT.

```bash
python -m xtts_trainer.stage1_dvae.train \
    --dataset sample_audio/dataset.jsonl \
    --output_dir checkpoints/dvae \
    --base_checkpoint xtts_base/dvae.pth \
    --batch_size 8 --epochs 25 [--frozen_codebook]
```

- **Logged on held-out clips:** validation loss, active codes and perplexity, and *code drift* (the share of
  frames whose code differs from the base DVAE's).
- **Collapse gate:** if the codebook collapses, the run fails.
- **Best checkpoint:** `dvae.pth` keeps the best by validation loss and is marked `requires_gpt_retrain`.
- **Stage 2 pairing:** Stage 2 refuses to pair a marked DVAE with the pretrained GPT unless given
  `--retrain_for_new_dvae`, and Stage 3 latent extraction refuses a GPT that wasn't trained on it.
- **`--frozen_codebook`** keeps the quantizer fixed. The encoder still adapts, so codes still drift; the
  drift metric shows how much.

---

## Measuring a fine-tune

Run the harness on the stock model and on yours, then compare them with paired statistics (same sentences,
voices and seeds) against pre-registered gates:

```bash
python scripts/run_eval_baseline.py --checkpoint_dir xtts_base --label base --output eval/baselines/base.json
python scripts/run_eval_baseline.py --checkpoint_dir checkpoints/gpt/export --label tuned --output eval/baselines/tuned.json
python scripts/compare_ab.py eval/baselines/base.json eval/baselines/tuned.json --gates eval/gates.json
```

This uses the eval set's CMU ARCTIC voices, speakers the fine-tune never saw: it checks that nothing regressed.
To check that a voice fine-tune actually learned **your** speaker, hold out some of their clips first. Training
on them is then refused while the lock exists:

```bash
python scripts/build_target_voices.py --dataset my_data/dataset.jsonl   # writes my_data/dataset_train.jsonl
# train on my_data/dataset_train.jsonl, then measure both models on the held-out voice:
python scripts/run_eval_baseline.py --checkpoint_dir xtts_base --voices target --label base --output eval/baselines/base_target.json
python scripts/run_eval_baseline.py --checkpoint_dir checkpoints/gpt/export --voices target --label tuned --output eval/baselines/tuned_target.json
```

Then copy the `gate` object inside `target_speaker_gate_template` in [eval/gates.json](https://github.com/amitbhakt/open-xtts-trainer/blob/main/eval/gates.json) into the `gates` list, with
`voices` set to the `target_<speaker>` ids the first command printed, and run `compare_ab.py` on the two target
manifests.

The gates are relative to the stock model: for example, word error rate may not get worse beyond a set margin,
and speaker similarity on the target speaker must improve. Fix them in [eval/gates.json](https://github.com/amitbhakt/open-xtts-trainer/blob/main/eval/gates.json) before you look at
results.

Earlier proof samples on Hugging Face (`proof_outputs/`) were produced by a retired script that checked absolute
thresholds without a baseline. They are not harness measurements.

---

## Citation

If you use this training suite in your academic work or project, please cite:

```bibtex
@software{bhakt2026openxtts,
  author = {Bhakt, Amit},
  title = {open-xtts-trainer: Reproducible 3-Component Training Suite for XTTS v2},
  year = {2026},
  doi = {10.5281/zenodo.23573888},
  url = {https://zenodo.org/records/23573888}
}
```

---

## License

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.
