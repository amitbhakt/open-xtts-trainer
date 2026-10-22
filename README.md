# open-xtts-trainer

> **Complete 3-Stage Training Suite for XTTS v2**  
> Train or fine-tune all 3 components of XTTS v2: **Discrete VAE (DVAE)**, **GPT Backbone**, and the **Latent HiFi-GAN Vocoder**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23573888.svg)](https://zenodo.org/records/23573888)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amitbhakt/open-xtts-trainer/blob/main/notebooks/open_xtts_train_demo.ipynb)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

---

## Overview

While [XTTS v2](https://huggingface.co/coqui/XTTS-v2) is one of the most widely used zero-shot multilingual text-to-speech models, the original release only provided training recipes for fine-tuning the autoregressive GPT backbone. The community was left without training recipes for the **Discrete VAE** and the **HiFi-GAN vocoder**.

This repository fills that gap by providing a clean, modular, end-to-end training pipeline for all 3 stages:

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

---

## When to use which stage?

| Goal | Required Stages |
| :--- | :--- |
| **New speaker / Voice adaptation (Supported language)** | Run **Stage 2 (GPT)** + **Stage 3 (HiFi-GAN)** |
| **New script / language (e.g. Arabic, Hindi, Dhivehi)** | Run **Stage 2 (with Vocab Extender)** + **Stage 3 (HiFi-GAN)** |
| **Custom audio codec / Sample rate change (e.g. 48kHz)** | Run **Stage 1 (DVAE from scratch)** + **Stage 2** + **Stage 3** |

---

## Installation

```bash
git clone https://github.com/amitbhakt/open-xtts-trainer.git
cd open-xtts-trainer
pip install -e .
```

---

## Quickstart

### Stage 3: Latent HiFi-GAN Vocoder

1. **Extract GPT Latents from your audio dataset:**
   ```bash
   python xtts_trainer/stage3_hifigan/extract_latents.py \
       --dataset_dir /path/to/wavs \
       --metadata_path /path/to/metadata.csv \
       --output_dir /path/to/extracted_latents \
       --device cuda
   ```

2. **Train the HiFi-GAN Vocoder:**
   ```bash
   python xtts_trainer/stage3_hifigan/train.py \
       --latents_dir /path/to/extracted_latents \
       --audio_dir /path/to/wavs \
       --output_dir ./checkpoints/hifigan \
       --batch_size 16 \
       --epochs 50
   ```

### Stage 2: Vocabulary Extender & GPT Fine-Tuning

1. **Extend Tokenizer for a New Language Script:**
   ```bash
   python xtts_trainer/stage2_gpt/vocab_extender.py \
       --target_text /path/to/corpus.txt \
       --base_vocab ./xtts_models/vocab.json \
       --output_vocab ./xtts_models/vocab_extended.json
   ```

2. **Fine-tune the GPT stage:**
   ```bash
   python xtts_trainer/stage2_gpt/train.py \
       --config_path ./configs/gpt_config.json
   ```

### Stage 1: Discrete VAE (DVAE) Training

```bash
python xtts_trainer/stage1_dvae/train.py \
    --audio_dir /path/to/wavs \
    --output_dir ./checkpoints/dvae \
    --batch_size 32 \
    --epochs 100
```

---

## Citation

If you use this training suite in your academic work or project, please cite:

```bibtex
@software{bhakt2026openxtts,
  author = {Bhakt, Amit},
  title = {open-xtts-trainer: Complete 3-Stage Training Suite for XTTS v2},
  year = {2026},
  doi = {10.5281/zenodo.23573888},
  url = {https://zenodo.org/records/23573888}
}
```

---

## License

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.
