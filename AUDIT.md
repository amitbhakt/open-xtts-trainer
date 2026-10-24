# Open-XTTS-Trainer — ML Audit

**Verdict:** Every stage runs and its loss goes down, but the stages train on wrong inputs or targets. The proof suite only checks that the pipeline runs, not that the speech is good. Your own `proof_outputs/pipeline_proof_manifest.json` shows the DVAE using 3 of 1024 codes, yet the gate is marked PASS.

## Critical

| # | Issue | Where | Why it matters |
|---|---|---|---|
| 1 | DVAE spectrogram uses frame size 4096 / hop 1024; upstream uses 1024 / 256 | `stage1_dvae/dataset.py:18`, `stage2_gpt/train.py:263`, `stage3_hifigan/extract_latents.py:82` | Tokens come out at ~5.4/s instead of 21.5/s. The pretrained DVAE sees input it never trained on, the GPT learns from mismatched tokens, and Stage 3 latents cover only ¼ of each clip. |
| 2 | Missing transcripts fall back to a fixed fake sentence | `stage2_gpt/train.py:88-93` | The README command has no `--metadata_path` and `speech_data.py` writes no transcripts. The GPT learns to ignore text. |
| 3 | Reference-voice spectrogram is 4096/1024, unnormalized; inference uses 2048/256 with `mel_stats` | `stage2_gpt/train.py:274` vs upstream `xtts.py:219` | Speaker conditioning differs between training and inference. |
| 4 | Reference audio is the target clip itself (0–6 s, not 3–6 s as the comment says); no masking; `cond_lens` not passed | `stage2_gpt/train.py:126-129` | Leaks the target into the input. Zero padding (loud in log-mel) gets attended to. |
| 5 | Latent extraction silently falls back to `torch.randn` latents and speaker embeddings | `stage3_hifigan/extract_latents.py:93-96, 146-151` | The README command has no `--checkpoint_dir`, so the vocoder trains on noise. |
| 6 | Latents are teacher-forced against one fixed sentence for every clip | `stage3_hifigan/extract_latents.py:25` | Training latents don't match the distribution the vocoder gets at inference. |
| 7 | Latent-to-audio slicing assumes 1024 samples per latent, and the end clamp hides misalignment | `stage3_hifigan/dataset.py:29, 100-103` | Combined with #1, latents and audio are misaligned 4×. |
| 8 | README Stage 3 command trains HiFi-GAN from scratch on 8–100 clips | README, `stage3_hifigan/train.py:81` | Final mel loss 63.5 (≈1.4 log-mel L1) is far from a usable vocoder. |
| 9 | Verify script reads key `"model"` but Stage 1 saves under `"dvae"`; loads with `strict=False` | `scripts/verify_pipeline_proof.py:50-53` | Most likely no weights load, so the gate checks a random DVAE (explains 3/1024 codes). |

## Major

| # | Issue | Where | Why it matters |
|---|---|---|---|
| 10 | `strict=False` used everywhere | stage 1 load, stage 2 DVAE load, `inference.py`, `extract_latents.py`, verify script | Failed weight loads go unnoticed. The verify script silently drops extended-vocab embeddings, so it runs the base model. |
| 11 | Audio tokens cut at 600 but text isn't | `stage2_gpt/train.py:123` | Teaches the model to stop mid-sentence. Filter long clips instead. |
| 12 | Unsupported language falls back to raw encoding with no `[lang]` token; no new language token added | `stage2_gpt/train.py:132-136`, `vocab_extender.py` | New-script training has no language token. |
| 13 | No `config.json` with updated `gpt_number_text_tokens` is saved | `stage2_gpt/train.py:491-526` | Extended-vocab `model.pth` won't load in stock XTTS. |
| 14 | Fine-tuning the DVAE invalidates the pretrained GPT | Stage 1 → 2 flow | Token meanings change, but nothing enforces retraining the GPT. |
| 15 | Fresh random discriminators vs. a pretrained generator, no discriminator warm-up | `stage3_hifigan/train.py:76-77` | Early GAN gradients can damage the pretrained vocoder. |
| 16 | Stage 2 duplicates upstream `GPTTrainer`, which already supports vocab extension | idiap `gpt_trainer.py:115-145` | Partly invalidates the "no recipe exists" premise; upstream is more correct. |

## Evaluation

| # | Issue | Why it matters |
|---|---|---|
| 17 | Gates only check finite values, RMS ≥ 0.02 and peak ≤ 1.05 | These check that audio exists, not that it's good. |
| 18 | >4 kHz energy ratio is measured but never checked (recorded values 0.07–0.3%) | A sentence full of "s" sounds should score far higher; the output is likely muffled. |
| 19 | No word error rate (`jiwer` unused), speaker similarity, MOS predictor, held-out set or A/B against base XTTS | Can't tell the trained model apart from base XTTS or a broken one. |

## Minor

- README overclaims: the "Gumbel temperature" row doesn't come from this code (the DVAE uses `Quantize`), loss falling on 8 clips is overfitting not convergence, and "zero fallbacks" is false.
- `stage1_dvae/losses.py` is never used.
- Stage 1 pads spectrograms with 0, which is loud in log-mel, not silence.
- `speech_data.py` cuts audio into fixed 3 s chunks, often mid-word, with no transcripts.
- No validation split, seed, GPT LR warmup or schedule, mixed precision or gradient accumulation; stages 2 and 3 can't resume.
- DVAE runs on the GPU inside `__getitem__`, which breaks with `num_workers > 0`.
- Tests use only random tensors.

## Fix order

1. Read every spectrogram setting from the XTTS config instead of hardcoding it (#1, #3).
2. Make missing transcripts, a missing checkpoint or a key mismatch a hard error, with `strict=True` by default (#2, #5, #9, #10).
3. Use upstream `GPTTrainer` for Stage 2 and focus this repo on the DVAE and HiFi-GAN recipes.
4. Add real evaluation: Whisper word error rate, speaker similarity and UTMOS on a held-out set, compared against base XTTS.
5. Tone down the README and DOI claims until those numbers exist.
