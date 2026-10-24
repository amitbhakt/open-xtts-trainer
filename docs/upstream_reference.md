# Upstream Reference Specifications

This document defines the exact upstream reference implementations from `coqui-ai-TTS` that Open-XTTS-Trainer must match for mathematical, structural, and numerical parity.

- **Pinned Repository:** `https://github.com/idiap/coqui-ai-TTS.git`
- **Pinned Commit:** `03339e5b790b5e16ea8867a142e8bb26ec646d94`
- **Target Architecture:** Coqui XTTS-v2

Every statement below was checked against the source installed from the pinned commit (`.venv/lib/python3.11/site-packages/TTS/`). Paths are relative to that `TTS/` package. If the pin changes, re-check every line number and value.

---

## 0. Shared log-mel recipe (applies to every spectrogram below)

Both `wav_to_mel_cloning` (`tts/models/xtts.py:60-79`) and `TorchMelSpectrogram.forward` (`tts/layers/tortoise/arch_utils.py:235-248`) compute:

1. `torchaudio.transforms.MelSpectrogram(..., power=2, normalized=False, norm="slaney")` (power spectrogram, Slaney mel normalization).
2. `mel = log(clamp(mel, min=1e-5))` (natural log; digital silence maps to `log(1e-5) ≈ -11.5129`).
3. `mel = mel / mel_norms[None, :, None]`: an **element-wise division** by an 80-element per-channel tensor. There is **no mean subtraction** and no standard deviation.

`mel_norms` is the tensor in `mel_stats.pth`. At inference, `Xtts` uses `self.mel_stats` (loaded with the checkpoint). In training, `GPTTrainer` uses `GPTArgs.mel_norm_file` (default `https://github.com/coqui-ai/TTS/releases/download/v0.14.0_models/mel_norms.pth`, `tts/layers/xtts/trainer/gpt_trainer.py:41`).

`TorchMelSpectrogram` skips step 3 only when `mel_norm_file=None` is passed explicitly.

**Verified equivalence (2026-10-24):**
- The packaged `xtts_trainer/mel_stats.pth` is byte-identical to `coqui/XTTS-v2/mel_stats.pth` on Hugging Face (SHA-256 matches `artifacts.lock.json`); see `docs/evidence/artifact_verification.json`.
- `https://github.com/coqui-ai/TTS/releases/download/v0.14.1_models/mel_norms.pth` (the `TorchMelSpectrogram` default) has **identical values** (`torch.equal`) to the XTTS-v2 `mel_stats.pth`.
- The `GPTTrainer` default (`.../v0.14.0_models/mel_norms.pth`) currently returns **HTTP 404**. Upstream training with default `GPTArgs` would fail to load norms; always pass the verified local file.
- Still pending: comparison with the `mel_stats` tensor stored inside `model.pth` (run `scripts/fetch_base_artifacts.py --check_mel_stats` once `model.pth` is downloaded).

---

## 1. Conditioning Latent & Spectrogram Extraction

### `wav_to_mel_cloning`
- **Location:** `tts/models/xtts.py:35-79`
- **Parameters used by XTTS-v2 inference** (perceiver path, `tts/models/xtts.py:219-230`):
  - Sample rate: `22,050 Hz`
  - FFT size (`n_fft`): `2048`
  - Hop length (`hop_length`): `256`
  - Window length (`win_length`): `1024`
  - Mel channels (`n_mels`): `80`
  - Mel fmin / fmax: `0` / `8000`
  - **Normalization: log-mel divided by `mel_stats`** (`mel_norms=self.mel_stats`); see section 0.
- The non-perceiver path (`tts/models/xtts.py:240-252`) uses `n_fft=4096`, `hop_length=1024`, `win_length=4096` with the same normalization. XTTS-v2 uses the perceiver path (`gpt_use_perceiver_resampler=True`).
- **Training equivalent:** `GPTTrainer.torch_mel_spectrogram_style_encoder` (`tts/layers/xtts/trainer/gpt_trainer.py:149-160`) uses `TorchMelSpectrogram(filter_length=2048, hop_length=256, win_length=1024, normalize=False, sampling_rate=22050, mel_fmin=0, mel_fmax=8000, n_mel_channels=80, mel_norm_file=self.args.mel_norm_file)`.

### `Xtts.get_gpt_cond_latents`
- **Location:** `tts/models/xtts.py:195-255`
- **Pipeline (perceiver path):**
  1. Resample to `22,050 Hz` if needed; truncate to `length` seconds (default 30).
  2. Split into `chunk_length`-second chunks (default 6 s); skip chunks shorter than 0.33 s.
  3. Each chunk → `wav_to_mel_cloning` (as above) → `gpt.get_style_emb(mel)`, i.e. `conditioning_encoder` followed by `conditioning_perceiver` (`tts/layers/xtts/gpt.py:272-287`) → `[1, 1024, 32]`.
  4. Average the chunk embeddings; return `cond_latent.transpose(1, 2)` → `[1, 32, 1024]`.

### `Xtts.get_speaker_embedding`
- **Location:** `tts/models/xtts.py:257-264`
- Audio is **resampled to 16,000 Hz**, then passed through `hifigan_decoder.speaker_encoder.forward(audio_16k, l2_norm=True)` → `.unsqueeze(-1)` → `[1, 512, 1]`.

### `Xtts.get_conditioning_latents`
- **Location:** `tts/models/xtts.py:278-332`
- Loads each reference file at `load_sr=22050` (optionally trimmed with `librosa_trim_db` and normalized with `sound_norm_refs`), truncated to `max_ref_length` seconds (default 30).
- Speaker embedding: computed per file with `get_speaker_embedding` (section above) and averaged across files.
- GPT conditioning: all reference audio concatenated, then `get_gpt_cond_latents(full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len)`.

---

## 2. Discrete Audio Codebook & DVAE

### DVAE Audio Mel-Spectrogram
- **Construction:** `GPTTrainer.torch_mel_spectrogram_dvae = TorchMelSpectrogram(mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate)` (`tts/layers/xtts/trainer/gpt_trainer.py:201-203`); every other argument is the class default (`tts/layers/tortoise/arch_utils.py:194-206`).
- **Use:** `GPTTrainer.format_batch_on_device` (`tts/layers/xtts/trainer/gpt_trainer.py:258-297`).
- **Parameters:**
  - Sample rate: `22,050 Hz` (`dvae_sample_rate`)
  - FFT size (`filter_length`): `1024`
  - Hop length (`hop_length`): `256`
  - Window length (`win_length`): `1024`
  - Mel channels (`n_mel_channels`): `80`
  - Mel fmin / fmax: `0` / `8000`
  - **Normalization: log-mel divided by `mel_norms`** (section 0). There is no mean/std normalization.
- **Downstream call:** `dvae.get_codebook_indices(dvae_mel_spec)` → `[B, T_codes]`.
- **Temporal stride:** the DVAE downsamples mel frames 4× (2 stride-2 layers), so one code = `4 × 256 = 1024` samples at 22,050 Hz (`gpt_code_stride_len = 1024`).

---

## 3. Autoregressive GPT Backbone

### `GPT.forward` Latent & Attention Slicing
- **Location:** `tts/layers/xtts/gpt.py:290-476`
- **Token ids:** `start_text_token` / `stop_text_token` are **read from the tokenizer** (`[START]` / `[STOP]`, `tts/models/xtts.py:163-164`); in the stock vocab they are `261` / `0`. Audio tokens: start `1024`, stop `1025` (`num_audio_tokens = 1026`). Code stride length: `1024`.
- **Sequence layout:** `[cond_latents (32 perceiver latents), START_TEXT, text_tokens, STOP_TEXT, START_AUDIO, audio_codes, STOP_AUDIO]`. The conditioning prompt is prepended in `get_logits` (`tts/layers/xtts/gpt.py:196-201`).
- **Audio-code padding:** `code_lengths = ceil(wav_lengths / code_stride_len) + 3`; positions past each item's real length are set to the stop token by `set_mel_padding` (`tts/layers/xtts/gpt.py:170-180, 322, 344-374`).
- **Teacher-forced latent extraction:** with `return_latent=True`, `GPT.forward` returns the final-norm hidden states at audio positions, sliced as `mel_logits[:, :sub]` where **`sub = -5` in eval mode and `-1` in train mode** (`tts/layers/xtts/gpt.py:427-444`). In eval mode this yields exactly `T_codes` latents → `[B, T_codes, 1024]`. Stage 3 latent extraction must run in eval mode to match `Xtts.inference`.

---

## 4. Latent HiFi-GAN Vocoder

### `HifiDecoder.forward`
- **Location:** `tts/layers/xtts/hifigan_decoder.py:81-108`
- **Inputs:**
  - `latents`: `[B, T_lat, 1024]`
  - `g`: `[B, 512, 1]` (speaker embedding from section 1)
- **Sampling & temporal alignment:**
  1. `F.interpolate(latents.transpose(1, 2), scale_factor=ar_mel_length_compression / output_hop_length, mode="linear")` = `1024 / 256 = 4×` (linear interpolation, not transposed convolution).
  2. If `output_sample_rate != input_sample_rate`: a second `F.interpolate(..., scale_factor=24000 / 22050, mode="linear")`.
  3. `waveform_decoder(z, g=g)` upsamples each frame by `output_hop_length = 256` samples.
  - Net rate: `4 × (24000 / 22050) × 256 ≈ 1114.56` output samples per latent frame. Because of integer truncation in the interpolation, a window of `n` latents produces `floor(4n × 24000 / 22050) × 256` samples (e.g. 16 latents → 69 × 256 = 17,664 samples).
- **Output:** float waveform tensor `[B, 1, T]` at `24,000 Hz`. Conversion to 16-bit PCM happens only when a caller writes a file.
