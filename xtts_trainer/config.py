"""Training hyperparameters and audio loading.

XTTS-v2 constants (sample rates, spectrogram settings, model architecture, languages) live only in
xtts_trainer.spec; nothing here may redefine them.
"""

from dataclasses import dataclass
from typing import Optional

from xtts_trainer.spec.xtts_spec import SPEC


@dataclass
class HiFiGANConfig:
    """Hyperparameters for Stage 3: Latent HiFi-GAN Vocoder training (stage3_hifigan/train.py)."""
    learning_rate: float = 0.0002            # generator, training from scratch
    finetune_learning_rate: float = 0.00002  # generator, fine-tuning the released decoder
    discriminator_lr_ratio: float = 0.5      # discriminator LR = ratio x generator LR
    lr_decay: float = 0.999                  # per epoch, applied per step
    lr_warmup_steps: int = 100
    # Fine-tuning starts with fresh discriminators (the released model has none): the generator trains on the
    # mel loss alone for this many steps while the discriminators catch up. A run that never gets past it is
    # refused unless mel_only.
    disc_warmup_steps: int = 1000
    mel_only: bool = False
    batch_size: int = 4
    epochs: int = 50
    segment_frames: int = 64                 # decoder-input frames per training segment (x256 output samples)
    seed: int = 1
    val_fraction: float = 0.1
    val_every_steps: int = 500
    sample_every_steps: int = 2000           # write validation audio samples
    max_val_clips: int = 16
    mixed_precision: bool = True             # CUDA only
    scl_weight: float = 0.0                  # Speaker Consistency Loss (XTTS paper); 0 = off
    full_band_mel: bool = False              # mel loss up to Nyquist instead of 8 kHz
    grad_clip: float = 1.0
    num_workers: int = 0


@dataclass
class DVAEConfig:
    """Hyperparameters for Stage 1: Discrete VAE training (stage1_dvae/train.py).

    num_tokens / codebook_dim / hidden_dim default to the XTTS-v2 architecture (SPEC.dvae); set them
    only to build a smaller model, e.g. in tests. A base checkpoint's codebook size always wins.
    """
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100
    num_tokens: Optional[int] = None
    codebook_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    seed: int = 1
    val_fraction: float = 0.1
    val_every_steps: int = 200
    lr_warmup_steps: int = 100
    lr_decay_per_epoch: float = 1.0
    # Off by default: the nearest-code search in fp16 can assign different codes than fp32
    mixed_precision: bool = False
    frozen_codebook: bool = False
    max_mel_frames: int = 256                # training crop (~3 s at 22.05 kHz, hop 256)
    max_val_mel_frames: int = 2048
    # Codebook-collapse gate on validation data ([E5], ablation A2): at least this many active codes, and
    # at least this fraction of the reference model's active codes when fine-tuning.
    min_active_codes: int = 16
    min_active_fraction_of_base: float = 0.5
    grad_clip: float = 1.0
    num_workers: int = 0


def load_audio(path: str, target_sr: int = SPEC.output_sample_rate):
    """Loads audio file safely using soundfile or torchaudio, returns [1, T] tensor."""
    import torch
    import torchaudio
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
    except RuntimeError:  # soundfile.LibsndfileError: format libsndfile can't decode (e.g. some MP3s)
        waveform, sr = torchaudio.load(path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, target_sr
