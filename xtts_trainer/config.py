"""Audio and model configuration defaults for XTTS v2 training."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AudioConfig:
    """Standard audio configuration matching XTTS v2."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0


@dataclass
class HiFiGANConfig:
    """Configuration for Stage 3: Latent HiFi-GAN Vocoder training."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    segment_size: int = 16384  # ~0.68 seconds at 24kHz
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    learning_rate: float = 0.0002
    discriminator_learning_rate: float = 0.0001
    lr_decay: float = 0.999
    warmup_steps: int = 500
    batch_size: int = 16
    epochs: int = 50


@dataclass
class DVAEConfig:
    """Configuration for Stage 1: Discrete VAE training."""
    sample_rate: int = 22050
    n_mels: int = 80
    num_tokens: int = 1024
    codebook_dim: int = 512
    hidden_dim: int = 512
    learning_rate: float = 0.0001
    temp_start: float = 1.0
    temp_min: float = 0.5
    temp_decay: float = 0.99995
    batch_size: int = 32
    epochs: int = 100


def safe_torch_load(path: str, map_location="cpu"):
    """Loads a PyTorch checkpoint safely across PyTorch 2.0 to 2.6+."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_audio(path: str, target_sr: int = 24000):
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
    except Exception:
        waveform, sr = torchaudio.load(path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, target_sr

