"""Audio and model configuration defaults for XTTS v2 training."""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default path to local mel_stats.pth normalized tensor (matching upstream XTTS-v2)
DEFAULT_MEL_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mel_stats.pth")
XTTS_MEL_STATS_SHA256 = "1f69422a8a8f344c4fca2f0c6b8d41d2151d6615b7321e48e6bb15ae949b119c"
XTTS_MEL_STATS_URL = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"


# ISO 639-1 two-letter codes + stock XTTS & common BCP-47 language identifiers
VALID_LANGUAGE_CODES = {
    # Stock 17 XTTS-v2 languages:
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
    # Extended language additions (Indic, European, Asian, African, Middle Eastern):
    "bn", "el", "ta", "te", "ur", "fa", "th", "vi", "id", "ms", "ro", "sv", "da", "fi", "no", "he", "uk",
    "bg", "ca", "hr", "sk", "sl", "lt", "lv", "et", "sw", "af", "gu", "kn", "ml", "mr", "pa", "dv", "ne",
    "si", "my", "km", "lo", "ka", "am", "so", "sq", "az", "hy", "eu", "gl", "is", "mk", "mt", "cy", "ga",
    "eo", "la", "tl", "jv", "su", "zh", "pt-br", "es-la",
}


def is_language_code(s: str, extra_languages: Optional[List[str]] = None) -> bool:
    """Determine if a token represents an ISO 639 / BCP-47 language code rather than a speaker identifier.

    Checks against standard ISO-639-1 language codes and any explicitly registered extra languages.
    Speaker identifiers (e.g. 'jfk', 'amy', 'bob', 'lj', 'spk1') are correctly rejected as non-languages.
    """
    token = s.strip().lower()
    if token in VALID_LANGUAGE_CODES:
        return True
    if extra_languages and token in [l.lower() for l in extra_languages]:
        return True
    return False


def ensure_mel_stats(target_path: Optional[str] = None) -> str:
    """Ensure that the upstream XTTS-v2 mel_stats.pth file exists locally and matches SHA-256."""
    path = target_path or DEFAULT_MEL_STATS_PATH
    import urllib.request
    import hashlib

    # Verify SHA-256 hash on every load, even if file already exists
    if os.path.exists(path):
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash == XTTS_MEL_STATS_SHA256:
            return path
        logger.warning(
            f"Existing mel_stats.pth at {path} SHA-256 mismatch (got {file_hash}, expected {XTTS_MEL_STATS_SHA256}). "
            f"Re-downloading authoritative upstream XTTS tensor from Hugging Face..."
        )
        try:
            os.remove(path)
        except OSError:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    logger.info(f"Downloading upstream XTTS-v2 mel_stats.pth from {XTTS_MEL_STATS_URL} to {path}...")
    urllib.request.urlretrieve(XTTS_MEL_STATS_URL, path)

    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    if file_hash != XTTS_MEL_STATS_SHA256:
        os.remove(path)
        raise RuntimeError(
            f"Downloaded mel_stats.pth SHA-256 mismatch! Expected {XTTS_MEL_STATS_SHA256}, got {file_hash}"
        )

    logger.info("Successfully downloaded and verified upstream mel_stats.pth.")
    return path


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
    batch_size: int = 4
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

