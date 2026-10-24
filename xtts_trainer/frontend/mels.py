"""The only spectrogram code in the project. Every stage, script and test builds its extractors here.

See docs/upstream_reference.md (section 0) for the log-mel recipe these must match.
"""

import torch
import torch.nn as nn
import torchaudio

from xtts_trainer.spec.artifacts import ensure_mel_stats
from xtts_trainer.spec.xtts_spec import SPEC, MelSpec


def _upstream_log_mel(mel_spec: MelSpec) -> nn.Module:
    """Upstream TorchMelSpectrogram (log-mel divided by the verified mel_stats)."""
    from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

    return TorchMelSpectrogram(
        filter_length=mel_spec.n_fft,
        hop_length=mel_spec.hop_length,
        win_length=mel_spec.win_length,
        n_mel_channels=mel_spec.n_mels,
        mel_fmin=mel_spec.fmin,
        mel_fmax=mel_spec.fmax,
        sampling_rate=mel_spec.sample_rate,
        normalize=False,
        mel_norm_file=ensure_mel_stats(),
    )


def dvae_mel_extractor(device: str = "cpu") -> nn.Module:
    """DVAE input spectrogram: waveform at SPEC.gpt_sample_rate -> [B, 80, T] normalized log-mel."""
    return _upstream_log_mel(SPEC.dvae_mel).to(device)


def cond_mel_extractor(device: str = "cpu") -> nn.Module:
    """GPT conditioning (perceiver) spectrogram, identical to the one Xtts inference feeds get_style_emb."""
    return _upstream_log_mel(SPEC.cond_mel).to(device)


class LogMel(nn.Module):
    """log(clamp(mel, 1e-5)) without mel_stats division; used only for the Stage 3 vocoder loss."""

    def __init__(self, mel_spec: MelSpec):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_spec.sample_rate,
            n_fft=mel_spec.n_fft,
            win_length=mel_spec.win_length,
            hop_length=mel_spec.hop_length,
            n_mels=mel_spec.n_mels,
            f_min=mel_spec.fmin,
            f_max=mel_spec.fmax,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(self.mel(wav), min=1e-5))


def vocoder_loss_mel(device: str = "cpu") -> nn.Module:
    """Stage 3 mel-loss front end at SPEC.output_sample_rate."""
    return LogMel(SPEC.vocoder_loss_mel).to(device)
