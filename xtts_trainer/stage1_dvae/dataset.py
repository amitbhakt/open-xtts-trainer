"""Dataset for Discrete VAE (DVAE) training on mel-spectrograms."""

from typing import List

import torch
from torch.utils.data import Dataset

from xtts_trainer.config import load_audio
from xtts_trainer.frontend import dvae_mel_extractor
from xtts_trainer.spec import SPEC


def trim_to_codes(mel: torch.Tensor) -> torch.Tensor:
    """Drop trailing frames so the length is a multiple of the DVAE's downsampling (one code per 4 frames)."""
    factor = SPEC.dvae.downsample_factor
    return mel[..., : mel.shape[-1] // factor * factor]


class MelDataset(Dataset):
    """DVAE log-mel spectrograms (SPEC.dvae_mel) of whole clips, never padded.

    Training (random_crop=True) takes a random window of at most max_mel_frames; validation takes the clip
    from the start, capped at max_mel_frames. Lengths are trimmed to a multiple of the downsampling factor.
    Padding was removed: zero-padded audio and padded mel frames taught the DVAE to reconstruct frames that
    are not speech.
    """

    def __init__(self, audio_files: List[str], max_mel_frames: int = 256, random_crop: bool = True):
        super().__init__()
        self.sample_rate = SPEC.dvae_mel.sample_rate
        self.max_mel_frames = max_mel_frames - max_mel_frames % SPEC.dvae.downsample_factor
        self.random_crop = random_crop
        self.audio_files = list(audio_files)
        if not self.audio_files:
            raise ValueError("MelDataset needs at least one audio file")
        self.mel_transform = dvae_mel_extractor()

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        waveform, _ = load_audio(self.audio_files[index], target_sr=self.sample_rate)  # mono, resampled
        with torch.no_grad():
            mel = self.mel_transform(waveform).squeeze(0)  # [80, T]
        if mel.shape[-1] > self.max_mel_frames:
            # torch's RNG (seeded and restored with the run, seeded per DataLoader worker), not python's
            start = int(torch.randint(0, mel.shape[-1] - self.max_mel_frames + 1, ())) if self.random_crop else 0
            mel = mel[:, start:start + self.max_mel_frames]
        mel = trim_to_codes(mel)
        if mel.shape[-1] == 0:
            raise ValueError(f"{self.audio_files[index]} is shorter than one DVAE code")
        return mel


def collate_mels(batch: List[torch.Tensor]) -> torch.Tensor:
    """Stack by cropping every item to the batch's shortest length (already a multiple of the factor)."""
    n = min(m.shape[-1] for m in batch)
    return torch.stack([m[:, :n] for m in batch])
