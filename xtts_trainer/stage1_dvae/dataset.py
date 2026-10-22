"""Dataset for Discrete VAE (DVAE) training on mel-spectrograms."""

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple


class MelDataset(Dataset):
    """Dataset that loads audio, extracts 80-channel mel-spectrograms, and slices fixed frames."""

    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        max_mel_frames: int = 128,  # ~1.36 seconds of audio
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_mel_frames = max_mel_frames

        self.audio_files = [
            os.path.join(audio_dir, f)
            for f in sorted(os.listdir(audio_dir))
            if f.endswith(".wav")
        ]
        if not self.audio_files:
            raise ValueError(f"No .wav audio files found in: {audio_dir}")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
            center=False,
        )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.audio_files[index]
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute log-mel spectrogram [1, n_mels, time]
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5)).squeeze(0)  # [n_mels, time]

        time_frames = log_mel.shape[-1]
        if time_frames < self.max_mel_frames:
            # Pad
            pad_amount = self.max_mel_frames - time_frames
            log_mel = torch.nn.functional.pad(log_mel, (0, pad_amount))
        else:
            # Random crop
            start = random.randint(0, time_frames - self.max_mel_frames)
            log_mel = log_mel[:, start : start + self.max_mel_frames]

        return log_mel
