"""Dataset for Discrete VAE (DVAE) training on mel-spectrograms."""

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple


class MelDataset(Dataset):
    """Dataset that loads audio, extracts 80-channel mel-spectrograms at 22050 Hz, and slices fixed frames."""

    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 22050,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: int = 4096,
        n_mels: int = 80,
        max_mel_frames: int = 64,
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
            raise FileNotFoundError(f"No .wav audio files found in: {audio_dir}")

        from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
        self.mel_transform = TorchMelSpectrogram(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sampling_rate=sample_rate,
            n_mel_channels=n_mels,
            mel_fmin=0,
            mel_fmax=8000,
        )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.audio_files[index]
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Compute mel-spectrogram [1, n_mels, time]
        with torch.no_grad():
            mel = self.mel_transform(waveform).squeeze(0)  # [80, time]

        time_frames = mel.shape[-1]
        if time_frames < self.max_mel_frames:
            # Pad
            pad_amount = self.max_mel_frames - time_frames
            mel = torch.nn.functional.pad(mel, (0, pad_amount))
        else:
            # Random crop
            start = random.randint(0, time_frames - self.max_mel_frames)
            mel = mel[:, start : start + self.max_mel_frames]

        return mel
