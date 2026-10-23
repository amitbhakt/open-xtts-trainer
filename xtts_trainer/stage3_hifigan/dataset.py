"""Dataset and Dataloader for Latent HiFi-GAN Vocoder training."""

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class LatentHiFiGANDataset(Dataset):
    """Dataset pairing ground-truth 24kHz audio with temporal GPT latents and speaker embeddings."""

    def __init__(
        self,
        audio_dir: str,
        latents_dir: str,
        latent_frames_slice: int = 16,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.latents_dir = latents_dir
        self.latent_frames_slice = latent_frames_slice
        self.sample_rate = sample_rate

        # 16 latent frames produces int(16 * 4 * 24000 / 22050) * 256 = 17,664 audio samples @ 24kHz
        self.audio_samples_slice = int(latent_frames_slice * 4 * (24000 / 22050)) * 256

        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
        if not os.path.isdir(latents_dir):
            raise FileNotFoundError(f"Latents directory does not exist: {latents_dir}")

        self.items = []
        for file in sorted(os.listdir(latents_dir)):
            if file.endswith(".pt"):
                base_name = os.path.splitext(file)[0]
                audio_path = os.path.join(audio_dir, f"{base_name}.wav")
                latent_path = os.path.join(latents_dir, file)
                if os.path.exists(audio_path):
                    self.items.append((audio_path, latent_path))

        if not self.items:
            raise ValueError(f"No matching .wav and .pt files found between {audio_dir} and {latents_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_path, latent_path = self.items[index]

        # 1. Load Audio at 24000 Hz
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # 2. Load Latents
        latent_data = torch.load(latent_path, map_location="cpu")
        if isinstance(latent_data, dict):
            gpt_latents = latent_data["gpt_latents"]
            speaker_emb = latent_data.get("speaker_embedding", torch.zeros(1, 512, 1))
        else:
            gpt_latents = latent_data
            speaker_emb = torch.zeros(1, 512, 1)

        # Ensure latent shape is [T_lat, 1024]
        if gpt_latents.ndim == 3:
            gpt_latents = gpt_latents.squeeze(0)
        if gpt_latents.shape[-1] != 1024 and gpt_latents.shape[0] == 1024:
            gpt_latents = gpt_latents.t()

        if speaker_emb.ndim == 3:
            speaker_emb = speaker_emb.squeeze(0)  # [512, 1]
        elif speaker_emb.ndim == 1:
            speaker_emb = speaker_emb.unsqueeze(-1)  # [512, 1]

        # 3. Synchronized Segment Slicing
        latent_len = gpt_latents.shape[0]
        audio_len = waveform.shape[-1]

        # Pad if shorter than required window
        if latent_len < self.latent_frames_slice:
            pad_lat = self.latent_frames_slice - latent_len
            gpt_latents = torch.nn.functional.pad(gpt_latents, (0, 0, 0, pad_lat))
            latent_len = self.latent_frames_slice

        if audio_len < self.audio_samples_slice:
            pad_audio = self.audio_samples_slice - audio_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_audio))
            audio_len = self.audio_samples_slice

        # Compute valid random slice offsets
        max_latent_start = max(0, latent_len - self.latent_frames_slice)
        latent_start = random.randint(0, max_latent_start)
        audio_start = int(latent_start * 4 * (24000 / 22050)) * 256

        # Guard audio boundary
        if audio_start + self.audio_samples_slice > audio_len:
            audio_start = max(0, audio_len - self.audio_samples_slice)

        waveform_slice = waveform[:, audio_start : audio_start + self.audio_samples_slice]
        latents_slice = gpt_latents[latent_start : latent_start + self.latent_frames_slice, :]

        return waveform_slice, latents_slice, speaker_emb
