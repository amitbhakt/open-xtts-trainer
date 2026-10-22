"""Dataset and Dataloader for Latent HiFi-GAN Vocoder training."""

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class LatentHiFiGANDataset(Dataset):
    """Dataset pairing ground-truth audio with pre-extracted GPT latents and speaker embeddings."""

    def __init__(
        self,
        audio_dir: str,
        latents_dir: str,
        segment_size: int = 16384,
        hop_length: int = 256,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.latents_dir = latents_dir
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Match audio files with extracted latent files
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
        from xtts_trainer.config import load_audio
        audio_path, latent_path = self.items[index]

        # 1. Load Audio safely
        waveform, sr = load_audio(audio_path, self.sample_rate)

        # 2. Load Latents
        latent_data = torch.load(latent_path, map_location="cpu")
        # latent_data expected format: {"gpt_latents": Tensor[B, C, T_lat], "speaker_embedding": Tensor[B, D]}
        if isinstance(latent_data, dict):
            gpt_latents = latent_data["gpt_latents"].squeeze(0)  # [C, T_lat]
            speaker_emb = latent_data.get("speaker_embedding", torch.zeros(512)).squeeze(0)
        else:
            gpt_latents = latent_data.squeeze(0)
            speaker_emb = torch.zeros(512)

        # 3. Synchronized Segment Slicing
        # Latent frames correspond to hop_length audio samples
        latent_len = gpt_latents.shape[-1]
        latent_frames_needed = self.segment_size // self.hop_length

        audio_len = waveform.shape[-1]

        # Pad if audio is shorter than segment_size
        if audio_len < self.segment_size:
            pad_amount = self.segment_size - audio_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            audio_len = self.segment_size

        if latent_len < latent_frames_needed:
            pad_lat = latent_frames_needed - latent_len
            gpt_latents = torch.nn.functional.pad(gpt_latents, (0, pad_lat))
            latent_len = latent_frames_needed

        # Random crop window
        max_latent_start = max(0, latent_len - latent_frames_needed)
        latent_start = random.randint(0, max_latent_start)
        audio_start = latent_start * self.hop_length

        # Slice
        waveform_slice = waveform[:, audio_start : audio_start + self.segment_size]
        latents_slice = gpt_latents[:, latent_start : latent_start + latent_frames_needed]

        return waveform_slice, latents_slice, speaker_emb
