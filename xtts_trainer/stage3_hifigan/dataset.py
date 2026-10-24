"""Dataset for Stage 3: ground-truth audio segments paired with the decoder input that produces them.

The GPT latents of each whole clip are interpolated to decoder frames first (hifigan_decoder_input, exactly
HifiDecoder.forward's input stage), then a window of frames and the matching audio are cut. Each decoder
frame becomes exactly SPEC.output_hop_length output samples, so frame i <-> samples [i*hop, (i+1)*hop): the
alignment holds by construction, and the window edges are interpolated with their real neighbours. (Slicing
latents first and interpolating each slice, as before, interpolated the window edges against nothing.)
"""

import os
from types import SimpleNamespace
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from xtts_trainer.config import load_audio
from xtts_trainer.data import DatasetRecord
from xtts_trainer.data.latents import StaleLatentError, load_latent
from xtts_trainer.models import hifigan_decoder_input
from xtts_trainer.spec import SPEC
from xtts_trainer.stage3_hifigan.extract_latents import latent_filename

SPEECH_RMS = 0.015  # windows quieter than this are re-drawn (up to 8 tries, then a scan)


def load_pair(audio_path: str, latent_path: str, checkpoint_sha256: str, allow_synthetic: bool):
    """(audio [1, T], latents [1, T_lat, C], speaker embedding [512, 1]) for one clip."""
    waveform, _ = load_audio(audio_path, target_sr=SPEC.output_sample_rate)
    payload = load_latent(latent_path, audio_path, checkpoint_sha256, allow_synthetic)
    latents = payload["gpt_latents"]
    if latents.ndim == 2:
        latents = latents.unsqueeze(0)
    speaker = payload["speaker_embedding"].reshape(SPEC.d_vector_dim, 1)
    return waveform, latents, speaker


class LatentHiFiGANDataset(Dataset):
    """Items: (audio [1, n*hop], decoder frames [C, n], speaker embedding [512, 1]) with n <= segment_frames.

    n is the segment length unless the clip is shorter; collate_segments crops a batch to its shortest item.
    Records come from a validated dataset.jsonl; every latent file must pass the latent schema and come from
    one GPT checkpoint.
    """

    def __init__(self, records: List[DatasetRecord], dataset_dir: str, latents_dir: str, decoder=None,
                 segment_frames: int = 64, allow_synthetic_latents: bool = False):
        super().__init__()
        if not os.path.isdir(latents_dir):
            raise FileNotFoundError(f"Latents directory does not exist: {latents_dir}")
        # Only the interpolation settings; the module itself (possibly on GPU) stays out of loader workers.
        # Without a decoder, the spec's values (the ones build_hifigan uses).
        keys = ("ar_mel_length_compression", "output_hop_length", "input_sample_rate", "output_sample_rate")
        spec_values = (SPEC.gpt.code_stride_len, SPEC.output_hop_length, SPEC.gpt_sample_rate, SPEC.output_sample_rate)
        self.decoder = SimpleNamespace(**({k: getattr(decoder, k) for k in keys} if decoder is not None
                                          else dict(zip(keys, spec_values))))
        self.segment_frames = segment_frames
        self.hop = SPEC.output_hop_length
        self.allow_synthetic_latents = allow_synthetic_latents
        self.items, missing, checkpoints = [], [], set()
        for record in records:
            audio_path = record.audio_path(dataset_dir)
            latent_path = os.path.join(latents_dir, latent_filename(record.audio))
            if not os.path.exists(latent_path):
                missing.append(latent_path)
                continue
            checkpoints.add(load_latent(latent_path, audio_path, allow_synthetic=allow_synthetic_latents)["checkpoint_sha256"])
            self.items.append((audio_path, latent_path))
        if missing:
            raise FileNotFoundError(f"{len(missing)} clip(s) have no latent file, e.g. {missing[:3]}; run extract_latents")
        if len(checkpoints) > 1:
            raise StaleLatentError(f"Latents come from {len(checkpoints)} different GPT checkpoints; re-extract them all")
        if not self.items:
            raise ValueError(f"No clips to train on in {latents_dir}")
        self.checkpoint_sha256 = checkpoints.pop()

    def __len__(self) -> int:
        return len(self.items)

    def full_clip(self, index: int):
        """(audio, latents, speaker) of a whole clip, for validation."""
        return load_pair(*self.items[index], self.checkpoint_sha256, self.allow_synthetic_latents)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform, latents, speaker = self.full_clip(index)
        with torch.no_grad():
            z = hifigan_decoder_input(self.decoder, latents)[0]  # [C, F] for the whole clip
        n_frames = min(z.shape[-1], waveform.shape[-1] // self.hop)
        n = min(self.segment_frames, n_frames)
        max_start = n_frames - n

        def window(start):
            return waveform[:, start * self.hop:(start + n) * self.hop], z[:, start:start + n]

        best, best_rms = None, -1.0
        # torch's RNG, not python's: it is seeded and restored with the run, and DataLoader seeds it per worker
        starts = torch.randint(0, max_start + 1, (8,)).tolist()
        starts += list(range(0, max_start + 1, max(1, max_start // 8)))  # deterministic scan as a fallback
        for start in starts:
            audio, frames = window(start)
            rms = float(audio.pow(2).mean().sqrt())
            if rms > best_rms:
                best, best_rms = (audio, frames), rms
            if rms >= SPEECH_RMS:
                break
        return best[0], best[1], speaker


def collate_segments(batch):
    """Crop every item to the batch's shortest segment (audio and frames together)."""
    n = min(frames.shape[-1] for _, frames, _ in batch)
    hop = SPEC.output_hop_length
    audio = torch.stack([a[:, :n * hop] for a, _, _ in batch])
    frames = torch.stack([f[:, :n] for _, f, _ in batch])
    speakers = torch.stack([s for _, _, s in batch])
    return audio, frames, speakers
