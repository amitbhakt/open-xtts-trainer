"""Shared helpers for tests: write a valid dataset.jsonl and schema-valid synthetic latent files."""

import json
import os
from typing import List, Optional

import soundfile as sf
import torch

from xtts_trainer.data import DATASET_FILENAME, DatasetRecord, write_dataset
from xtts_trainer.data.latents import SYNTHETIC_CHECKPOINT, make_latent_payload
from xtts_trainer.spec import SPEC


def write_test_dataset(
    dataset_dir: str,
    wav_paths: List[str],
    texts: Optional[List[str]] = None,
    speaker: str = "spk",
    lang: str = "en",
) -> str:
    """dataset.jsonl in dataset_dir for existing wav files; one speaker, durations read from the files."""
    texts = texts or ["hello world from the trainer"] * len(wav_paths)
    records = []
    for path, text in zip(wav_paths, texts):
        info = sf.info(path)
        records.append(DatasetRecord(
            audio=os.path.relpath(path, dataset_dir),
            text=text,
            lang=lang,
            speaker=speaker,
            duration_s=round(info.frames / info.samplerate, 4),
            sr=info.samplerate,
            transcript_source="human",
            single_clip_speaker=len(wav_paths) == 1,
        ))
    return write_dataset(os.path.join(dataset_dir, DATASET_FILENAME), records)


def write_synthetic_latent(latent_path: str, audio_path: str, n_frames: int, text: str = "hello") -> str:
    """A schema-valid latent file marked synthetic (only loadable with allow_synthetic_latents=True)."""
    payload = make_latent_payload(
        torch.randn(1, n_frames, SPEC.gpt.model_dim),
        torch.randn(1, SPEC.d_vector_dim, 1),
        audio_path, text, "en", SYNTHETIC_CHECKPOINT,
    )
    torch.save(payload, latent_path)
    return latent_path


def write_test_vocab(path: str, letters: str = "abcdefghijklmnopqrstuvwxyz") -> str:
    """Minimal XTTS-style tokenizer: letters, specials, language tags, and unk_token set like the real vocab."""
    vocab = {"[STOP]": 0, "[UNK]": 1, "[SPACE]": 2, "[START]": 3, "[en]": 4, "[hi]": 5}
    for ch in letters + ".,?!'":
        vocab[ch] = len(vocab)
    added = [{"id": vocab[t], "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for t in ("[STOP]", "[UNK]", "[SPACE]", "[START]", "[en]", "[hi]")]
    data = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
            "pre_tokenizer": None, "post_processor": None, "decoder": None,
            "model": {"type": "BPE", "vocab": vocab, "merges": [], "unk_token": "[UNK]"}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path
