"""Latent-file schema for Stage 3: what extract_latents writes and what the vocoder dataset accepts.

A latent file is only valid for the exact audio, transcript, GPT checkpoint and spec it came from. Each file
records all four, so a stale file (audio re-cut, transcript fixed, GPT retrained, spec changed) is refused
instead of silently training the vocoder on mismatched pairs.
"""

import hashlib
import os
from typing import Any, Dict, Optional

import torch

from xtts_trainer.models.checkpoint_io import check_spec_hash, safe_torch_load, stamp

LATENT_SCHEMA_VERSION = 1
SYNTHETIC_CHECKPOINT = "synthetic"  # marks unit-test latents; never valid for real training
REQUIRED_KEYS = ("schema_version", "spec_hash", "checkpoint_sha256", "audio_sha256", "text", "lang",
                 "n_codes", "gpt_latents", "speaker_embedding")


class StaleLatentError(ValueError):
    pass


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_latent_payload(
    gpt_latents: torch.Tensor,
    speaker_embedding: torch.Tensor,
    audio_path: str,
    text: str,
    lang: str,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    return stamp({
        "schema_version": LATENT_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "audio_sha256": sha256_file(audio_path),
        "text": text,
        "lang": lang,
        "n_codes": int(gpt_latents.shape[-2]),
        "gpt_latents": gpt_latents.detach().cpu(),
        "speaker_embedding": speaker_embedding.detach().cpu(),
    })


def load_latent(
    path: str,
    audio_path: str,
    expected_checkpoint_sha256: Optional[str] = None,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """Load a latent file and refuse it unless it matches the current spec, audio and checkpoint."""
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise StaleLatentError(f"{path}: not a latent payload (got {type(payload).__name__}); re-extract it")
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise StaleLatentError(f"{path}: missing {missing}; written by an older extractor, re-extract it")
    if payload["schema_version"] != LATENT_SCHEMA_VERSION:
        raise StaleLatentError(f"{path}: schema_version {payload['schema_version']} != {LATENT_SCHEMA_VERSION}")
    check_spec_hash(payload, path)
    if payload["checkpoint_sha256"] == SYNTHETIC_CHECKPOINT and not allow_synthetic:
        raise StaleLatentError(f"{path}: synthetic test latents can't be used for training")
    if expected_checkpoint_sha256 and payload["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise StaleLatentError(f"{path}: extracted with a different GPT checkpoint; re-extract it")
    if payload["audio_sha256"] != sha256_file(audio_path):
        raise StaleLatentError(f"{path}: {os.path.basename(audio_path)} changed since extraction; re-extract it")
    return payload
