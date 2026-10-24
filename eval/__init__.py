"""Open-XTTS-Trainer Evaluation Framework."""

from eval.metrics import (
    normalize_text,
    compute_wer,
    compute_cer,
    compute_secs,
    compute_duration_ratio,
    is_too_long,
    verbalize_text,
    check_audio_sanity,
)

__all__ = [
    "normalize_text",
    "compute_wer",
    "compute_cer",
    "compute_secs",
    "compute_duration_ratio",
    "is_too_long",
    "verbalize_text",
    "check_audio_sanity",
]
