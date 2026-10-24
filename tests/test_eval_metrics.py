"""Unit tests for evaluation metrics and evaluation dataset loading."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import torch

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
from eval.dataset import load_eval_set, save_frozen_eval_set
from scripts.run_eval_baseline import generate_baseline, resolve_reference_voices, PINNED_UPSTREAM_COMMIT


def test_whisper_text_normalization():
    """Verify Whisper standard text normalization handles punctuation, case, and numbers."""
    raw1 = "She sells 10 sea-shells by the seashore!"
    norm1 = normalize_text(raw1, lang="en")
    assert "she sells 10 sea shells by the seashore" in norm1

    raw_num = "I have 5 apples and 20 oranges."
    raw_spelled = "I have five apples and twenty oranges."
    # Both should normalize equivalently
    norm_num = normalize_text(raw_num, lang="en")
    norm_spelled = normalize_text(raw_spelled, lang="en")
    assert norm_num == norm_spelled


def test_compute_wer_cases():
    """Test Word Error Rate on exact match, substitution, insertion, deletion, and empty strings."""
    # Exact match
    assert compute_wer("Hello world", "hello world") == 0.0

    # Substitution (1 substitution in 2 words = 0.5)
    assert compute_wer("Hello world", "hello there") == 0.5

    # Insertion (1 insertion in 2 words = 0.5)
    assert compute_wer("Hello world", "hello beautiful world") == 0.5

    # Deletion (1 deletion in 2 words = 0.5)
    assert compute_wer("Hello world", "hello") == 0.5

    # Empty cases
    assert compute_wer("", "") == 0.0
    assert compute_wer("hello", "") == 1.0
    assert compute_wer("", "hello") == 1.0


def test_compute_cer_cases():
    """Test Character Error Rate on exact match, substitution, and non-spacing languages."""
    # Exact match
    assert compute_cer("apple", "apple") == 0.0

    # Substitution (1 char in 5 = 0.2)
    assert compute_cer("apple", "apply") == 0.2

    # Empty cases
    assert compute_cer("", "") == 0.0
    assert compute_cer("abc", "") == 1.0


def test_compute_secs_similarity():
    """Test Speaker Embedding Cosine Similarity."""
    # Exact identical embeddings -> 1.0
    vec1 = torch.tensor([1.0, 0.0, 0.0])
    assert pytest.approx(compute_secs(vec1, vec1), rel=1e-5) == 1.0

    # Orthogonal embeddings -> 0.0
    vec2 = torch.tensor([0.0, 1.0, 0.0])
    assert pytest.approx(compute_secs(vec1, vec2), abs=1e-5) == 0.0

    # Opposite embeddings -> -1.0
    vec3 = torch.tensor([-1.0, 0.0, 0.0])
    assert pytest.approx(compute_secs(vec1, vec3), rel=1e-5) == -1.0


def test_compute_duration_ratio():
    """Test duration ratio detects normal pacing vs babbling or early truncation."""
    text = "This is a sentence containing about forty characters."
    expected_duration = len(verbalize_text(text)) / 14.0
    ratio_normal = compute_duration_ratio(expected_duration, text, expected_cps=14.0)
    assert 0.95 <= ratio_normal <= 1.05

    # Early stop (< 0.8s)
    ratio_short = compute_duration_ratio(0.8, text, expected_cps=14.0)
    assert ratio_short < 0.35

    # Babbling (> 10.0s)
    ratio_long = compute_duration_ratio(10.0, text, expected_cps=14.0)
    assert ratio_long > 2.5


def test_duration_uses_the_words_xtts_says():
    """Digits are expanded the way XTTS expands them; Whisper's normalizer would shrink them instead."""
    text = "In 1984, the population reached 4,800,000 across 350 districts."
    assert "four million, eight hundred thousand".replace(",", "") in verbalize_text(text)
    assert len(verbalize_text(text)) > 2 * len(normalize_text(text))
    assert compute_duration_ratio(8.68, text) < 1.0  # run 2's slowest take of this sentence is normal speech
    assert verbalize_text("The six sick sheiks' sixth sheep is sick.").startswith("the six sick")


def test_too_long_needs_ratio_and_absolute_excess():
    assert not is_too_long(1.15, "Yes.")  # run 2: ratio 2.3 against the 0.5 s floor, but only 0.65 s over
    assert is_too_long(3.0, "Yes.")
    text = "This is a sentence containing about forty characters."
    estimate = len(verbalize_text(text)) / 14.0
    assert not is_too_long(2.1 * estimate, text)
    assert is_too_long(2.3 * estimate, text)


def test_audio_sanity_checks():
    """Test audio sanity checks on normal, silent, clipped, and NaN waveforms."""
    # Normal waveform: 1 second sine at 24kHz
    t = torch.linspace(0, 1, 24000)
    normal_audio = 0.2 * torch.sin(2 * np.pi * 440 * t)
    health = check_audio_sanity(normal_audio, sample_rate=24000)
    assert health["is_finite"] is True
    assert health["is_silent"] is False
    assert health["is_clipped"] is False
    assert health["duration_s"] == 1.0

    # Silent waveform
    silence = torch.zeros(24000)
    health_silence = check_audio_sanity(silence, sample_rate=24000)
    assert health_silence["is_silent"] is True

    # Clipped waveform
    clipped = torch.tensor([1.5, -1.2, 0.5])
    health_clipped = check_audio_sanity(clipped, sample_rate=24000)
    assert health_clipped["is_clipped"] is True

    # Non-finite waveform (NaN)
    nan_audio = torch.tensor([0.1, float("nan"), 0.5])
    health_nan = check_audio_sanity(nan_audio, sample_rate=24000)
    assert health_nan["is_finite"] is False
    assert health_nan["is_silent"] is True


def test_frozen_evaluation_dataset_contracts(tmp_path):
    """Test frozen evaluation set loads, contains >=50 en sentences and target languages."""
    eval_file = os.path.join(str(tmp_path), "eval_set.json")
    sha = save_frozen_eval_set(eval_file)
    assert len(sha) == 64

    data = load_eval_set(eval_file)
    assert len(data["english_sentences"]) >= 50
    assert len(data["target_languages"]["es"]) >= 30
    assert len(data["target_languages"]["fr"]) >= 30
    assert len(data["target_languages"]["de"]) >= 30
    assert len(data["target_languages"]["hi"]) >= 30
    assert len(data["reference_voices"]) >= 5


def test_baseline_requires_real_checkpoint(tmp_path):
    """A baseline must never be produced without running the model: a missing checkpoint is an error."""
    out = os.path.join(str(tmp_path), "base.json")
    with pytest.raises(FileNotFoundError, match="missing"):
        generate_baseline(
            checkpoint_dir=str(tmp_path / "no_checkpoint"),
            reference_audio_dir=str(tmp_path),
            output_path=out,
        )
    assert not os.path.exists(out)


def test_baseline_requires_all_reference_voice_audio(tmp_path):
    """Every eval-set reference voice must have a real audio file; descriptions alone are rejected."""
    data = load_eval_set()
    with pytest.raises(FileNotFoundError, match="Missing reference voice audio"):
        resolve_reference_voices(data, str(tmp_path))


def test_eval_set_is_never_regenerated_silently(tmp_path, monkeypatch):
    """A missing eval set is an error, not a fresh file: a wrong path must not silently change what is measured."""
    from eval.dataset import DEFAULT_EVAL_SET_PATH
    from eval.reference_voices import DEFAULT_LOCK_PATH, DEFAULT_OUT_DIR

    missing = tmp_path / "nope" / "eval_set.json"
    with pytest.raises(FileNotFoundError, match="frozen eval set"):
        load_eval_set(str(missing))
    assert not missing.exists()
    assert all(os.path.isabs(p) for p in (DEFAULT_EVAL_SET_PATH, DEFAULT_LOCK_PATH, DEFAULT_OUT_DIR))
    monkeypatch.chdir(tmp_path)  # defaults don't depend on the working directory
    assert load_eval_set()["sha256"] == json.load(open(DEFAULT_EVAL_SET_PATH))["sha256"]
