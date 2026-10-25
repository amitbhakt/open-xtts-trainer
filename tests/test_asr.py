"""Unit tests for Whisper ASR module and batched transcription."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from xtts_trainer.asr import transcribe, transcribe_batch, _generate_kwargs


def test_transcribe_batch_empty_list():
    """Empty audio list returns empty list immediately without calling ASR pipeline."""
    res = transcribe_batch([], sample_rate=16000, lang="en", model_name="openai/whisper-base.en")
    assert res == []


def test_transcribe_batch_delegates_to_pipeline_with_batch_size(monkeypatch):
    """Verify transcribe_batch passes audio dictionaries with batch_size to pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [
        {"text": "  First sentence. "},
        {"text": "Second sentence.  "},
    ]

    import xtts_trainer.asr as asr_module

    monkeypatch.setattr(asr_module, "load_asr", lambda model_name, device: mock_pipeline)

    audio1 = np.zeros(16000, dtype=np.float32)
    audio2 = np.ones(16000, dtype=np.float32)

    results = transcribe_batch(
        [audio1, audio2],
        sample_rate=16000,
        lang="en",
        model_name="openai/whisper-base.en",
        device="cpu",
        batch_size=8,
    )

    assert results == ["First sentence.", "Second sentence."]
    mock_pipeline.assert_called_once()
    call_args, call_kwargs = mock_pipeline.call_args
    assert call_kwargs["batch_size"] == 8
    assert len(call_args[0]) == 2
    assert np.array_equal(call_args[0][0]["raw"], audio1)
    assert call_args[0][0]["sampling_rate"] == 16000


def test_transcribe_batch_multilingual_kwargs(monkeypatch):
    """Verify multilingual language kwargs are passed to the pipeline in batch mode."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"text": "Bonjour tout le monde."}]

    import xtts_trainer.asr as asr_module

    monkeypatch.setattr(asr_module, "load_asr", lambda model_name, device: mock_pipeline)

    audio = np.zeros(16000, dtype=np.float32)
    results = transcribe_batch(
        [audio],
        sample_rate=16000,
        lang="fr",
        model_name="openai/whisper-large-v3-turbo",
        device="cpu",
        batch_size=4,
    )

    assert results == ["Bonjour tout le monde."]
    call_args, call_kwargs = mock_pipeline.call_args
    assert call_kwargs["batch_size"] == 4
    assert call_kwargs["generate_kwargs"] == {"language": "fr", "task": "transcribe"}


def test_harness_transcribe_batch_delegation(monkeypatch):
    """Verify EvaluationHarness.transcribe_batch delegates to asr_transcribe_batch."""
    from eval.harness import EvaluationHarness
    from eval.metrics import SpeakerSimilarity, UTMOSPredictor

    mock_batch = MagicMock(return_value=["Transcription A", "Transcription B"])
    import eval.harness as harness_module

    monkeypatch.setattr(harness_module, "asr_transcribe_batch", mock_batch)

    harness = EvaluationHarness(
        device="cpu",
        speaker_similarity=SpeakerSimilarity(model=MagicMock()),
        mos_predictor=UTMOSPredictor(model=MagicMock()),
    )

    audios = [np.zeros(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]
    res = harness.transcribe_batch(audios, sample_rate=24000, lang="en", batch_size=8)

    assert res == ["Transcription A", "Transcription B"]
    mock_batch.assert_called_once_with(
        audios,
        24000,
        "en",
        "openai/whisper-large-v3",
        device="cpu",
        batch_size=8,
    )
