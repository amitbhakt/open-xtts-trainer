"""Unit tests for Whisper ASR module and batched transcription."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from xtts_trainer.asr import transcribe, transcribe_batch, _generate_kwargs


def test_transcribe_batch_empty_list(monkeypatch):
    """Empty audio list returns empty list immediately without calling ASR pipeline."""
    mock_load = MagicMock()
    monkeypatch.setattr("xtts_trainer.asr.load_asr", mock_load)
    res = transcribe_batch([], sample_rate=16000, lang="en", model_name="openai/whisper-base.en")
    assert res == []
    mock_load.assert_not_called()


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


def test_transcribe_phrases_rejects_missing_word_timestamps_on_long_audio(monkeypatch):
    """If Whisper produces no word chunks for long audio (>50s), raise RuntimeError."""
    from xtts_trainer.asr import transcribe_phrases

    mock_pipeline = MagicMock(return_value={"chunks": [], "text": "A very long transcript."})
    import xtts_trainer.asr as asr_module

    monkeypatch.setattr(asr_module, "load_asr", lambda model_name, device: mock_pipeline)

    audio = np.zeros(16000 * 60, dtype=np.float32)  # 60s
    with pytest.raises(RuntimeError, match="cannot safely window long-form transcript"):
        transcribe_phrases(audio, sample_rate=16000, lang="en", model_name="dummy")


def test_transcribe_phrases_unclosed_chunk_preserves_none(monkeypatch):
    """When a chunk has timestamp (start, None), None is preserved for the phrase end."""
    from xtts_trainer.asr import transcribe_phrases

    chunks = [
        {"text": "First", "timestamp": (0.0, 2.0)},
        {"text": "final", "timestamp": (2.5, None)},
    ]
    mock_pipeline = MagicMock(return_value={"chunks": chunks, "text": "First final"})
    import xtts_trainer.asr as asr_module

    monkeypatch.setattr(asr_module, "load_asr", lambda model_name, device: mock_pipeline)

    audio = np.zeros(16000 * 10, dtype=np.float32)  # 10s audio
    phrases = transcribe_phrases(audio, sample_rate=16000, lang="en", model_name="dummy")
    assert len(phrases) == 1
    assert phrases[0][0] == "First final"
    assert phrases[0][1] == 0.0
    assert phrases[0][2] is None


def test_transcribe_phrases_bounds_at_20s_before_adding_crossing_word(monkeypatch):
    """Words that would cross 20s bound trigger phrase split before being added."""
    from xtts_trainer.asr import transcribe_phrases

    chunks = [
        {"text": "word1", "timestamp": (0.0, 10.0)},
        {"text": "word2", "timestamp": (10.0, 18.0)},
        {"text": "word3", "timestamp": (18.0, 22.0)},  # would cross 20s if added
    ]
    mock_pipeline = MagicMock(return_value={"chunks": chunks, "text": "word1 word2 word3"})
    import xtts_trainer.asr as asr_module

    monkeypatch.setattr(asr_module, "load_asr", lambda model_name, device: mock_pipeline)

    audio = np.zeros(16000 * 25, dtype=np.float32)
    phrases = transcribe_phrases(audio, sample_rate=16000, lang="en", model_name="dummy")
    assert len(phrases) == 2
    assert phrases[0][0] == "word1 word2"
    assert phrases[0][1] == 0.0
    assert phrases[0][2] == 18.0
    assert phrases[1][0] == "word3"
    assert phrases[1][1] == 18.0
    assert phrases[1][2] == 22.0


def test_harness_rejects_duplicate_seeds():
    """Duplicate seeds in run_benchmark raise ValueError before running tasks."""
    from eval.harness import EvaluationHarness
    from eval.metrics import SpeakerSimilarity, UTMOSPredictor

    harness = EvaluationHarness(
        device="cpu",
        speaker_similarity=SpeakerSimilarity(model=MagicMock()),
        mos_predictor=UTMOSPredictor(model=MagicMock()),
    )
    with pytest.raises(ValueError, match="Duplicate seeds are not permitted"):
        harness.run_benchmark(
            model=MagicMock(),
            reference_audio_path="dummy.wav",
            seeds=[42, 42],
            language="en",
        )


def test_harness_custom_evaluate_utterance_override_preserved(tmp_path):
    """Subclass overriding evaluate_utterance is invoked by run_benchmark."""
    import soundfile as sf
    import torch
    from eval.harness import EvaluationHarness
    from eval.metrics import SpeakerSimilarity, UTMOSPredictor

    class CustomHarness(EvaluationHarness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_calls = []

        def evaluate_utterance(self, model, text, lang, cond_latents, speaker_emb, ref_secs_emb, seed=42, temperature=0.75):
            self.eval_calls.append((text, seed))
            return {
                "text": text,
                "lang": lang,
                "seed": seed,
                "asr_text": "custom",
                "wer": 0.0,
                "cer": 0.0,
                "secs": 0.8,
                "utmos": 4.5,
                "duration_s": 2.0,
                "duration_ratio": 1.0,
                "rms": 0.1,
                "peak": 0.5,
                "hf_ratio": 0.2,
                "synthesis_time_s": 0.1,
                "failures": {"is_empty": False, "is_too_long": False, "is_nan": False},
            }

    ref_wav = str(tmp_path / "ref.wav")
    sf.write(ref_wav, np.zeros(24000, dtype=np.float32), 24000)

    mock_model = MagicMock()
    mock_model.get_conditioning_latents.return_value = (torch.zeros(1, 1), torch.zeros(1, 1))

    mock_sim = SpeakerSimilarity(model=MagicMock())
    mock_sim.embed_utterance = MagicMock(return_value=torch.zeros(1, 192))
    mock_mos = UTMOSPredictor(model=MagicMock())

    harness = CustomHarness(device="cpu", speaker_similarity=mock_sim, mos_predictor=mock_mos)
    res = harness.run_benchmark(
        model=mock_model,
        reference_audio_path=ref_wav,
        seeds=[123],
        subset_size=2,
        language="en",
    )
    assert len(harness.eval_calls) == 2
    assert res["summary"]["wer"]["mean"] == 0.0


def test_harness_custom_transcribe_override_preserved():
    """Subclass overriding transcribe is invoked during transcribe_batch."""
    from eval.harness import EvaluationHarness
    from eval.metrics import SpeakerSimilarity, UTMOSPredictor

    class TranscribeHarness(EvaluationHarness):
        def transcribe(self, audio, sample_rate=24000, lang="en"):
            return "custom transcription"

    harness = TranscribeHarness(
        device="cpu",
        speaker_similarity=SpeakerSimilarity(model=MagicMock()),
        mos_predictor=UTMOSPredictor(model=MagicMock()),
    )
    audios = [np.zeros(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]
    results = harness.transcribe_batch(audios, sample_rate=16000, lang="en")
    assert results == ["custom transcription", "custom transcription"]
