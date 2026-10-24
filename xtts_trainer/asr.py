"""The one Whisper ASR entry point, used by data preparation, the proof script and the eval harness."""

from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np

from xtts_trainer.spec.languages import whisper_language

ENGLISH_ASR_MODEL = "openai/whisper-base.en"
MULTILINGUAL_ASR_MODEL = "openai/whisper-large-v3-turbo"


def default_asr_model(lang: str) -> str:
    """Model used to transcribe training data (docs/recipe_design.md §6, phase 6: large-v3 class for every language)."""
    whisper_language(lang)  # raises ValueError for languages Whisper cannot transcribe
    return MULTILINGUAL_ASR_MODEL


@lru_cache(maxsize=4)
def load_asr(model_name: str, device: str = "cpu") -> Any:
    """Load (and cache) a Hugging Face Whisper ASR pipeline. Failures propagate; there is no fallback model."""
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=model_name, device=device)


def _generate_kwargs(model_name: str, lang: str) -> dict:
    if model_name.endswith(".en"):
        # English-only Whisper checkpoints reject language/task arguments
        if lang != "en":
            raise ValueError(f"ASR model {model_name} is English-only; cannot transcribe lang='{lang}'.")
        return {}
    return {"language": whisper_language(lang), "task": "transcribe"}


def transcribe(audio: np.ndarray, sample_rate: int, lang: str, model_name: str, device: str = "cpu") -> str:
    """Transcribe mono audio, passing its true sample rate so the pipeline resamples correctly."""
    asr = load_asr(model_name, device)
    asr_input = {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": sample_rate}
    kwargs = _generate_kwargs(model_name, lang)
    out = asr(asr_input, generate_kwargs=kwargs) if kwargs else asr(asr_input)
    return out.get("text", "").strip()


def transcribe_phrases(
    audio: np.ndarray, sample_rate: int, lang: str, model_name: str, device: str = "cpu"
) -> List[Tuple[str, float, Optional[float]]]:
    """Long-form transcription as (text, start_s, end_s) phrases.

    The phrase times only window the recording for forced alignment; they are never used as cut points
    (Whisper's timestamps are contiguous and don't locate pauses). end_s may be None for the final phrase.
    """
    asr = load_asr(model_name, device)
    asr_input = {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": sample_rate}
    out = asr(asr_input, return_timestamps=True, chunk_length_s=30, generate_kwargs=_generate_kwargs(model_name, lang))
    phrases = []
    for chunk in out.get("chunks", []):
        text = chunk["text"].strip()
        start, end = chunk["timestamp"]
        if text:
            phrases.append((text, float(start), None if end is None else float(end)))
    return phrases
