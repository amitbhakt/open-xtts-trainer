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
    (Whisper's timestamps are contiguous and don't locate pauses). Phrases are bounded to <= 20s.
    """
    asr = load_asr(model_name, device)
    asr_input = {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": sample_rate}
    out = asr(
        asr_input,
        return_timestamps="word",
        chunk_length_s=30,
        generate_kwargs=_generate_kwargs(model_name, lang),
    )
    chunks = out.get("chunks", [])
    if not chunks:
        raw_text = out.get("text", "").strip()
        if raw_text:
            return [(raw_text, 0.0, float(len(audio) / sample_rate))]
        return []

    phrases = []
    curr_words = []
    win_start = None
    last_end = 0.0

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        ts = chunk.get("timestamp")
        start = float(ts[0]) if (ts and ts[0] is not None) else last_end
        end = float(ts[1]) if (ts and len(ts) > 1 and ts[1] is not None) else start + 0.5
        last_end = max(last_end, end)
        if win_start is None:
            win_start = start
        curr_words.append(text)

        win_dur = last_end - win_start
        if win_dur >= 20.0 or (win_dur >= 5.0 and text.endswith((".", "!", "?"))):
            phrases.append((" ".join(curr_words), win_start, last_end))
            curr_words = []
            win_start = None

    if curr_words and win_start is not None:
        phrases.append((" ".join(curr_words), win_start, last_end))

    return phrases
