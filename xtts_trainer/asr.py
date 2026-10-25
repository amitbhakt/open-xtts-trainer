"""The one Whisper ASR entry point, used by data preparation, the proof script and the eval harness."""

from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from xtts_trainer.spec.languages import whisper_language

ENGLISH_ASR_MODEL = "openai/whisper-base.en"
MULTILINGUAL_ASR_MODEL = "openai/whisper-large-v3-turbo"


def default_asr_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_asr_model(lang: str) -> str:
    """Model used to transcribe training data (docs/recipe_design.md §6, phase 6: large-v3 class for every language)."""
    whisper_language(lang)  # raises ValueError for languages Whisper cannot transcribe
    return MULTILINGUAL_ASR_MODEL


@lru_cache(maxsize=4)
def _load_asr_cached(model_name: str, device: str) -> Any:
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=model_name, device=device)


def load_asr(model_name: str, device: Optional[str] = None) -> Any:
    """Load (and cache) a Hugging Face Whisper ASR pipeline. Failures propagate; there is no fallback model."""
    dev = device or default_asr_device()
    return _load_asr_cached(model_name, dev)


def _generate_kwargs(model_name: str, lang: str) -> dict:
    if model_name.endswith(".en"):
        # English-only Whisper checkpoints reject language/task arguments
        if lang != "en":
            raise ValueError(f"ASR model {model_name} is English-only; cannot transcribe lang='{lang}'.")
        return {}
    return {"language": whisper_language(lang), "task": "transcribe"}


def transcribe(audio: np.ndarray, sample_rate: int, lang: str, model_name: str, device: Optional[str] = None) -> str:
    """Transcribe mono audio, passing its true sample rate so the pipeline resamples correctly."""
    asr = load_asr(model_name, device)
    asr_input = {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": sample_rate}
    kwargs = _generate_kwargs(model_name, lang)
    out = asr(asr_input, generate_kwargs=kwargs) if kwargs else asr(asr_input)
    return out.get("text", "").strip()


def transcribe_batch(
    audios: List[np.ndarray],
    sample_rate: int,
    lang: str,
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> List[str]:
    """Transcribe a batch of mono audio waveforms using Whisper in batched pipeline mode.

    Passes a list of audio inputs to the Hugging Face pipeline with `batch_size`,
    leveraging GPU tensor parallelism and avoiding sequential invocation warnings.
    """
    if not audios:
        return []
    asr = load_asr(model_name, device)
    inputs = [{"raw": np.asarray(a, dtype=np.float32), "sampling_rate": sample_rate} for a in audios]
    kwargs = _generate_kwargs(model_name, lang)
    outs = asr(inputs, batch_size=batch_size, generate_kwargs=kwargs) if kwargs else asr(inputs, batch_size=batch_size)
    results = []
    for out in outs:
        if isinstance(out, dict):
            results.append(out.get("text", "").strip())
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            results.append(out[0].get("text", "").strip())
        else:
            results.append("")
    return results


def transcribe_phrases(
    audio: np.ndarray, sample_rate: int, lang: str, model_name: str, device: Optional[str] = None
) -> List[Tuple[str, float, Optional[float]]]:
    """Long-form transcription as (text, start_s, end_s) phrases.

    The phrase times only window the recording for forced alignment; they are never used as cut points
    (Whisper's timestamps are contiguous and don't locate pauses). Phrases are bounded to <= 20s.
    """
    audio_s = float(len(audio) / sample_rate)
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
        if not raw_text:
            return []
        if audio_s > 50.0:
            raise RuntimeError(
                f"ASR produced no word timestamps for a {audio_s:.1f}s recording (>50s); "
                "cannot safely window long-form transcript for forced alignment"
            )
        return [(raw_text, 0.0, audio_s)]

    phrases = []
    curr_words = []
    win_start = None
    last_end = 0.0
    last_chunk_unclosed = False

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        ts = chunk.get("timestamp")
        start = float(ts[0]) if (ts and ts[0] is not None) else last_end
        start = min(audio_s, max(0.0, start))
        unclosed = not (ts and len(ts) > 1 and ts[1] is not None)
        if not unclosed:
            end = min(audio_s, max(start, float(ts[1])))
        else:
            end = None

        # Split before adding a word that would cause the phrase to exceed 20s
        projected_end = end if end is not None else start
        if curr_words and win_start is not None and (projected_end - win_start > 20.0):
            phrase_end = None if last_chunk_unclosed else last_end
            phrases.append((" ".join(curr_words), win_start, phrase_end))
            curr_words = []
            win_start = None
            last_chunk_unclosed = False

        if win_start is None:
            win_start = start
        curr_words.append(text)
        if end is not None:
            last_end = max(last_end, end)
            last_chunk_unclosed = False
        else:
            last_chunk_unclosed = True

        # Handle a single oversized word explicitly, or natural punctuation split
        win_dur = (last_end if not last_chunk_unclosed else start) - win_start
        if not last_chunk_unclosed and (last_end - win_start > 20.0):
            phrases.append((" ".join(curr_words), win_start, last_end))
            curr_words = []
            win_start = None
            last_chunk_unclosed = False
        elif win_dur >= 5.0 and text.endswith((".", "!", "?")):
            phrase_end = None if last_chunk_unclosed else last_end
            phrases.append((" ".join(curr_words), win_start, phrase_end))
            curr_words = []
            win_start = None
            last_chunk_unclosed = False

    if curr_words and win_start is not None:
        phrase_end = None if last_chunk_unclosed else last_end
        phrases.append((" ".join(curr_words), win_start, phrase_end))

    return phrases
