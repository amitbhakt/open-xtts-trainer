"""CTC forced alignment: word start/end times for a known transcript (docs/recipe_design.md §6, phase 6).

Whisper's word timestamps (Hugging Face cross-attention DTW) are contiguous: each word ends where the next
begins, so they cannot locate pauses (tests/fixtures/jfk_whisper_alignment.json; docs/recipe_design.md F5).
Word times here come from torchaudio's forced_align over MMS_FA emissions (Pratap et al., 2023).
MMS_FA weights are CC BY-NC 4.0 and download on first use (1.26 GB).

CTC emissions are peaky, so a word's end is its last character's frame: measured gaps include the word's
acoustic tail and run slightly long. Thresholds built on gaps are [E4]/[E5] until ablation A6 calibrates them.
"""

import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

# The aligner model's own input rate (read from the bundle; no download)
ALIGN_SAMPLE_RATE = torchaudio.pipelines.MMS_FA.sample_rate
STAR = "*"
BLANK_INDEX = 0
# Longest window aligned in one pass (wav2vec2 attention cost grows with the square of its length)
MAX_WINDOW_S = 60.0


class AlignmentError(ValueError):
    """The transcript cannot be aligned to the audio at all (too long for the window, nothing alignable)."""


@dataclass
class AlignedWord:
    text: str  # as written in the transcript, punctuation included
    start_s: float
    end_s: float
    score: float  # mean frame probability of the word's characters
    star: bool = False  # no alignable characters (digits, symbols): matched by the wildcard token

    def to_json(self, offset_s: float = 0.0) -> dict:
        return {"w": self.text, "start_s": round(self.start_s - offset_s, 3),
                "end_s": round(self.end_s - offset_s, 3), "score": round(self.score, 3)}


class Emitter:
    """Audio (16 kHz mono) -> CTC log-probabilities [frames, len(labels)]. labels[0] is the blank."""

    name: str
    labels: Tuple[str, ...]

    def __call__(self, audio_16k: np.ndarray) -> torch.Tensor:
        raise NotImplementedError


class MmsFaEmitter(Emitter):
    """torchaudio's MMS_FA bundle, with the wildcard (star) column appended as upstream does."""

    name = "torchaudio-mms_fa"

    def __init__(self, device: Optional[str] = None):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torchaudio.pipelines.MMS_FA
        self.labels = tuple(bundle.get_labels(star=STAR))
        self.model = bundle.get_model(with_star=True).to(dev).eval()
        self.device = dev

    @torch.inference_mode()
    def __call__(self, audio_16k: np.ndarray) -> torch.Tensor:
        wav = torch.as_tensor(np.asarray(audio_16k, dtype=np.float32), device=self.device).unsqueeze(0)
        emission, _ = self.model(wav)
        return emission[0].float().cpu()


def romanize_word(word: str, labels: Sequence[str]) -> Optional[str]:
    """The aligner's spelling of one transcript token.

    Returns "" for pure punctuation (not spoken), None when the token is spoken but has no alignable
    characters (digits, symbols), else lowercase characters from labels. Latin diacritics are stripped;
    other scripts need uroman (pip install uroman) and raise without it.
    """
    allowed = set(labels) - {labels[BLANK_INDEX], STAR}
    if not any(unicodedata.category(ch)[0] in "LN" for ch in word):
        return ""
    if any(unicodedata.category(ch)[0] == "L" and not _is_latin(ch) for ch in word):
        try:
            import uroman
        except ImportError as e:
            raise AlignmentError(
                f"'{word}' is not in Latin script; MMS_FA needs romanized text. Install uroman (pip install uroman)."
            ) from e
        word = uroman.Uroman().romanize_string(word)
    base = unicodedata.normalize("NFKD", word.replace("’", "'")).lower()
    spelled = "".join(ch for ch in base if ch in allowed)
    return spelled.strip("'") or None


def _is_latin(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "")


def transcript_units(text: str, labels: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    """Split a transcript into (display text, aligner spelling or None for star) units.

    Punctuation-only tokens ("—", "-") attach to the previous unit, so every unit is one spoken word.
    """
    units: List[List] = []
    for token in text.split():
        spelled = romanize_word(token, labels)
        if spelled == "":
            if units:
                units[-1][0] += " " + token
            continue
        units.append([token, spelled])
    if not units:
        raise AlignmentError(f"nothing to align in transcript {text!r}")
    return [(display, spelled) for display, spelled in units]


def align_emission(
    emission: torch.Tensor, units: List[Tuple[str, Optional[str]]], labels: Sequence[str], frame_s: float
) -> Tuple[List[AlignedWord], np.ndarray]:
    """Forced-align units to an emission [frames, classes].

    Returns the words (seconds from the emission's start) and the forced path (label id per frame).
    """
    index = {c: i for i, c in enumerate(labels)}
    star_id = index[STAR]
    unit_ids = [[star_id] if spelled is None else [index[c] for c in spelled] for _, spelled in units]
    targets = torch.tensor([i for ids in unit_ids for i in ids], dtype=torch.int32)
    # CTC needs a frame per token, plus one per repeated neighbour (a blank must separate them)
    needed = len(targets) + sum(1 for a, b in zip(targets[:-1], targets[1:]) if a == b)
    if emission.shape[0] < needed:
        raise AlignmentError(f"{emission.shape[0]} frames cannot hold {needed} aligned characters: text too long for audio")
    path, scores = F.forced_align(emission.unsqueeze(0), targets.unsqueeze(0), blank=BLANK_INDEX)
    spans = F.merge_tokens(path[0], scores[0].exp(), blank=BLANK_INDEX)
    words, k = [], 0
    for (display, spelled), ids in zip(units, unit_ids):
        token_spans = spans[k:k + len(ids)]
        k += len(ids)
        n_frames = sum(s.end - s.start for s in token_spans)
        score = sum(s.score * (s.end - s.start) for s in token_spans) / n_frames
        words.append(AlignedWord(display, token_spans[0].start * frame_s, token_spans[-1].end * frame_s,
                                 float(score), star=spelled is None))
    return words, path[0].numpy()


@dataclass
class WindowAlignment:
    """One aligned window: words in recording time, plus speech the transcript doesn't explain (for checks)."""

    words: List[AlignedWord]
    offset_s: float
    frame_s: float
    # Per frame: the model hears a character (argmax not blank or wildcard) but the forced path says blank.
    # Missing text shows up here wherever the aligner puts it: between words, or inside a word it stretched.
    unexplained: np.ndarray

    def unexplained_speech_s(self, start_s: float, end_s: float) -> float:
        """Seconds within [start_s, end_s) of speech that no transcript character accounts for."""
        times = self.offset_s + (np.arange(len(self.unexplained)) + 0.5) * self.frame_s
        return float((self.unexplained & (times >= start_s) & (times < end_s)).sum() * self.frame_s)


class CtcAligner:
    """Aligns a transcript to audio. emitter defaults to MMS_FA (loaded on first use)."""

    def __init__(self, emitter: Optional[Emitter] = None, device: Optional[str] = None, max_window_s: float = MAX_WINDOW_S):
        self._emitter = emitter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_window_s = max_window_s

    @property
    def emitter(self) -> Emitter:
        if self._emitter is None:
            self._emitter = MmsFaEmitter(self.device)
        return self._emitter

    @property
    def name(self) -> str:
        return MmsFaEmitter.name if self._emitter is None else self._emitter.name

    def align(self, audio: np.ndarray, sample_rate: int, text: str, offset_s: float = 0.0) -> WindowAlignment:
        duration = len(audio) / sample_rate
        if duration > self.max_window_s:
            raise AlignmentError(f"{duration:.1f}s window exceeds {self.max_window_s}s; split it first")
        audio_16k = to_align_rate(audio, sample_rate)
        emitter = self.emitter
        emission = emitter(audio_16k)
        frame_s = len(audio_16k) / ALIGN_SAMPLE_RATE / emission.shape[0]
        words, path = align_emission(emission, transcript_units(text, emitter.labels), emitter.labels, frame_s)
        for w in words:
            w.start_s += offset_s
            w.end_s += offset_s
        heard = emission.clone()
        heard[:, emitter.labels.index(STAR)] = -float("inf")
        speech = heard.argmax(dim=-1).numpy() != BLANK_INDEX
        return WindowAlignment(words, offset_s, frame_s, speech & (path == BLANK_INDEX))


def to_align_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    wav = torch.as_tensor(np.asarray(audio, dtype=np.float32))
    if sample_rate != ALIGN_SAMPLE_RATE:
        wav = F.resample(wav, sample_rate, ALIGN_SAMPLE_RATE)
    return wav.numpy()
