"""Evaluation metrics for Open-XTTS-Trainer.

Provides standardized, objective metrics:
1. Whisper Text Normalization (WhisperTokenizer / EnglishTextNormalizer / BasicTextNormalizer).
2. Word Error Rate (WER) and Character Error Rate (CER) via dynamic programming Levenshtein distance.
3. Speaker Embedding Cosine Similarity (SECS).
4. Duration ratio and audio quality / sanity checks (clipping, RMS collapse, NaNs).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torchaudio

from xtts_trainer.spec import SPEC

from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


_CACHED_WHISPER_TOKENIZER = None
_CACHED_BASIC_NORMALIZER = None


def get_whisper_normalizer(lang: str = "en"):
    """Get the standard Whisper text normalizer for the specified language.

    English uses Whisper's EnglishTextNormalizer (via the tokenizer, which carries its spelling map); other
    languages use BasicTextNormalizer. Failures raise: a silently different normalizer changes every WER.
    """
    global _CACHED_WHISPER_TOKENIZER, _CACHED_BASIC_NORMALIZER

    if lang in ("en", "english"):
        if _CACHED_WHISPER_TOKENIZER is None:
            _CACHED_WHISPER_TOKENIZER = WhisperTokenizer.from_pretrained("openai/whisper-base.en")
        return _CACHED_WHISPER_TOKENIZER.normalize

    if _CACHED_BASIC_NORMALIZER is None:
        _CACHED_BASIC_NORMALIZER = BasicTextNormalizer(remove_diacritics=False)
    return _CACHED_BASIC_NORMALIZER


def normalize_text(text: str, lang: str = "en") -> str:
    """Normalize text using the official Whisper normalizer.

    Args:
        text: Raw text string.
        lang: Language code ('en', 'es', etc.).

    Returns:
        Standardized lowercase normalized text string.
    """
    if not text:
        return ""
    return get_whisper_normalizer(lang=lang)(text)


def _levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    d = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,      # deletion
                    d[i, j - 1] + 1,      # insertion
                    d[i - 1, j - 1] + 1,  # substitution
                )
    return int(d[n, m])


def compute_wer(reference: str, hypothesis: str, lang: str = "en") -> float:
    """Compute Word Error Rate (WER) using standard text normalization.

    Args:
        reference: Ground truth reference transcript.
        hypothesis: ASR generated transcription.
        lang: Language code for normalization.

    Returns:
        WER as a float in [0.0, +inf).
    """
    ref_norm = normalize_text(reference, lang=lang)
    hyp_norm = normalize_text(hypothesis, lang=lang)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    edit_dist = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_dist / len(ref_words))


def compute_cer(reference: str, hypothesis: str, lang: str = "en") -> float:
    """Compute Character Error Rate (CER) using standard text normalization.

    Args:
        reference: Ground truth reference transcript.
        hypothesis: ASR generated transcription.
        lang: Language code for normalization.

    Returns:
        CER as a float in [0.0, +inf).
    """
    ref_norm = normalize_text(reference, lang=lang)
    hyp_norm = normalize_text(hypothesis, lang=lang)

    # Strip whitespace for character-level distance if desired, or keep non-space chars
    ref_chars = [c for c in ref_norm if not c.isspace()]
    hyp_chars = [c for c in hyp_norm if not c.isspace()]

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    edit_dist = _levenshtein_distance(ref_chars, hyp_chars)
    return float(edit_dist / len(ref_chars))


def compute_secs(
    ref_emb: Union[np.ndarray, torch.Tensor],
    syn_emb: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Speaker Embedding Cosine Similarity (SECS) between reference and synthesis.

    Args:
        ref_emb: 1D or 2D embedding vector of reference speaker.
        syn_emb: 1D or 2D embedding vector of synthesized audio.

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    if isinstance(ref_emb, np.ndarray):
        ref_t = torch.from_numpy(ref_emb).float()
    else:
        ref_t = ref_emb.detach().cpu().float()

    if isinstance(syn_emb, np.ndarray):
        syn_t = torch.from_numpy(syn_emb).float()
    else:
        syn_t = syn_emb.detach().cpu().float()

    ref_flat = ref_t.flatten().unsqueeze(0)
    syn_flat = syn_t.flatten().unsqueeze(0)

    sim = torch.cosine_similarity(ref_flat, syn_flat, dim=-1).item()
    return float(sim)


ECAPA2_REPO_ID = "Jenthe/ECAPA2"  # CC BY-NC 4.0; same speaker encoder the XTTS paper uses for SECS
ECAPA2_FILENAME = "ecapa2.pt"
ECAPA2_SAMPLE_RATE = 16000
UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"  # MIT
UTMOS_MODEL_NAME = "utmos22_strong"
UTMOS_SAMPLE_RATE = 16000


def _to_mono_tensor(audio: Union[np.ndarray, torch.Tensor], sample_rate: int, target_rate: int) -> torch.Tensor:
    """[1, T] float tensor at target_rate."""
    wav = audio.detach().cpu().float() if isinstance(audio, torch.Tensor) else torch.from_numpy(np.asarray(audio)).float()
    wav = wav.squeeze()
    if wav.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {tuple(wav.shape)}")
    if sample_rate != target_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_rate)
    return wav.unsqueeze(0)


class SpeakerSimilarity:
    """SECS with ECAPA2, independent of XTTS's own speaker encoder (which the model is conditioned on).

    `model` may be injected (tests); otherwise the TorchScript checkpoint is fetched from the Hugging Face Hub.
    """

    def __init__(self, device: str = "cpu", model: Optional[Any] = None):
        self.device = device
        self._model = model

    def _get_model(self):
        if self._model is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=ECAPA2_REPO_ID, filename=ECAPA2_FILENAME)
            self._model = torch.jit.load(path, map_location=self.device)
        return self._model

    def embed(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int) -> torch.Tensor:
        wav = _to_mono_tensor(audio, sample_rate, ECAPA2_SAMPLE_RATE).to(self.device)
        with torch.inference_mode():
            return self._get_model()(wav).detach().cpu().float()

    def similarity(self, ref_emb: torch.Tensor, audio: Union[np.ndarray, torch.Tensor], sample_rate: int) -> float:
        return compute_secs(ref_emb, self.embed(audio, sample_rate))


class UTMOSPredictor:
    """Predicted naturalness (UTMOS22 strong). Only meaningful relative to a baseline, per the XTTS authors."""

    def __init__(self, device: str = "cpu", model: Optional[Any] = None):
        self.device = device
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = torch.hub.load(UTMOS_HUB_REPO, UTMOS_MODEL_NAME, trust_repo=True).to(self.device)
            self._model.eval()
        return self._model

    def score(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int) -> float:
        wav = _to_mono_tensor(audio, sample_rate, UTMOS_SAMPLE_RATE).to(self.device)
        with torch.inference_mode():
            return float(self._get_model()(wav, UTMOS_SAMPLE_RATE).reshape(-1)[0].item())


# Languages upstream XTTS cleans with multilingual_cleaners (VoiceBpeTokenizer.preprocess_text); zh and ko
# are transliterated after cleaning, which doesn't change how much is spoken.
MULTILINGUAL_CLEANER_LANGS = {"ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}
# Duration-based "too long" (babbling): the ratio alone flags normal one-word utterances ("Yes." at 1.15 s is
# ratio 2.3 against the 0.5 s floor, Colab run 2), so the excess must also be at least this many seconds.
TOO_LONG_RATIO = 2.2
TOO_LONG_MIN_EXCESS_S = 1.0
MIN_ESTIMATED_DURATION_S = 0.5


def verbalize_text(text: str, lang: str = "en") -> str:
    """What the model is asked to say, without punctuation: upstream XTTS's own text cleaner.

    Numbers, abbreviations and symbols are expanded the way XTTS expands them ("4,800,000" -> "four million,
    eight hundred thousand"). Whisper's normalizer goes the other way ("six" -> "6"), so it must not be used
    to estimate how long speech should take.
    """
    import re
    import unicodedata

    code = lang.split("-")[0]
    if code in MULTILINGUAL_CLEANER_LANGS:
        from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

        text = multilingual_cleaners(text, code)
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def estimated_duration_s(text: str, lang: str = "en", expected_cps: float = 14.0) -> float:
    return max(len(verbalize_text(text, lang)) / expected_cps, MIN_ESTIMATED_DURATION_S)


def compute_duration_ratio(
    actual_duration_s: float,
    text: str,
    lang: str = "en",
    expected_cps: float = 14.0,
) -> float:
    """Actual duration / estimated natural duration of the verbalized text (verbalize_text).

    Catches early truncation (< 0.4) and babbling (see is_too_long). expected_cps ~14 for spoken English.
    """
    return float(actual_duration_s / estimated_duration_s(text, lang, expected_cps))


def is_too_long(actual_duration_s: float, text: str, lang: str = "en", expected_cps: float = 14.0) -> bool:
    """Babbling: more than TOO_LONG_RATIO x the estimate *and* at least TOO_LONG_MIN_EXCESS_S seconds over it."""
    estimate = estimated_duration_s(text, lang, expected_cps)
    return bool(actual_duration_s > TOO_LONG_RATIO * estimate and actual_duration_s - estimate >= TOO_LONG_MIN_EXCESS_S)


def check_audio_sanity(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = SPEC.output_sample_rate,
) -> Dict[str, Any]:
    """Perform audio health and sanity verification on a waveform.

    Args:
        audio: 1D numpy array or torch tensor of audio samples.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Dict of metrics: rms, peak, duration_s, hf_ratio, is_finite, is_silent, is_clipped.
    """
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy().squeeze()
    else:
        audio_np = np.asarray(audio).squeeze()

    duration_s = float(len(audio_np) / sample_rate)
    is_finite = bool(np.isfinite(audio_np).all())

    if not is_finite or len(audio_np) == 0:
        return {
            "duration_s": duration_s,
            "rms": 0.0,
            "peak": 0.0,
            "hf_ratio": 0.0,
            "is_finite": is_finite,
            "is_silent": True,
            "is_clipped": False,
        }

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    peak = float(np.max(np.abs(audio_np)))
    is_silent = bool(rms < 0.015)
    is_clipped = bool(peak > 1.02)

    # High frequency energy (> 4kHz)
    if len(audio_np) > 128:
        fft = np.abs(np.fft.rfft(audio_np))
        freqs = np.fft.rfftfreq(len(audio_np), 1.0 / sample_rate)
        hf_energy = np.sum(fft[freqs >= 4000] ** 2)
        total_energy = np.sum(fft ** 2) + 1e-9
        hf_ratio = float(hf_energy / total_energy)
    else:
        hf_ratio = 0.0

    return {
        "duration_s": duration_s,
        "rms": rms,
        "peak": peak,
        "hf_ratio": hf_ratio,
        "is_finite": is_finite,
        "is_silent": is_silent,
        "is_clipped": is_clipped,
    }
