"""Evaluation harness for Open-XTTS-Trainer.

Runs fixed, multi-seed ground truth evaluations against base XTTS or fine-tuned checkpoints:
- Measures WER and CER using the official Whisper text normalizer.
- Measures SECS (Speaker Embedding Cosine Similarity) with ECAPA2, independent of XTTS's own speaker encoder.
- Measures predicted naturalness with UTMOS (relative to baseline only).
- Measures duration ratio and checks audio health (clipping, RMS, NaNs).
- Tracks failure counts (empty, babbling: > 2.2x and >= 1 s over the expected duration, NaNs).
- Outputs reproducible JSON reports with mean, standard deviation, per-sentence metrics, and manifest hashes.
"""

import os
import json
import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import soundfile as sf

from eval.dataset import DEFAULT_EVAL_SET_PATH, load_eval_set
from xtts_trainer.asr import transcribe as asr_transcribe, transcribe_batch as asr_transcribe_batch

_UNPATCHED_ASR_TRANSCRIBE = asr_transcribe
from xtts_trainer.spec import SPEC
from eval.metrics import (
    normalize_text,
    compute_wer,
    compute_cer,
    compute_duration_ratio,
    is_too_long,
    check_audio_sanity,
    SpeakerSimilarity,
    UTMOSPredictor,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINGERPRINT_DIRS = ("xtts_trainer", "eval", "scripts")
FINGERPRINT_FILES = ("pyproject.toml", "artifacts.lock.json")


def get_code_fingerprint(repo_root: str = REPO_ROOT) -> Dict[str, Any]:
    """SHA-256 over every source file's path and content, so results trace to exact code without git.

    Covers *.py under xtts_trainer/, eval/, scripts/, the JSON vendored inside xtts_trainer/ (the XTTS-v2
    config), pyproject.toml and artifacts.lock.json. Data files are not code: the eval set and reference
    voices are recorded by their own hashes, and the reference lock is generated during a run (Colab run 2
    fingerprinted 48 files at setup and 49 after step 6 built the lock). Two runs with the same fingerprint
    ran the same code.
    """
    import hashlib

    entries = []
    for top in FINGERPRINT_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(repo_root, top)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in filenames:
                if name.endswith(".py") or (top == "xtts_trainer" and name.endswith(".json")):
                    entries.append(os.path.relpath(os.path.join(dirpath, name), repo_root))
    entries.extend(f for f in FINGERPRINT_FILES if os.path.exists(os.path.join(repo_root, f)))
    digest = hashlib.sha256()
    for rel in sorted(p.replace(os.sep, "/") for p in entries):
        with open(os.path.join(repo_root, rel), "rb") as f:
            digest.update(rel.encode() + b"\0" + hashlib.sha256(f.read()).hexdigest().encode() + b"\n")
    return {"sha256": digest.hexdigest(), "n_files": len(entries)}


def get_git_info(repo_dir: str = ".") -> Dict[str, Any]:
    """Current commit SHA, and whether tracked files differ from it.

    Untracked files don't count: a run writes outputs (datasets, checkpoints, manifests) into the checkout, and
    those don't change the code. get_code_fingerprint covers the source files either way.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"  # not a git checkout (e.g. an uploaded zip); recorded honestly in the manifest

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir, text=True
        ).strip()
        is_dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        is_dirty = None  # unknown, not "clean"

    return {"commit_sha": sha, "is_dirty": is_dirty}


def aggregate_metric(values: List[float]) -> Dict[str, float]:
    """Mean, sample std, 95% CI half-width, min and max over finite values; n_nan counts the excluded ones."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = len(finite)
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(np.mean(finite)) if n else float("nan"),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0,
        "min": float(np.min(finite)) if n else float("nan"),
        "max": float(np.max(finite)) if n else float("nan"),
        "n": n,
        "n_nan": int(len(arr) - n),
    }


METRICS = ("wer", "cer", "secs", "utmos", "duration_ratio")
# Version of the metric definitions; manifests scored under different versions must not be compared.
# 2: duration on verbalized text, absolute-excess too_long, headline SECS over clips >= 2 s (Colab run 2).
METRICS_VERSION = 2
# ECAPA2 on under a second of speech is not a reliable speaker embedding (run 2: SECS 0.33 on 0.7-1.0 s
# one-word clips vs 0.53 overall), so the headline SECS only counts clips at least this long; secs_all keeps all.
HEADLINE_SECS_MIN_S = 2.0


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-utterance results: every metric, headline SECS over clips >= HEADLINE_SECS_MIN_S,
    secs_all over every clip, and failure counts."""
    summary = {key: aggregate_metric([r[key] for r in results]) for key in METRICS}
    summary["secs_all"] = summary["secs"]
    summary["secs"] = aggregate_metric([r["secs"] for r in results if r["duration_s"] >= HEADLINE_SECS_MIN_S])
    summary["failure_counts"] = {
        "empty": sum(r["failures"]["is_empty"] for r in results),
        "too_long": sum(r["failures"]["is_too_long"] for r in results),
        "nan": sum(r["failures"]["is_nan"] for r in results),
    }
    return summary


def summarize_by_category(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(r.get("category") or "_all", []).append(r)
    return {cat: summarize_results(rs) for cat, rs in groups.items()}


def select_subset(sentences: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Deterministic subset of n sentences taken round-robin across categories.

    Categories are visited in order of first appearance and each keeps its own order, so every category
    contributes before any contributes twice. Items without a category form one group.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in sentences:
        groups.setdefault(item.get("category", "_all"), []).append(item)
    picked: List[Dict[str, Any]] = []
    depth = 0
    while len(picked) < min(n, len(sentences)):
        for items in groups.values():
            if depth < len(items) and len(picked) < n:
                picked.append(items[depth])
        depth += 1
    return picked


class EvaluationHarness:
    """Rigorous evaluation runner for XTTS models."""

    def __init__(
        self,
        eval_set_path: str = DEFAULT_EVAL_SET_PATH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        asr_model_name: str = "openai/whisper-large-v3",
        asr_device: Optional[str] = None,
        speaker_similarity: Optional[SpeakerSimilarity] = None,
        mos_predictor: Optional[UTMOSPredictor] = None,
    ):
        """Initialize the evaluation harness with the frozen evaluation set.

        Args:
            eval_set_path: Path to frozen eval_set.json.
            device: Computation device ('cuda', 'mps', or 'cpu').
            asr_model_name: HuggingFace model identifier for Whisper ASR.
        """
        self.device = device
        self.eval_set_path = eval_set_path
        self.eval_dataset = load_eval_set(eval_set_path)
        self.eval_set_sha256 = self.eval_dataset.get("sha256", "")
        self.asr_model_name = asr_model_name
        self.asr_device = asr_device or device
        self.speaker_similarity = speaker_similarity or SpeakerSimilarity(device=device)
        self.mos_predictor = mos_predictor or UTMOSPredictor(device=device)

    def transcribe(self, audio: np.ndarray, sample_rate: int = SPEC.output_sample_rate, lang: str = "en") -> str:
        """Transcribe audio using Whisper with explicit sample rate.

        Args:
            audio: 1D numpy array of audio samples.
            sample_rate: Sampling rate of audio.
            lang: Expected language for transcription.

        Returns:
            Raw transcription string from ASR.
        """
        return asr_transcribe(audio, sample_rate, lang, self.asr_model_name, device=self.asr_device)

    def transcribe_batch(
        self,
        audios: List[np.ndarray],
        sample_rate: int = SPEC.output_sample_rate,
        lang: str = "en",
        batch_size: int = 16,
    ) -> List[str]:
        """Transcribe a list of audio arrays in batches using Whisper.

        Args:
            audios: List of 1D numpy arrays of audio samples.
            sample_rate: Sampling rate of audio.
            lang: Expected language for transcription.
            batch_size: Batch size for GPU inference.

        Returns:
            List of raw transcription strings from ASR.
        """
        if asr_transcribe is not _UNPATCHED_ASR_TRANSCRIBE:
            return [self.transcribe(a, sample_rate=sample_rate, lang=lang) for a in audios]
        return asr_transcribe_batch(
            audios,
            sample_rate,
            lang,
            self.asr_model_name,
            device=self.asr_device,
            batch_size=batch_size,
        )

    def evaluate_utterance(
        self,
        model: Any,
        text: str,
        lang: str,
        cond_latents: torch.Tensor,
        speaker_emb: torch.Tensor,
        ref_secs_emb: torch.Tensor,
        seed: int = 42,
        temperature: float = 0.75,
    ) -> Dict[str, Any]:
        """Synthesize and evaluate a single utterance across all metrics.

        Args:
            model: Initialized Xtts model.
            text: Text to synthesize.
            lang: Language code.
            cond_latents: Conditioning latents tensor.
            speaker_emb: Reference speaker embedding tensor (XTTS conditioning input).
            ref_secs_emb: ECAPA2 embedding of the reference clip, used only for SECS.
            seed: Random seed for synthesis.
            temperature: Sampling temperature.

        Returns:
            Dict containing per-utterance metric scores and health checks.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = time.time()
        with torch.inference_mode():
            res = model.inference(
                text=text,
                language=lang,
                gpt_cond_latent=cond_latents,
                speaker_embedding=speaker_emb,
                temperature=temperature,
            )
        elapsed_s = time.time() - t0

        audio = np.asarray(res["wav"]).squeeze()
        health = check_audio_sanity(audio, sample_rate=SPEC.output_sample_rate)

        # Transcribe & compute intelligibility
        asr_text = self.transcribe(audio, sample_rate=SPEC.output_sample_rate, lang=lang)
        wer = compute_wer(text, asr_text, lang=lang)
        cer = compute_cer(text, asr_text, lang=lang)

        # Speaker similarity (ECAPA2) and predicted naturalness (UTMOS); skipped for non-finite audio
        if health["is_finite"] and health["duration_s"] > 0:
            secs = self.speaker_similarity.similarity(ref_secs_emb, audio, SPEC.output_sample_rate)
            utmos = self.mos_predictor.score(audio, SPEC.output_sample_rate)
        else:
            secs, utmos = float("nan"), float("nan")

        # Duration ratio
        duration_ratio = compute_duration_ratio(health["duration_s"], text, lang=lang)

        # Failure flags
        is_empty = bool(health["duration_s"] < 0.3 or health["is_silent"])
        too_long = is_too_long(health["duration_s"], text, lang=lang)
        is_nan = bool(not health["is_finite"])

        return {
            "text": text,
            "lang": lang,
            "seed": seed,
            "asr_text": asr_text,
            "wer": wer,
            "cer": cer,
            "secs": secs,
            "utmos": utmos,
            "duration_s": health["duration_s"],
            "duration_ratio": duration_ratio,
            "rms": health["rms"],
            "peak": health["peak"],
            "hf_ratio": health["hf_ratio"],
            "synthesis_time_s": elapsed_s,
            "failures": {
                "is_empty": is_empty,
                "is_too_long": too_long,
                "is_nan": is_nan,
            },
        }

    def run_benchmark(
        self,
        model: Any,
        reference_audio_path: str,
        seeds: List[int] = (42, 123, 999),
        subset_size: Optional[int] = None,
        language: str = "en",
        output_dir: Optional[str] = None,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Run full evaluation suite across fixed seeds and aggregate statistics.

        Args:
            model: Initialized Xtts model.
            reference_audio_path: Path to reference clean audio clip.
            seeds: List of seeds to evaluate for error-bar calculation.
            subset_size: Optional number of sentences, taken evenly across categories (select_subset).
            language: Language to benchmark ('en', 'es', etc.).
            output_dir: Optional directory to save synthesized test waveforms.
            batch_size: Batch size for Whisper ASR inference.

        Returns:
            Dict containing aggregated metrics (mean, std, confidence intervals) and individual runs.
        """
        assert os.path.exists(reference_audio_path), f"Reference audio not found: {reference_audio_path}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Extract conditioning latents from reference audio
        with torch.inference_mode():
            cond_latents, speaker_emb = model.get_conditioning_latents(audio_path=[reference_audio_path])

        # ECAPA2 embedding of the reference clip, for SECS only
        ref_audio, ref_sr = sf.read(reference_audio_path, dtype="float32", always_2d=True)
        ref_secs_emb = self.speaker_similarity.embed(ref_audio.mean(axis=1), ref_sr)

        if language == "en":
            sentences = self.eval_dataset["english_sentences"]
        else:
            sentences = self.eval_dataset["target_languages"].get(language, [])
        if not sentences:
            raise ValueError(f"Eval set has no sentences for language '{language}'.")

        if subset_size is not None and subset_size > 0:
            sentences = select_subset(sentences, subset_size)

        tasks = []
        for seed in seeds:
            for item in sentences:
                torch.manual_seed(seed)
                np.random.seed(seed)

                t0 = time.time()
                with torch.inference_mode():
                    res = model.inference(
                        text=item["text"],
                        language=language,
                        gpt_cond_latent=cond_latents,
                        speaker_embedding=speaker_emb,
                        temperature=0.75,
                    )
                elapsed_s = time.time() - t0
                audio = np.asarray(res["wav"]).squeeze()
                health = check_audio_sanity(audio, sample_rate=SPEC.output_sample_rate)
                tasks.append({
                    "id": item["id"],
                    "category": item.get("category"),
                    "text": item["text"],
                    "seed": seed,
                    "audio": audio,
                    "health": health,
                    "synthesis_time_s": elapsed_s,
                })

        # Batched Whisper transcription
        audios = [t["audio"] for t in tasks]
        asr_texts = self.transcribe_batch(
            audios,
            sample_rate=SPEC.output_sample_rate,
            lang=language,
            batch_size=batch_size,
        )

        results_by_seed: Dict[int, List[Dict[str, Any]]] = {s: [] for s in seeds}
        for task, asr_text in zip(tasks, asr_texts):
            health = task["health"]
            text = task["text"]
            audio = task["audio"]

            wer = compute_wer(text, asr_text, lang=language)
            cer = compute_cer(text, asr_text, lang=language)

            if health["is_finite"] and health["duration_s"] > 0:
                secs = self.speaker_similarity.similarity(ref_secs_emb, audio, SPEC.output_sample_rate)
                utmos = self.mos_predictor.score(audio, SPEC.output_sample_rate)
            else:
                secs, utmos = float("nan"), float("nan")

            duration_ratio = compute_duration_ratio(health["duration_s"], text, lang=language)

            is_empty = bool(health["duration_s"] < 0.3 or health["is_silent"])
            too_long = is_too_long(health["duration_s"], text, lang=language)
            is_nan = bool(not health["is_finite"])

            res = {
                "id": task["id"],
                "category": task["category"],
                "text": text,
                "lang": language,
                "seed": task["seed"],
                "asr_text": asr_text,
                "wer": wer,
                "cer": cer,
                "secs": secs,
                "utmos": utmos,
                "duration_s": health["duration_s"],
                "duration_ratio": duration_ratio,
                "rms": health["rms"],
                "peak": health["peak"],
                "hf_ratio": health["hf_ratio"],
                "synthesis_time_s": task["synthesis_time_s"],
                "failures": {
                    "is_empty": is_empty,
                    "is_too_long": too_long,
                    "is_nan": is_nan,
                },
            }
            results_by_seed[task["seed"]].append(res)

        all_results = [r for rs in results_by_seed.values() for r in rs]
        summary = {
            "num_sentences": len(sentences),
            "num_seeds": len(seeds),
            "total_evaluations": len(all_results),
            **summarize_results(all_results),
            "by_category": summarize_by_category(all_results),
        }

        git_info = get_git_info()
        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "git": git_info,
            "eval_set_sha256": self.eval_set_sha256,
            "reference_audio": reference_audio_path,
            "asr_model": self.asr_model_name,
            "code_fingerprint": get_code_fingerprint(),
            "metrics_version": METRICS_VERSION,
            "language": language,
            "seeds": list(seeds),
            "summary": summary,
            "per_seed_results": results_by_seed,
        }

        return manifest
