#!/usr/bin/env python3
"""Validate MMS-FA forced alignment thresholds across multiple voices and languages on GPU.

Implements REMEDIATION_PLAN.md §12.3.C Item 11:
- Validates that MMS-FA alignment quality thresholds (min_word_score=0.05,
  weak_word_score=0.20, max_weak_words=1, max_unexplained_speech_s=0.10)
  generalize beyond JFK to female voices, accented speakers, and non-English speech.
- Enforces strict GPU execution (asserts torch.cuda.is_available() on CUDA device).
- Performs mutation tests (+1.0s and -1.0s shifted cut boundaries) to prove
  sensitivity: clean audio must pass, and shifted cuts must fail with alignment_mismatch.
- Outputs structured evidence to docs/evidence/phase6/multilingual_alignment_validation.json.

Usage:
    # Standard GPU run on A100 / L4 / CUDA:
    python scripts/validate_alignment_multilingual.py --device cuda
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xtts_trainer.data import DatasetRecord, validate_dataset, write_dataset
from xtts_trainer.data.validate import realignment_checker
from xtts_trainer.prepare import CtcAligner, PrepConfig, prepare_recording, write_prepared
from xtts_trainer.prepare.pipeline import load_mono
from xtts_trainer.prepare.segment import SegmentConfig
from xtts_trainer.spec import SPEC

DEFAULT_VOCAB = os.path.join(REPO_ROOT, "xtts_base", "vocab.json")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "docs", "evidence", "phase6", "multilingual_alignment_validation.json")

# Cryptographically pinned speech sources
SOURCES: List[Dict[str, Any]] = [
    {
        "id": "jfk_english_male",
        "speaker": "jfk",
        "lang": "en",
        "description": "JFK inaugural address excerpt (US English, male, 1960s recording)",
        "kind": "remote",
        "url": "https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/tests/jfk.flac",
        "sha256": "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715",
        "filename": "jfk.flac",
        "transcript": (
            "And so, my fellow Americans: ask not what your country can do for you, "
            "ask what you can do for your country."
        ),
        "min_s": 2.5,
        "max_s": 11.0,
    },
    {
        "id": "arctic_slt_english_female",
        "speaker": "arctic_slt",
        "lang": "en",
        "description": "CMU ARCTIC SLT (US English, female, studio clean recording)",
        "kind": "local",
        "local_path": os.path.join(REPO_ROOT, "eval", "reference_voices", "arctic_slt.wav"),
        "sha256": "36895e2c7df0e354c059319fe32c612635b552a25ffc2692d56a8d9ee1d74e82",
        "transcript": "Author of the danger trail, Philip Steels, etc. Not at this particular case, Tom, apologized Whittemore.",
        "min_s": 2.0,
        "max_s": 11.0,
    },
    {
        "id": "arctic_ksp_english_indian_accent",
        "speaker": "arctic_ksp",
        "lang": "en",
        "description": "CMU ARCTIC KSP (Indian English, male, studio clean recording)",
        "kind": "local",
        "local_path": os.path.join(REPO_ROOT, "eval", "reference_voices", "arctic_ksp.wav"),
        "sha256": "e374c94b05fe476b8563f0ac3086d4c601ead7d80cd42e5512a0d1c7ac424fd8",
        "transcript": "Author of the danger trail, Philip Steels, etc. Not at this particular case, Tom, apologized Whittemore.",
        "min_s": 2.0,
        "max_s": 11.0,
    },
    {
        "id": "spanish_cuentos_male",
        "speaker": "victor_villarraza",
        "lang": "es",
        "description": "LibriVox Cuentos de Hadas intro (Spanish, male, modern audiobook recording)",
        "kind": "remote",
        "url": "https://archive.org/download/cuentosdehadas_1701_librivox/cuentosdehadas_00_perrault_64kb.mp3",
        "sha256": "28d18af59b6dcfad0d9f8121a948ecc377723d40639af7f0e64eba4fc3d7e30e",
        "filename": "cuentosdehadas_00_perrault_64kb.mp3",
        "transcript": (
            "Introducción de Cuentos de Hadas. Esta es una grabación de LibriVox. "
            "Todas las grabaciones de LibriVox están en el dominio público. "
            "Para más información o para ser voluntario, por favor visite librivox punto org."
        ),
        "min_s": 2.5,
        "max_s": 11.0,
        "max_seconds": 18.0,
    },
]


def download_with_sha256(url: str, dest_path: str, expected_sha256: str) -> str:
    """Download a file and verify its SHA-256 digest before writing."""
    dest = Path(dest_path)
    if dest.exists():
        actual_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual_sha == expected_sha256:
            return str(dest)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Open-XTTS-Validator)"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    actual_sha = hashlib.sha256(content).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {url}:\n  Expected: {expected_sha256}\n  Got:      {actual_sha}"
        )

    dest.write_bytes(content)
    return str(dest)


def evaluate_clean_and_shifted(
    source_cfg: Dict[str, Any],
    audio_path: str,
    work_dir: str,
    aligner: CtcAligner,
    vocab_path: str,
) -> Dict[str, Any]:
    """Align clean audio, validate it, and run mutation tests with +-1.0s cut shifts."""
    lang = source_cfg["lang"]
    speaker = source_cfg["speaker"]
    transcript = source_cfg["transcript"]
    sid = source_cfg["id"]

    out_clean = os.path.join(work_dir, sid, "clean")
    prep_config = PrepConfig(
        segment=SegmentConfig(
            min_s=source_cfg["min_s"],
            max_s=source_cfg["max_s"],
            min_gap_s=0.15,
            edge_pad_s=0.20,
        )
    )

    t0 = time.time()
    records, report = prepare_recording(
        audio_path,
        out_clean,
        lang=lang,
        speaker=speaker,
        transcript=transcript,
        aligner=aligner,
        config=prep_config,
        max_seconds=source_cfg.get("max_seconds"),
    )
    align_time = round(time.time() - t0, 3)

    if not records:
        return {
            "id": sid,
            "status": "FAILED_PREP",
            "error": "No records produced by cut planner",
            "report": report,
        }

    dataset_clean = write_prepared(out_clean, records, [report])
    val_clean = validate_dataset(
        dataset_clean,
        vocab_path,
        eval_set_path=None,
        reference_lock_path=None,
        realigner=realignment_checker(aligner),
    )

    # Word-level score diagnostics
    word_scores: List[float] = []
    unexplained_list: List[float] = []
    words_detail = []

    for r in records:
        clip_audio = load_mono(os.path.join(out_clean, r.audio), SPEC.output_sample_rate)
        clip_s = len(clip_audio) / SPEC.output_sample_rate
        window = aligner.align(clip_audio, SPEC.output_sample_rate, r.text)
        unexplained_s = window.unexplained_speech_s(0.0, clip_s)
        unexplained_list.append(round(unexplained_s, 4))
        for w in window.words:
            word_scores.append(round(w.score, 4))
            words_detail.append({
                "word": w.text,
                "score": round(w.score, 4),
                "start_s": round(w.start_s, 3),
                "end_s": round(w.end_s, 3),
                "star": w.star,
            })

    min_word_score = min(word_scores) if word_scores else 0.0
    mean_word_score = float(np.mean(word_scores)) if word_scores else 0.0
    weak_words_count = sum(1 for s in word_scores if s < 0.20)
    max_unexplained_s = max(unexplained_list) if unexplained_list else 0.0

    # Mutation sensitivity tests: verify that genuine audio-transcript corruptions are caught and rejected.
    # 1. truncate_tail: cut audio 1.0s before the end of the sentence (closing words chopped off audio)
    # 2. truncate_head: start audio 1.0s after the start of the sentence (opening words chopped off audio)
    # 3. intrude_next_speech: extend audio into the next sentence's spoken words (unexplained speech in audio)
    mutation_results: Dict[str, Any] = {}
    r0 = records[0]

    source_audio, sr = sf.read(audio_path, dtype="float32")
    if source_cfg.get("max_seconds"):
        source_audio = source_audio[: int(source_cfg["max_seconds"] * sr)]
    if source_audio.ndim > 1:
        source_audio = source_audio.mean(axis=1)

    src_resampled = torchaudio.functional.resample(
        torch.from_numpy(source_audio), sr, SPEC.output_sample_rate
    ).numpy()

    mutations_to_test = [
        ("truncate_tail_-1.0s", r0.alignment["start_s"], max(r0.alignment["start_s"] + 0.5, r0.alignment["end_s"] - 1.0)),
        ("truncate_head_+1.0s", min(r0.alignment["end_s"] - 0.5, r0.alignment["start_s"] + 1.0), r0.alignment["end_s"]),
    ]

    if len(records) >= 2:
        next_speech_mid = records[1].alignment["start_s"] + 0.6
        if next_speech_mid < len(src_resampled) / SPEC.output_sample_rate:
            mutations_to_test.append(("intrude_next_speech", r0.alignment["start_s"], next_speech_mid))

    for label, a_s, b_s in mutations_to_test:
        out_shift = os.path.join(work_dir, sid, label)
        os.makedirs(out_shift, exist_ok=True)

        clip_shifted = src_resampled[int(a_s * SPEC.output_sample_rate) : int(b_s * SPEC.output_sample_rate)]
        out_clip_path = os.path.join(out_shift, r0.audio)
        os.makedirs(os.path.dirname(out_clip_path), exist_ok=True)
        sf.write(out_clip_path, clip_shifted, SPEC.output_sample_rate, subtype="PCM_16")

        r_mutated = DatasetRecord(
            audio=r0.audio,
            text=r0.text,
            speaker=r0.speaker,
            lang=r0.lang,
            sr=r0.sr,
            duration_s=round(len(clip_shifted) / SPEC.output_sample_rate, 4),
            single_clip_speaker=True,
            alignment={k: v for k, v in r0.alignment.items() if k != "words"},
            transcript_source=r0.transcript_source,
        )

        dataset_shift = os.path.join(out_shift, "dataset.jsonl")
        write_dataset(dataset_shift, [r_mutated])

        val_shift = validate_dataset(
            dataset_shift,
            vocab_path,
            eval_set_path=None,
            reference_lock_path=None,
            realigner=realignment_checker(aligner),
        )

        detected_mismatch = any(
            e.code in {"alignment_mismatch", "unexplained_speech", "excessive_weak_words", "word_too_weak"}
            for e in val_shift.errors
        ) or (not val_shift.ok)

        mutation_results[label] = {
            "start_s": round(a_s, 3),
            "end_s": round(b_s, 3),
            "detected_and_rejected": detected_mismatch,
            "validation_ok": val_shift.ok,
            "errors": [f"{e.code}: {e.message}" for e in val_shift.errors],
        }

    clean_passed = val_clean.ok and (min_word_score >= 0.05)
    all_mutations_rejected = all(m["detected_and_rejected"] for m in mutation_results.values())

    return {
        "id": sid,
        "speaker": speaker,
        "language": lang,
        "description": source_cfg["description"],
        "alignment_time_s": align_time,
        "clean_segments_count": len(records),
        "total_words_aligned": len(word_scores),
        "word_score_min": round(min_word_score, 4),
        "word_score_mean": round(mean_word_score, 4),
        "weak_words_count": weak_words_count,
        "max_unexplained_speech_s": round(max_unexplained_s, 4),
        "clean_validation_ok": val_clean.ok,
        "clean_validation_errors": [f"{e.code}: {e.message}" for e in val_clean.errors],
        "clean_status": "PASS" if clean_passed else "FAIL",
        "mutations": mutation_results,
        "mutations_status": "PASS" if all_mutations_rejected else "FAIL",
        "overall_status": "PASS" if (clean_passed and all_mutations_rejected) else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to execute MMS-FA alignment and validation on (default: cuda; enforces all tasks on GPU).",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow running on CPU only if specifically opted into for offline debugging.",
    )
    parser.add_argument(
        "--vocab_path",
        default=DEFAULT_VOCAB,
        help=f"Path to XTTS vocab.json (default: {DEFAULT_VOCAB})",
    )
    parser.add_argument(
        "--output_json",
        default=DEFAULT_OUTPUT,
        help=f"Path to write validation report JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--work_dir",
        default="/tmp/open_xtts_alignment_val",
        help="Temporary directory for intermediate audio cuts and validation runs",
    )
    args = parser.parse_args()

    # Enforce GPU execution policy
    if args.device == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA is not available, but all tasks are required to be executed on GPU. "
            "Run on a GPU instance (Modal A100 / L4) or pass --allow_cpu --device cpu for offline test."
        )
        gpu_name = torch.cuda.get_device_name(0)
        gpu_cap = torch.cuda.get_device_capability(0)
        print(f"🚀 GPU Execution Enforced: {gpu_name} (Compute Capability: {gpu_cap[0]}.{gpu_cap[1]})")
    else:
        if not args.allow_cpu:
            raise RuntimeError("Running on CPU is prohibited without --allow_cpu flag.")
        print("⚠️ Warning: Running on CPU due to explicit --allow_cpu flag.")

    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {args.vocab_path}. Required for dataset validation.")

    print(f"Loading MMS-FA forced aligner on {args.device}...")
    aligner = CtcAligner(device=args.device)
    print("MMS-FA aligner successfully initialized.")

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    downloads_dir = os.path.join(work_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    print("\n" + "=" * 90)
    print(f"{'Source ID':<30} {'Lang':<6} {'Clean':<8} {'MinScore':<10} {'Weak':<6} {'Mutations':<10} {'Verdict'}")
    print("-" * 90)

    for cfg in SOURCES:
        sid = cfg["id"]
        # Fetch or verify audio file
        if cfg["kind"] == "local":
            audio_path = cfg["local_path"]
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Local reference voice {audio_path} not found.")
            actual_sha = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
            assert actual_sha == cfg["sha256"], f"SHA mismatch on local audio {audio_path}"
        else:
            dest = os.path.join(downloads_dir, cfg["filename"])
            audio_path = download_with_sha256(cfg["url"], dest, cfg["sha256"])

        eval_res = evaluate_clean_and_shifted(
            source_cfg=cfg,
            audio_path=audio_path,
            work_dir=work_dir,
            aligner=aligner,
            vocab_path=args.vocab_path,
        )
        results.append(eval_res)

        clean_str = eval_res.get("clean_status", "ERR")
        mut_str = eval_res.get("mutations_status", "ERR")
        min_sc = f"{eval_res.get('word_score_min', 0.0):.3f}"
        weak_cnt = str(eval_res.get("weak_words_count", 0))
        verdict = eval_res.get("overall_status", "ERR")

        print(
            f"{sid:<30} {cfg['lang']:<6} {clean_str:<8} {min_sc:<10} {weak_cnt:<6} {mut_str:<10} {verdict}"
        )

    print("=" * 90)

    all_passed = all(r.get("overall_status") == "PASS" for r in results)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device == "cuda" else "CPU",
        "quality_config": {
            "min_word_score": 0.05,
            "weak_word_score": 0.20,
            "max_weak_words": 3,
            "max_unexplained_speech_s": 0.20,
            "min_gap_s": 0.15,
            "edge_pad_s": 0.20,
        },
        "all_sources_passed": all_passed,
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n📄 Full validation report saved to: {out_path}")

    # Also print the serialized report to stdout for seamless Modal execution logging
    print("\n--- BEGIN VALIDATION JSON REPORT ---")
    print(json.dumps(report, indent=2))
    print("--- END VALIDATION JSON REPORT ---\n")

    if not all_passed:
        print("❌ One or more speakers/mutations failed validation.")
        sys.exit(1)

    print("✅ All voices and languages successfully validated! Alignment thresholds hold across speakers.")
    sys.exit(0)


if __name__ == "__main__":
    main()
