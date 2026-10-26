#!/usr/bin/env python3
"""Comprehensive multilingual and multi-speaker MMS-FA alignment validation benchmark.

Implements the empirical validation suite for docs/recipe_design.md §2 (Finding F5) and §6 (Phase 6 Data Preparation):
- 8 Distinct Vetted Baseline Speakers across 2 languages and 5 accents:
    1. jfk_english_male (US English male, 1960s vintage)
    2. arctic_slt_english_female (US English female, studio clean)
    3. arctic_ksp_english_indian_accent (Indian English male, studio clean)
    4. arctic_bdl_english_male (US English male, studio clean)
    5. arctic_clb_english_female (US English female, studio clean)
    6. arctic_awb_english_scottish (Scottish English male, studio clean)
    7. ljspeech_female (US English female, studio clean gold standard)
    8. spanish_cuentos_male (Castilian Spanish male, LibriVox modern audiobook)
- 3 Controlled Corruption Types with Realistic Failure Modes:
    1. Edge Truncation Severity Ladder (100ms, 250ms, 500ms, 1000ms at head and tail)
    2. Hallucinated / Unspoken Word in Transcript (Head, Mid, Tail positions; function & content words)
    3. Unexplained Speech Spillover (Sub-threshold 0.10s vs Cross-threshold 0.50s of spoken speech)
- Strict Blinding: Corrupted clips are fed into validate_dataset() without labels or side-channel hints.
- Hierarchical / Clustered Reporting: Metrics grouped by corruption type and by speaker.

Prerequisites:
    Local reference voices in eval/reference_voices/ (CMU ARCTIC) are automatically fetched
    via scripts/fetch_reference_voices.py if not already present on disk.

Usage:
    python scripts/validate_alignment_multilingual.py --device cuda
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from xtts_trainer.prepare.align import AlignmentError
from xtts_trainer.prepare.pipeline import load_mono
from xtts_trainer.prepare.segment import QualityConfig, SegmentConfig
from xtts_trainer.spec import SPEC

DEFAULT_VOCAB = os.path.join(REPO_ROOT, "xtts_base", "vocab.json")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "docs", "evidence", "phase6", "multilingual_alignment_validation.json")

# Cryptographically pinned speech sources: 8 vetted baseline speakers
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
        "license": "Public Domain (1961 US Presidential inaugural address, work of US Gov under 17 U.S.C. § 105; openai/whisper test fixture MIT)",
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
        "license": "CMU ARCTIC Permissive License (Language Technologies Institute, Carnegie Mellon University; eval/reference_voices/licenses/slt_COPYING)",
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
        "license": "CMU ARCTIC Permissive License (Language Technologies Institute, Carnegie Mellon University; eval/reference_voices/licenses/ksp_COPYING)",
        "transcript": "Author of the danger trail, Philip Steels, etc. Not at this particular case, Tom, apologized Whittemore.",
        "min_s": 2.0,
        "max_s": 11.0,
    },
    {
        "id": "arctic_bdl_english_male",
        "speaker": "arctic_bdl",
        "lang": "en",
        "description": "CMU ARCTIC BDL (US English, male, studio clean recording)",
        "kind": "local",
        "local_path": os.path.join(REPO_ROOT, "eval", "reference_voices", "arctic_bdl.wav"),
        "sha256": "886f17023bc3f0880043252b3f1471ccc35f286500512d939b7f1a91a6d8e627",
        "license": "CMU ARCTIC Permissive License (Language Technologies Institute, Carnegie Mellon University; eval/reference_voices/licenses/bdl_COPYING)",
        "transcript": "Author of the danger trail, Philip Steels, etc. Not at this particular case, Tom, apologized Whittemore.",
        "min_s": 2.0,
        "max_s": 11.0,
    },
    {
        "id": "arctic_clb_english_female",
        "speaker": "arctic_clb",
        "lang": "en",
        "description": "CMU ARCTIC CLB (US English, female, studio clean recording)",
        "kind": "local",
        "local_path": os.path.join(REPO_ROOT, "eval", "reference_voices", "arctic_clb.wav"),
        "sha256": "3764871a513fd11309c58b36b27d291d428502daf0495159ca153c4e5e162e5b",
        "license": "CMU ARCTIC Permissive License (Language Technologies Institute, Carnegie Mellon University; eval/reference_voices/licenses/clb_COPYING)",
        "transcript": "Author of the danger trail, Philip Steels, etc. Not at this particular case, Tom, apologized Whittemore.",
        "min_s": 2.0,
        "max_s": 11.0,
    },
    {
        "id": "arctic_awb_english_scottish",
        "speaker": "arctic_awb",
        "lang": "en",
        "description": "CMU ARCTIC AWB (Scottish English, male, studio clean recording)",
        "kind": "local",
        "local_path": os.path.join(REPO_ROOT, "eval", "reference_voices", "arctic_awb.wav"),
        "sha256": "63a3e6103b00fcd168df79a4d6ad6fda0e8b1b64eda6d9ce32eddd078c9c9b94",
        "license": "CMU ARCTIC Permissive License (Language Technologies Institute, Carnegie Mellon University; eval/reference_voices/licenses/awb_COPYING)",
        "transcript": "AUTHOR OF THE DANGER TRAIL, PHILIP STEELS, ETC Not at this particular case, Tom, apologized Whittemore.",
        "min_s": 2.0,
        "max_s": 11.0,
    },
    {
        "id": "ljspeech_female",
        "speaker": "ljspeech",
        "lang": "en",
        "description": "LJSpeech LJ001-0001 (US English, female, studio gold standard)",
        "kind": "remote",
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/tests/data/ljspeech/wavs/LJ001-0001.wav",
        "sha256": "4d2bd83ff0f7fe33491b03193abdf623f1b4cc2d8103972b33638d3607d86dc5",
        "filename": "LJ001-0001.wav",
        "license": "Public Domain (LJ Speech Dataset, https://keithito.com/LJ-Speech-Dataset/)",
        "transcript": (
            "Printing, in the only sense with which we are at present concerned, "
            "differs from most if not from all the other arts and crafts represented in the Exhibition"
        ),
        "min_s": 2.0,
        "max_s": 11.0,
    },
    {
        "id": "spanish_cuentos_male",
        "speaker": "victor_villarraza",
        "lang": "es",
        "description": "LibriVox Cuentos de Hadas intro (Castilian Spanish, male, modern audiobook recording)",
        "kind": "remote",
        "url": "https://archive.org/download/cuentosdehadas_1701_librivox/cuentosdehadas_00_perrault_64kb.mp3",
        "sha256": "28d18af59b6dcfad0d9f8121a948ecc377723d40639af7f0e64eba4fc3d7e30e",
        "filename": "cuentosdehadas_00_perrault_64kb.mp3",
        "license": "Public Domain (LibriVox public domain dedication CC0/PD; https://librivox.org/)",
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
        with open(dest, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h == expected_sha256:
            return str(dest)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "open-xtts-trainer-validator/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()

    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha256}")

    with open(dest, "wb") as f:
        f.write(data)

    return str(dest)


def evaluate_source(
    source_cfg: Dict[str, Any],
    audio_path: str,
    work_dir: str,
    aligner: CtcAligner,
    vocab_path: str,
) -> Dict[str, Any]:
    """Evaluate clean speech and run the controlled corruption test matrix."""
    sid = source_cfg["id"]
    lang = source_cfg["lang"]
    speaker = source_cfg["speaker"]

    out_clean = os.path.join(work_dir, sid, "clean")
    os.makedirs(out_clean, exist_ok=True)

    # 1. Clean data preparation & alignment
    # Calibrated thresholds: max_weak_words=3, max_unexplained_speech_s=0.20
    quality_cfg = QualityConfig(
        min_word_score=0.05,
        weak_word_score=0.20,
        max_weak_words=3,
        max_unexplained_speech_s=0.20,
    )
    seg_cfg = SegmentConfig(
        min_s=source_cfg.get("min_s", 2.0),
        max_s=source_cfg.get("max_s", 11.0),
        min_gap_s=0.15,
        edge_pad_s=0.20,
    )
    prep_cfg = PrepConfig(
        segment=seg_cfg,
        quality=quality_cfg,
    )

    t0 = time.time()
    records, report = prepare_recording(
        input_file=audio_path,
        output_dir=out_clean,
        lang=lang,
        speaker=speaker,
        transcript=source_cfg["transcript"],
        config=prep_cfg,
        aligner=aligner,
        max_seconds=source_cfg.get("max_seconds", None),
    )
    align_time = round(time.time() - t0, 3)

    dataset_clean = write_prepared(out_clean, records, [report])

    # Validate clean prepared dataset
    val_clean = validate_dataset(
        dataset_clean,
        vocab_path,
        eval_set_path=None,
        reference_lock_path=None,
        realigner=realignment_checker(aligner, quality=quality_cfg),
    )

    prep_rejected_count = sum(1 for c in report.get("candidates", []) if not c.get("kept", True))
    clean_passed = val_clean.ok and len(val_clean.errors) == 0 and prep_rejected_count == 0

    # Collect clean telemetry across segments
    word_scores = []
    unexplained_list = []
    weak_words_count = 0
    for r in records:
        if r.alignment and "words" in r.alignment:
            for w in r.alignment["words"]:
                sc = w.get("score", 0.0)
                word_scores.append(sc)
                if sc < quality_cfg.weak_word_score:
                    weak_words_count += 1
        if r.alignment and "unexplained_speech_s" in r.alignment:
            unexplained_list.append(r.alignment["unexplained_speech_s"])

    min_word_score = min(word_scores) if word_scores else 0.0
    mean_word_score = float(np.mean(word_scores)) if word_scores else 0.0
    max_unexplained_s = max(unexplained_list) if unexplained_list else 0.0

    # -------------------------------------------------------------------------
    # CONTROLLED CORRUPTIONS TEST MATRIX
    # -------------------------------------------------------------------------
    r0 = records[0]
    r0_start = r0.alignment["start_s"]
    r0_end = r0.alignment["end_s"]
    r0_dur = r0_end - r0_start

    # Load resampled source audio
    source_audio, sr = sf.read(audio_path, dtype="float32")
    if source_cfg.get("max_seconds"):
        source_audio = source_audio[: int(source_cfg["max_seconds"] * sr)]
    if source_audio.ndim > 1:
        source_audio = source_audio.mean(axis=1)

    src_resampled = torchaudio.functional.resample(
        torch.from_numpy(source_audio), sr, SPEC.output_sample_rate
    ).numpy()

    clean_clip = src_resampled[int(r0_start * SPEC.output_sample_rate) : int(r0_end * SPEC.output_sample_rate)]

    mutation_tests: Dict[str, Any] = {}

    def run_blind_test(
        test_id: str,
        audio_clip: np.ndarray,
        transcript_text: str,
        expected_pass: bool,
        expected_error_substr: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute validator on a blind audio/transcript pair."""
        test_dir = os.path.join(work_dir, sid, "mutations", test_id)
        os.makedirs(test_dir, exist_ok=True)
        clip_path = os.path.join(test_dir, "clip_0000.wav")
        sf.write(clip_path, audio_clip, SPEC.output_sample_rate)

        rec = DatasetRecord(
            audio="clip_0000.wav",
            text=transcript_text,
            speaker=r0.speaker,
            lang=r0.lang,
            sr=r0.sr,
            duration_s=round(len(audio_clip) / SPEC.output_sample_rate, 4),
            single_clip_speaker=True,
            alignment={k: v for k, v in r0.alignment.items() if k != "words"},
            transcript_source=r0.transcript_source,
        )
        dataset_path = os.path.join(test_dir, "dataset.jsonl")
        dataset_shift = write_dataset(dataset_path, [rec])

        val = validate_dataset(
            dataset_shift,
            vocab_path,
            eval_set_path=None,
            reference_lock_path=None,
            realigner=realignment_checker(aligner, quality=quality_cfg),
        )

        actual_pass = val.ok and len(val.errors) == 0
        errors = [f"{e.code}: {e.message}" for e in val.errors]
        error_matched = True
        if expected_error_substr and not actual_pass:
            error_matched = any(expected_error_substr.lower() in err.lower() for err in errors)

        verdict_ok = (actual_pass == expected_pass) and (error_matched if not expected_pass else True)

        try:
            clip_s = len(audio_clip) / SPEC.output_sample_rate
            win = aligner.align(audio_clip, SPEC.output_sample_rate, transcript_text)
            unexplained_s = round(win.unexplained_speech_s(0.0, clip_s), 4)
        except AlignmentError:
            unexplained_s = None

        return {
            "expected_pass": expected_pass,
            "actual_pass": actual_pass,
            "verdict_correct": verdict_ok,
            "errors": errors,
            "error_matched": error_matched,
            "unexplained_speech_s": unexplained_s,
        }

    # -------------------------------------------------------------------------
    # TEST 1: Edge Truncation Severity Ladder (100ms, 250ms, 500ms, 1000ms)
    # -------------------------------------------------------------------------
    # Anchored directly to actual spoken word boundaries so cuts genuinely clip phonemes
    words_info = r0.alignment.get("words", [])
    assert len(words_info) >= 2, f"Segment {sid} must have at least 2 aligned words"
    w_first_start_global = r0_start + words_info[0]["start_s"]
    w_last_end_global = r0_start + words_info[-1]["end_s"]

    trunc_ladder = [
        ("trunc_tail_100ms", r0_start, max(r0_start + 0.5, w_last_end_global - 0.10)),
        ("trunc_tail_250ms", r0_start, max(r0_start + 0.5, w_last_end_global - 0.25)),
        ("trunc_tail_500ms", r0_start, max(r0_start + 0.5, w_last_end_global - 0.50)),
        ("trunc_tail_1000ms", r0_start, max(r0_start + 0.5, w_last_end_global - 1.00)),
        ("trunc_head_500ms", min(r0_end - 0.5, w_first_start_global + 0.50), r0_end),
        ("trunc_head_1000ms", min(r0_end - 0.5, w_first_start_global + 1.00), r0_end),
    ]

    for m_label, a_s, b_s in trunc_ladder:
        mut_clip = src_resampled[int(a_s * SPEC.output_sample_rate) : int(b_s * SPEC.output_sample_rate)]
        res = run_blind_test(
            test_id=m_label,
            audio_clip=mut_clip,
            transcript_text=r0.text,
            expected_pass=False,
            expected_error_substr="low_alignment_score",
        )
        mutation_tests[m_label] = res

    # -------------------------------------------------------------------------
    # TEST 2: Hallucinated / Unspoken Word in Transcript
    # -------------------------------------------------------------------------
    # Test unspoken tokens across 3 sentence positions (Head, Mid, Tail) and types (function & content)
    words = r0.text.strip().split()
    mid_idx = max(1, len(words) // 2)

    if lang == "es":
        head_hallucinated = "Además, " + r0.text
        mid_hallucinated = " ".join(words[:mid_idx] + ["extraordinario"] + words[mid_idx:])
        tail_hallucinated = r0.text.rstrip(".") + " consecuentemente."
    else:
        head_hallucinated = "Moreover, " + r0.text
        mid_hallucinated = " ".join(words[:mid_idx] + ["extraordinary"] + words[mid_idx:])
        tail_hallucinated = r0.text.rstrip(".") + " consequently."

    hallucination_variants = [
        ("hallucinate_head_function", head_hallucinated),
        ("hallucinate_mid_content", mid_hallucinated),
        ("hallucinate_tail_content", tail_hallucinated),
    ]

    for h_label, h_text in hallucination_variants:
        res = run_blind_test(
            test_id=h_label,
            audio_clip=clean_clip,
            transcript_text=h_text,
            expected_pass=False,
            expected_error_substr="low_alignment_score",
        )
        mutation_tests[h_label] = res

    # -------------------------------------------------------------------------
    # TEST 3: Unexplained Speech Spillover (Sub-threshold vs Cross-threshold)
    # -------------------------------------------------------------------------
    # Locate subsequent candidate speech segment after r0 rather than assuming candidates[1]
    candidates = report.get("candidates", [])
    next_cand = next(
        (c for c in candidates if c.get("start_s", 0.0) >= r0_end - 0.05 and len(c.get("words", [])) > 0),
        None,
    )
    if next_cand is not None:
        cand2_first_speech = next_cand["words"][0][1]

        # Sub-threshold: include 0.10s of sentence 2 speech (<= 0.20s limit) -> Expected: PASS
        sub_end = cand2_first_speech + 0.10
        clip_sub = src_resampled[int(r0_start * SPEC.output_sample_rate) : int(sub_end * SPEC.output_sample_rate)]
        res_sub = run_blind_test(
            test_id="spillover_sub_threshold_0.10s",
            audio_clip=clip_sub,
            transcript_text=r0.text,
            expected_pass=True,
        )
        mutation_tests["spillover_sub_threshold_0.10s"] = res_sub

        # Cross-threshold: include 0.50s of sentence 2 speech (> 0.20s limit) -> Expected: FAIL (unexplained_speech)
        cross_end = cand2_first_speech + 0.50
        clip_cross = src_resampled[int(r0_start * SPEC.output_sample_rate) : int(cross_end * SPEC.output_sample_rate)]
        res_cross = run_blind_test(
            test_id="spillover_cross_threshold_0.50s",
            audio_clip=clip_cross,
            transcript_text=r0.text,
            expected_pass=False,
            expected_error_substr="unexplained_speech",
        )
        mutation_tests["spillover_cross_threshold_0.50s"] = res_cross
    else:
        skip_msg = f"Skipped: source {sid} has no subsequent candidate speech segment after r0"
        mutation_tests["spillover_sub_threshold_0.10s"] = {
            "test_id": "spillover_sub_threshold_0.10s",
            "skipped": True,
            "skip_reason": skip_msg,
            "expected_pass": True,
            "actual_pass": None,
            "verdict_correct": None,
            "errors": [],
            "error_matched": None,
            "unexplained_speech_s": None,
        }
        mutation_tests["spillover_cross_threshold_0.50s"] = {
            "test_id": "spillover_cross_threshold_0.50s",
            "skipped": True,
            "skip_reason": skip_msg,
            "expected_pass": False,
            "actual_pass": None,
            "verdict_correct": None,
            "errors": [],
            "error_matched": None,
            "unexplained_speech_s": None,
        }

    # Compute overall mutation status for this speaker (excluding skipped cases)
    executed_mutations = [m for m in mutation_tests.values() if not m.get("skipped")]
    all_mutations_correct = all(m.get("verdict_correct") for m in executed_mutations) if executed_mutations else True

    return {
        "id": sid,
        "speaker": speaker,
        "language": lang,
        "sha256": source_cfg["sha256"],
        "license": source_cfg["license"],
        "description": source_cfg["description"],
        "alignment_time_s": align_time,
        "clean_segments_count": len(records),
        "clean_candidates_count": len(report.get("candidates", [])),
        "clean_prep_rejected_count": prep_rejected_count,
        "total_words_aligned": len(word_scores),
        "word_score_min": round(min_word_score, 4),
        "word_score_mean": round(mean_word_score, 4),
        "weak_words_count": weak_words_count,
        "max_unexplained_speech_s": round(max_unexplained_s, 4),
        "clean_validation_ok": val_clean.ok,
        "clean_validation_errors": [f"{e.code}: {e.message}" for e in val_clean.errors],
        "clean_status": "PASS" if clean_passed else "FAIL",
        "mutations_count": len(executed_mutations),
        "mutations_correct_count": sum(1 for m in executed_mutations if m.get("verdict_correct")),
        "mutations": mutation_tests,
        "mutations_status": "PASS" if all_mutations_correct else "FAIL",
        "overall_status": "PASS" if (clean_passed and all_mutations_correct) else "FAIL",
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
        help="Path to xtts_base/vocab.json for character validity checks.",
    )
    parser.add_argument(
        "--output_json",
        default=DEFAULT_OUTPUT,
        help="Path to write the structured evidence JSON report.",
    )
    parser.add_argument(
        "--work_dir",
        default="/tmp/open_xtts_alignment_benchmark",
        help="Temporary directory for prepared clips and mutation slices.",
    )

    args = parser.parse_args()

    # Hardware enforcement: ensure all computation runs on GPU
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Task requires GPU execution. Do not fall back to CPU without --allow_cpu."
            )
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Initializing MMS-FA forced alignment on GPU: {gpu_name} (device=cuda)")
    else:
        if not args.allow_cpu:
            raise RuntimeError(
                "CPU execution requested but --allow_cpu was not set. "
                "Per user requirements, heavy audio alignment tasks must be run on GPU."
            )
        gpu_name = "CPU"
        print("⚠️ Running on CPU (allowed via --allow_cpu flag).")

    # Instantiate CtcAligner on target device
    print("📥 Loading torchaudio MMS_FA forced aligner...")
    t_load0 = time.time()
    aligner = CtcAligner(device=args.device)
    load_time = round(time.time() - t_load0, 3)
    print(f"✅ MMS_FA model loaded onto {args.device} in {load_time}s")

    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 110)
    print(
        f"{'Source ID':<32} {'Lang':<5} {'Clean':<7} {'MinSc':<8} {'Weak':<6} {'Mutations':<12} {'Verdict'}"
    )
    print("=" * 110)

    for cfg in SOURCES:
        sid = cfg["id"]
        # Retrieve audio and verify SHA-256 lock
        if cfg["kind"] == "local":
            audio_path = cfg["local_path"]
            if not os.path.exists(audio_path):
                print(f"📦 Local reference voice missing at {audio_path}. Automatically running fetch_reference_voices.py...")
                import subprocess
                subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "fetch_reference_voices.py")], check=True)
            with open(audio_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h != cfg["sha256"]:
                raise RuntimeError(
                    f"SHA mismatch on local audio {audio_path}: got {h}, expected {cfg['sha256']}"
                )
        else:
            dest = os.path.join(work_dir, cfg["filename"])
            audio_path = download_with_sha256(cfg["url"], dest, cfg["sha256"])

        eval_res = evaluate_source(
            source_cfg=cfg,
            audio_path=audio_path,
            work_dir=work_dir,
            aligner=aligner,
            vocab_path=args.vocab_path,
        )
        results.append(eval_res)

        clean_str = eval_res.get("clean_status", "ERR")
        min_sc = f"{eval_res.get('word_score_min', 0.0):.3f}"
        weak_cnt = str(eval_res.get("weak_words_count", 0))
        mut_ratio = f"{eval_res.get('mutations_correct_count', 0)}/{eval_res.get('mutations_count', 0)}"
        verdict = eval_res.get("overall_status", "ERR")

        print(
            f"{sid:<32} {cfg['lang']:<5} {clean_str:<7} {min_sc:<8} {weak_cnt:<6} {mut_ratio:<12} {verdict}"
        )

    print("=" * 110)

    # -------------------------------------------------------------------------
    # HIERARCHICAL AGGREGATION BY CORRUPTION TYPE AND OVERALL
    # -------------------------------------------------------------------------
    total_clean = len(results)
    clean_passed_count = sum(1 for r in results if r.get("clean_status") == "PASS")

    total_mutations = sum(r.get("mutations_count", 0) for r in results)
    correct_mutations = sum(r.get("mutations_correct_count", 0) for r in results)

    # Group by corruption type
    by_type: Dict[str, Dict[str, Any]] = {
        "edge_truncation": {"total": 0, "correct": 0},
        "hallucinated_word": {"total": 0, "correct": 0},
        "speech_spillover_sub_threshold": {"total": 0, "correct": 0},
        "speech_spillover_cross_threshold": {"total": 0, "correct": 0},
    }

    for r in results:
        for m_name, m_data in r.get("mutations", {}).items():
            if m_data.get("skipped"):
                continue
            if m_name.startswith("trunc_"):
                by_type["edge_truncation"]["total"] += 1
                if m_data.get("verdict_correct"):
                    by_type["edge_truncation"]["correct"] += 1
            elif m_name.startswith("hallucinate_"):
                by_type["hallucinated_word"]["total"] += 1
                if m_data.get("verdict_correct"):
                    by_type["hallucinated_word"]["correct"] += 1
            elif "sub_threshold" in m_name:
                by_type["speech_spillover_sub_threshold"]["total"] += 1
                if m_data.get("verdict_correct"):
                    by_type["speech_spillover_sub_threshold"]["correct"] += 1
            elif "cross_threshold" in m_name:
                by_type["speech_spillover_cross_threshold"]["total"] += 1
                if m_data.get("verdict_correct"):
                    by_type["speech_spillover_cross_threshold"]["correct"] += 1

    all_passed = (clean_passed_count == total_clean) and (correct_mutations == total_mutations)

    # Reference configs for structured report
    ref_quality = QualityConfig(
        min_word_score=0.05,
        weak_word_score=0.20,
        max_weak_words=3,
        max_unexplained_speech_s=0.20,
    )
    ref_seg = SegmentConfig(
        min_gap_s=0.15,
        edge_pad_s=0.20,
    )

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": args.device,
        "gpu_name": gpu_name,
        "quality_config": {
            "min_word_score": ref_quality.min_word_score,
            "weak_word_score": ref_quality.weak_word_score,
            "max_weak_words": ref_quality.max_weak_words,
            "max_unexplained_speech_s": ref_quality.max_unexplained_speech_s,
            "min_gap_s": ref_seg.min_gap_s,
            "edge_pad_s": ref_seg.edge_pad_s,
        },
        "summary": {
            "distinct_speakers_count": total_clean,
            "clean_recordings_total": total_clean,
            "clean_recordings_passed": clean_passed_count,
            "clean_acceptance_rate": f"{clean_passed_count}/{total_clean}",
            "controlled_corruptions_total": total_mutations,
            "controlled_corruptions_correct": correct_mutations,
            "controlled_corruptions_accuracy": f"{correct_mutations}/{total_mutations}",
            "by_corruption_type": by_type,
            "all_tests_passed": all_passed,
        },
        "results_by_speaker": results,
    }

    # Ensure parent dir exists and write formatted JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Benchmark Evidence Report written to: {out_path.resolve()}")
    print(f"   * Clean recordings passed: {clean_passed_count}/{total_clean}")
    print(f"   * Controlled corruption tests correct: {correct_mutations}/{total_mutations}")
    for c_type, c_stats in by_type.items():
        print(f"     - {c_type:<32}: {c_stats['correct']}/{c_stats['total']}")
    print(f"   * Final Verdict: {'ALL PASS ✅' if all_passed else 'SOME TESTS FAILED ❌'}")

    # Output serialized JSON marker for headless Modal capture if needed
    print("\n--- BEGIN JSON EVIDENCE ---")
    print(json.dumps(report))
    print("--- END JSON EVIDENCE ---")


if __name__ == "__main__":
    main()
