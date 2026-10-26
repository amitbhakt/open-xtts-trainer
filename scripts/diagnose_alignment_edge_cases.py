#!/usr/bin/env python3
"""Diagnostic script: Measure exact frame-level emissions, token paths, and word scores

Investigates the two failure modes identified in the multilingual alignment benchmark:
1. The 7 accepted transcript-insertion cases:
   - Measure the exact frame-weighted score of the inserted word.
   - Measure the individual scores of all other words in the sentence.
   - Determine whether the inserted word exceeded 0.05 or was absorbed by the weak-word limit.
2. The speech spillover cases (+500ms):
   - Measure the exact duration change of the terminal word.
   - Inspect the trellis path over the appended frames: are frames assigned to the final character or to blanks?
   - Measure the argmax speech emission over the appended frames to verify speech was actually present.
   - Measure the exact numerical value of unexplained_speech_s.

Usage:
    python scripts/diagnose_alignment_edge_cases.py --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xtts_trainer.prepare.align import CtcAligner, ALIGN_SAMPLE_RATE, BLANK_INDEX, STAR
from xtts_trainer.prepare.pipeline import PrepConfig, prepare_recording, load_mono
from xtts_trainer.prepare.segment import QualityConfig, SegmentConfig, segment_problems, Segment
from xtts_trainer.spec import SPEC
from scripts.validate_alignment_multilingual import SOURCES, download_with_sha256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    aligner = CtcAligner(device=args.device)
    quality_cfg = QualityConfig(
        min_word_score=0.05,
        weak_word_score=0.20,
        max_weak_words=3,
        max_unexplained_speech_s=0.20,
    )
    work_dir = "/tmp/diag_alignment"
    os.makedirs(work_dir, exist_ok=True)

    print("=" * 100)
    print("PART 1: INVESTIGATION OF THE 7 ACCEPTED TRANSCRIPT INSERTIONS")
    print("=" * 100)

    # The 7 cases identified in the benchmark:
    cases_to_check = [
        ("jfk_english_male", "hallucinate_tail_content"),
        ("arctic_slt_english_female", "hallucinate_tail_content"),
        ("arctic_bdl_english_male", "hallucinate_mid_content"),
        ("arctic_awb_english_scottish", "hallucinate_mid_content"),
        ("ljspeech_female", "hallucinate_head_function"),
        ("ljspeech_female", "hallucinate_tail_content"),
        ("spanish_cuentos_male", "hallucinate_tail_content"),
    ]

    for sid, m_name in cases_to_check:
        cfg = next(c for c in SOURCES if c["id"] == sid)
        if cfg["kind"] == "local":
            audio_path = cfg["local_path"]
        else:
            dest = os.path.join(work_dir, cfg["filename"])
            audio_path = download_with_sha256(cfg["url"], dest, cfg["sha256"])

        # Prepare clean recording to get clean r0 clip
        out_clean = os.path.join(work_dir, sid, "clean")
        os.makedirs(out_clean, exist_ok=True)
        seg_cfg = SegmentConfig(min_s=cfg.get("min_s", 2.0), max_s=cfg.get("max_s", 11.0))
        prep_cfg = PrepConfig(segment=seg_cfg, quality=quality_cfg)
        records, report = prepare_recording(
            input_file=audio_path,
            output_dir=out_clean,
            lang=cfg["lang"],
            speaker=cfg["speaker"],
            transcript=cfg["transcript"],
            config=prep_cfg,
            aligner=aligner,
            max_seconds=cfg.get("max_seconds", None),
        )
        r0 = records[0]

        # Load clean clip
        source_audio, sr = sf.read(audio_path, dtype="float32")
        if cfg.get("max_seconds"):
            source_audio = source_audio[: int(cfg["max_seconds"] * sr)]
        if source_audio.ndim > 1:
            source_audio = source_audio.mean(axis=1)
        src_resampled = torchaudio.functional.resample(
            torch.from_numpy(source_audio), sr, SPEC.output_sample_rate
        ).numpy()

        clean_clip = src_resampled[int(r0.alignment["start_s"] * SPEC.output_sample_rate) : int(r0.alignment["end_s"] * SPEC.output_sample_rate)]

        # Construct mutated text
        words = r0.text.strip().split()
        mid_idx = max(1, len(words) // 2)
        if m_name == "hallucinate_tail_content":
            inserted_token = "consecuentemente." if cfg["lang"] == "es" else "consequently."
            mut_text = r0.text.rstrip(".") + " " + inserted_token
        elif m_name == "hallucinate_mid_content":
            inserted_token = "extraordinario" if cfg["lang"] == "es" else "extraordinary"
            mut_text = " ".join(words[:mid_idx] + [inserted_token] + words[mid_idx:])
        elif m_name == "hallucinate_head_function":
            inserted_token = "Moreover,"
            mut_text = inserted_token + " " + r0.text

        # Run direct alignment on the clip
        clip_s = len(clean_clip) / SPEC.output_sample_rate
        window = aligner.align(clean_clip, SPEC.output_sample_rate, mut_text)
        seg = Segment(0.0, clip_s, window.words)
        unexplained = window.unexplained_speech_s(0.0, clip_s)
        problems = segment_problems(seg, unexplained, quality_cfg)

        # Locate inserted word
        inserted_w = next((w for w in window.words if inserted_token.strip(".,").lower() in w.text.lower()), None)
        min_word = min(window.words, key=lambda w: w.score)
        weak_words = [w for w in window.words if w.score < quality_cfg.weak_word_score]

        print(f"\n--- {sid} | Mutation: {m_name} ---")
        print(f"Transcript: \"{mut_text}\"")
        if inserted_w:
            print(f"Inserted Word: '{inserted_w.text}' | Score: {inserted_w.score:.4f} | Span: {inserted_w.start_s:.3f}s .. {inserted_w.end_s:.3f}s (dur={inserted_w.end_s - inserted_w.start_s:.3f}s)")
        else:
            print("Inserted word not found in alignment!")
        print(f"Overall Worst Word: '{min_word.text}' (score={min_word.score:.4f})")
        print(f"Total Weak Words (<0.20): {len(weak_words)} / allowed {quality_cfg.max_weak_words}")
        for ww in weak_words:
            print(f"  * weak: '{ww.text}' score={ww.score:.4f}")
        print(f"Unexplained Speech: {unexplained:.4f}s")
        print(f"Validator Problems: {problems}")
        print(f"Accepted by Validator? {'YES (Evaded Rejection)' if len(problems) == 0 else 'NO (Rejected)'}")

    print("\n" + "=" * 100)
    print("PART 2: INVESTIGATION OF SPEECH SPILLOVER (+500ms)")
    print("=" * 100)

    # Check 3 representative speakers with consecutive sentences:
    spillover_speakers = ["jfk_english_male", "arctic_slt_english_female", "spanish_cuentos_male"]

    for sid in spillover_speakers:
        cfg = next(c for c in SOURCES if c["id"] == sid)
        audio_path = cfg["local_path"] if cfg["kind"] == "local" else os.path.join(work_dir, cfg["filename"])
        
        out_clean = os.path.join(work_dir, sid, "clean")
        records, report = prepare_recording(
            input_file=audio_path,
            output_dir=out_clean,
            lang=cfg["lang"],
            speaker=cfg["speaker"],
            transcript=cfg["transcript"],
            config=PrepConfig(segment=SegmentConfig(min_s=cfg.get("min_s", 2.0), max_s=cfg.get("max_s", 11.0)), quality=quality_cfg),
            aligner=aligner,
            max_seconds=cfg.get("max_seconds", None),
        )
        r0 = records[0]
        r0_start = r0.alignment["start_s"]
        r0_end = r0.alignment["end_s"]

        source_audio, sr = sf.read(audio_path, dtype="float32")
        if cfg.get("max_seconds"):
            source_audio = source_audio[: int(cfg["max_seconds"] * sr)]
        if source_audio.ndim > 1:
            source_audio = source_audio.mean(axis=1)
        src_resampled = torchaudio.functional.resample(torch.from_numpy(source_audio), sr, SPEC.output_sample_rate).numpy()

        candidates = report.get("candidates", [])
        assert len(candidates) >= 2, f"Expected at least 2 candidates for {sid}"
        cand2_first_speech = candidates[1]["words"][0][1]

        # 1. Clean slice
        clean_clip = src_resampled[int(r0_start * SPEC.output_sample_rate) : int(r0_end * SPEC.output_sample_rate)]
        win_clean = aligner.align(clean_clip, SPEC.output_sample_rate, r0.text)
        last_w_clean = win_clean.words[-1]

        # 2. Appended slice: include +500ms into sentence 2 speech
        cross_end = cand2_first_speech + 0.50
        cross_clip = src_resampled[int(r0_start * SPEC.output_sample_rate) : int(cross_end * SPEC.output_sample_rate)]
        
        # Audio energy in the appended 500ms region:
        appended_slice = src_resampled[int(cand2_first_speech * SPEC.output_sample_rate) : int(cross_end * SPEC.output_sample_rate)]
        rms_appended = float(np.sqrt(np.mean(appended_slice**2)))

        win_cross = aligner.align(cross_clip, SPEC.output_sample_rate, r0.text)
        last_w_cross = win_cross.words[-1]

        # Inspect raw emission and path
        audio_16k = torchaudio.functional.resample(torch.from_numpy(cross_clip), SPEC.output_sample_rate, ALIGN_SAMPLE_RATE).numpy()
        emitter = aligner.emitter
        with torch.no_grad():
            emission = emitter(audio_16k)
        frame_s = len(audio_16k) / ALIGN_SAMPLE_RATE / emission.shape[0]

        # Check argmax labels on the appended frames:
        appended_start_s = (cand2_first_speech - r0_start)
        appended_end_s = (cross_end - r0_start)
        f_start = int(appended_start_s / frame_s)
        f_end = int(appended_end_s / frame_s)

        heard_labels = [emitter.labels[idx] for idx in emission[f_start:f_end].argmax(dim=-1).cpu().numpy()]
        non_blank_heard = sum(1 for l in heard_labels if l != '-' and l != STAR)

        unexplained_s = win_cross.unexplained_speech_s(0.0, len(cross_clip) / SPEC.output_sample_rate)

        print(f"\n--- {sid} | Spillover Analysis (+500ms appended speech) ---")
        print(f"Candidate 2 speech onset: {cand2_first_speech:.3f}s (pause before cand 2: {cand2_first_speech - r0_end:.3f}s)")
        print(f"Appended audio slice RMS energy: {rms_appended:.4f} (speech confirmed: {rms_appended > 0.02})")
        print(f"Terminal word in clean clip: '{last_w_clean.text}' | span: {last_w_clean.start_s:.3f}..{last_w_clean.end_s:.3f} (dur={last_w_clean.end_s - last_w_clean.start_s:.3f}s, score={last_w_clean.score:.4f})")
        print(f"Terminal word in cross clip: '{last_w_cross.text}' | span: {last_w_cross.start_s:.3f}..{last_w_cross.end_s:.3f} (dur={last_w_cross.end_s - last_w_cross.start_s:.3f}s, score={last_w_cross.score:.4f})")
        print(f"Terminal word duration growth: {last_w_cross.end_s - last_w_cross.start_s - (last_w_clean.end_s - last_w_clean.start_s):.3f}s")
        print(f"Frames in appended region: {f_end - f_start} frames ({len(heard_labels)} total, {non_blank_heard} non-blank speech frames heard)")
        print(f"Argmax heard sample in appended region: {''.join(heard_labels[:30])}...")
        print(f"Calculated unexplained_speech_s: {unexplained_s:.4f}s (threshold={quality_cfg.max_unexplained_speech_s}s)")
        problems_cross = segment_problems(Segment(0.0, len(cross_clip)/SPEC.output_sample_rate, win_cross.words), unexplained_s, quality_cfg)
        print(f"Validator Problems: {problems_cross}")
        print(f"Accepted by Validator? {'YES (Evaded Rejection)' if len(problems_cross) == 0 else 'NO (Rejected)'}")

if __name__ == "__main__":
    main()
