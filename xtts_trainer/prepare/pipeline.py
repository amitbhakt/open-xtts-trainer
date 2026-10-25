"""Recording -> aligned training segments -> dataset.jsonl (docs/recipe_design.md §6, phase 6).

Transcribe, then segment, never the reverse:
1. Transcript: a human transcript of the whole recording, or Whisper (large-v3 class) long-form phrases.
2. Forced alignment (align.py) of that transcript, in windows of at most MAX_WINDOW_S.
3. Cuts at word gaps only (segment.py); each segment's text is its aligned words.
4. Segments whose alignment can't be trusted are dropped, with the reason, in the prep report.
"""

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

from xtts_trainer.data import DATASET_FILENAME, DatasetRecord, write_dataset
from xtts_trainer.prepare.align import AlignedWord, CtcAligner, WindowAlignment
from xtts_trainer.prepare.pauses import PauseConfig, insert_pause_commas, pause_issues
from xtts_trainer.prepare.segment import (
    Dropped, QualityConfig, Segment, SegmentConfig, plan_segments, segment_problems,
)
from xtts_trainer.spec import SPEC

PREP_REPORT_FILENAME = "prep_report.json"
# Whisper phrases are grouped into alignment windows of about this length, padded on both sides
WINDOW_TARGET_S = 30.0
WINDOW_PAD_S = 0.5

# (audio, sample_rate, lang) -> [(text, start_s, end_s or None)]
PhraseTranscriber = Callable[[np.ndarray, int, str], List[Tuple[str, float, Optional[float]]]]


@dataclass
class PrepConfig:
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    pauses: PauseConfig = field(default_factory=PauseConfig)
    # Ablation A6 "corrected" arm: add a comma after each long unpunctuated pause. Off by default.
    fix_pauses: bool = False


def load_mono(path: str, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    if sr != sample_rate:
        wav = F.resample(wav, sr, sample_rate)
    return wav.numpy()


def phrase_windows(phrases: Sequence[Tuple[str, float, Optional[float]]], audio_s: float,
                   target_s: float = WINDOW_TARGET_S) -> List[Tuple[float, float, str]]:
    """Group consecutive Whisper phrases into (start_s, end_s, text) windows of about target_s."""
    windows: List[List] = []
    last_end = 0.0
    for text, start, end in phrases:
        text = text.strip()
        if not text:
            continue
        start = min(audio_s, max(last_end, 0.0 if start is None else float(start)))
        if start >= audio_s:
            continue
        if end is None or float(end) <= start:
            end = min(audio_s, start + target_s)
        else:
            end = min(audio_s, float(end))
        if end - start > 50.0:
            end = min(audio_s, start + target_s)
        if windows and (end - windows[-1][0]) <= target_s and start >= windows[-1][0]:
            windows[-1][1] = max(windows[-1][1], end)
            windows[-1][2] += " " + text
        else:
            windows.append([start, end, text])
        last_end = max(last_end, end)
    return [(s, e, t) for s, e, t in windows]


def align_recording(audio: np.ndarray, sample_rate: int, windows: Sequence[Tuple[float, float, str]],
                    aligner: CtcAligner) -> List[WindowAlignment]:
    """Align each window's text inside its padded span; word times are in recording time."""
    audio_s = len(audio) / sample_rate
    out = []
    for k, (start, end, text) in enumerate(windows):
        # Pad into the neighbouring windows by at most half the space between them
        lo = max(0.0, start - WINDOW_PAD_S) if k == 0 else max(start - WINDOW_PAD_S, (windows[k - 1][1] + start) / 2)
        hi = min(audio_s, end + WINDOW_PAD_S) if k == len(windows) - 1 else min(end + WINDOW_PAD_S, (end + windows[k + 1][0]) / 2)
        a, b = int(lo * sample_rate), int(hi * sample_rate)
        out.append(aligner.align(audio[a:b], sample_rate, text, offset_s=a / sample_rate))
    return out


def unexplained_speech_s(alignments: Sequence[WindowAlignment], segment: Segment) -> float:
    return sum(w.unexplained_speech_s(segment.start_s, segment.end_s) for w in alignments)


def prepare_recording(
    input_file: str,
    output_dir: str,
    lang: str,
    speaker: str,
    transcript: Optional[str] = None,
    asr_model: Optional[str] = None,
    phrase_transcriber: Optional[PhraseTranscriber] = None,
    aligner: Optional[CtcAligner] = None,
    config: Optional[PrepConfig] = None,
    clip_prefix: str = "clip",
    source: Optional[str] = None,
    max_seconds: Optional[float] = None,
    device: str = "cpu",
) -> Tuple[List[DatasetRecord], dict]:
    """Prepare one recording. Returns (records, report); writes clips under output_dir/clips.

    transcript: a human transcript of the whole recording (aligned in one window, so at most MAX_WINDOW_S).
    Without it, phrase_transcriber (default: Whisper asr_model) transcribes the recording.
    """
    config = config or PrepConfig()
    aligner = aligner or CtcAligner(device=device)
    out_rate = SPEC.output_sample_rate
    audio = load_mono(input_file, out_rate)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * out_rate)]
    audio_s = len(audio) / out_rate

    if transcript is not None:
        transcript_source = "human"
        windows = [(0.0, audio_s, transcript)]
    else:
        from xtts_trainer.asr import default_asr_model, transcribe_phrases

        asr_model = asr_model or default_asr_model(lang)
        transcript_source = f"asr:{asr_model}"
        if phrase_transcriber is None:
            phrase_transcriber = lambda a, sr, l: transcribe_phrases(a, sr, l, asr_model, device=device)  # noqa: E731
        phrases = phrase_transcriber(audio, out_rate, lang)
        if not phrases:
            raise RuntimeError(f"ASR produced no transcript for {input_file}")
        windows = phrase_windows(phrases, audio_s)

    alignments = align_recording(audio, out_rate, windows, aligner)
    words: List[AlignedWord] = [w for a in alignments for w in a.words]
    plan = plan_segments(words, audio_s, config.segment)

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    records, kept, dropped, candidates = [], [], list(plan.dropped), []
    for seg in plan.segments:
        unexplained = unexplained_speech_s(alignments, seg)
        problems = segment_problems(seg, unexplained, config.quality)
        # Every planned segment's word scores, kept or not: the quality thresholds are [E5] and get
        # calibrated from these (docs/recipe_design.md §3)
        candidates.append({"text": seg.text, "start_s": round(seg.start_s, 3), "end_s": round(seg.end_s, 3),
                           "kept": not problems, "problems": problems, "unexplained_speech_s": round(unexplained, 3),
                           "words": [[w.text, round(w.start_s, 3), round(w.end_s, 3), round(w.score, 3)] for w in seg.words]})
        if problems:
            dropped.append(Dropped(seg.start_s, seg.end_s, seg.text, "; ".join(problems)))
            continue
        timed = [(w.text, w.start_s, w.end_s) for w in seg.words]
        texts = insert_pause_commas(timed, config.pauses) if config.fix_pauses else [t for t, _, _ in timed]
        name = f"{clip_prefix}_{len(records) + 1:04d}.wav"
        a, b = int(round(seg.start_s * out_rate)), int(round(seg.end_s * out_rate))
        sf.write(os.path.join(clips_dir, name), audio[a:b], out_rate, subtype="PCM_16")
        clip_start = a / out_rate
        word_json = []
        for w, text in zip(seg.words, texts):
            entry = w.to_json(offset_s=clip_start)
            entry["w"] = text
            word_json.append(entry)
        records.append(DatasetRecord(
            audio=f"clips/{name}", text=" ".join(texts), lang=lang, speaker=speaker,
            duration_s=round((b - a) / out_rate, 4), sr=out_rate, transcript_source=transcript_source,
            source=source or os.path.basename(input_file),
            alignment={"source_audio": os.path.basename(input_file), "start_s": round(clip_start, 3),
                       "end_s": round(b / out_rate, 3), "aligner": aligner.name, "words": word_json},
        ))
        kept.append(seg)

    report = prep_report(input_file, audio_s, transcript_source, aligner.name, kept, records, dropped, config)
    report["candidates"] = candidates
    return records, report


def prep_report(input_file: str, audio_s: float, transcript_source: str, aligner_name: str,
                kept: Sequence[Segment], records: Sequence[DatasetRecord], dropped: Sequence[Dropped],
                config: PrepConfig) -> dict:
    durations = [s.duration_s for s in kept]
    cps = [len(r.text.replace(" ", "")) / r.duration_s for r in records]
    scores = [w.score for s in kept for w in s.words]
    pauses = Counter(i["kind"] for s in kept for i in pause_issues([(w.text, w.start_s, w.end_s) for w in s.words], config.pauses))
    return {
        "input": os.path.basename(input_file),
        "input_s": round(audio_s, 3),
        "transcript_source": transcript_source,
        "aligner": aligner_name,
        "segments": len(kept),
        "kept_s": round(sum(durations), 3),
        "kept_fraction": round(sum(durations) / audio_s, 4) if audio_s else 0.0,
        "duration_histogram": dict(sorted(Counter(f"{int(d)}-{int(d) + 1}s" for d in durations).items())),
        "chars_per_second": {"min": round(min(cps), 2), "max": round(max(cps), 2),
                             "mean": round(float(np.mean(cps)), 2)} if cps else None,
        "alignment_score": {"min": round(min(scores), 3), "mean": round(float(np.mean(scores)), 3)} if scores else None,
        "pause_issues_before_fix": dict(pauses),
        "fix_pauses": config.fix_pauses,
        "dropped": [asdict(d) for d in dropped],
        "config": asdict(config),
    }


def write_prepared(output_dir: str, records: List[DatasetRecord], reports: List[dict]) -> str:
    """Write dataset.jsonl and prep_report.json (one entry per input recording)."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, PREP_REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"recordings": reports}, f, indent=2, ensure_ascii=False)
    if not records:
        raise RuntimeError(f"no usable segments; see the dropped reasons in {report_path}")
    return write_dataset(os.path.join(output_dir, DATASET_FILENAME), records)
