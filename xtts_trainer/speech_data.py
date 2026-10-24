"""Prepare speech data for XTTS training: public sample sources or your own recording.

Every transcript is traceable (docs/recipe_design.md §6, phase 6):
- corpus clips take their text from the corpus's own metadata (LJSpeech metadata.csv, normalized column);
- recordings are transcribed (human transcript or Whisper), forced-aligned, and cut only at word gaps,
  so each clip's text is exactly the words aligned inside it. Alignment is stored in dataset.jsonl.
Each public source declares its own language; --lang applies only to --input_file.
The output is validated (xtts_trainer.data.validate) before this command reports success.

    python -m xtts_trainer.speech_data --source sample --output_dir sample_audio
    python -m xtts_trainer.speech_data --input_file me.wav --lang en --speaker_name me --output_dir my_data
"""

import argparse
import csv
import io
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import soundfile as sf

from xtts_trainer.data import DatasetRecord, validate_dataset
from xtts_trainer.prepare.align import CtcAligner
from xtts_trainer.prepare.pipeline import PrepConfig, load_mono, prepare_recording, write_prepared
from xtts_trainer.spec import SPEC

DEFAULT_VOCAB = "xtts_base/vocab.json"
LJSPEECH_BASE = "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech"


@dataclass
class Source:
    """A public speech source. lang is the source's own language, never taken from the command line."""

    lang: str
    speaker: str
    description: str
    urls: List[str] = field(default_factory=list)
    # "corpus_clips": already one utterance per file, text from metadata_url; "recording": align and cut
    kind: str = "recording"
    metadata_url: Optional[str] = None
    # A human transcript of the whole recording; None = Whisper
    transcript: Optional[str] = None


SOURCES: Dict[str, Source] = {
    "ljspeech_sample": Source(
        lang="en", speaker="ljspeech", kind="corpus_clips",
        description="LJSpeech LJ001-0001..0006 (public domain); text from metadata.csv, normalized column",
        urls=[f"{LJSPEECH_BASE}/wavs/LJ001-000{i}.wav" for i in range(1, 7)],
        metadata_url=f"{LJSPEECH_BASE}/metadata.csv",
    ),
    "jfk": Source(
        lang="en", speaker="jfk",
        description="JFK inaugural address excerpt (public domain, Whisper test clip), human transcript, aligned",
        urls=["https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"],
        transcript="And so, my fellow Americans: ask not what your country can do for you, "
                   "ask what you can do for your country.",
    ),
    "long_10m": Source(
        lang="en", speaker="speaker_1",
        description="10-minute continuous English speech (reach-vb/random-audios), Whisper transcript, aligned",
        urls=["https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/4469669-10.mp3"],
    ),
    "long_2h": Source(
        lang="en", speaker="speaker_1",
        description="2-hour continuous English speech (reach-vb/random-audios), Whisper transcript, aligned",
        urls=["https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/4469669.mp3"],
    ),
}
# Named bundles of sources
SOURCE_SETS = {"sample": ["ljspeech_sample", "jfk"]}


def _download(url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "open-xtts-trainer"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    return path


def ljspeech_texts(metadata_csv: str) -> Dict[str, str]:
    """id -> normalized transcript (column 3) from an LJSpeech metadata.csv."""
    texts = {}
    for row in csv.reader(io.StringIO(metadata_csv), delimiter="|", quoting=csv.QUOTE_NONE):
        if len(row) != 3:
            raise ValueError(f"LJSpeech metadata row has {len(row)} columns, expected 3: {row}")
        texts[row[0]] = row[2]
    return texts


def prepare_corpus_clips(name: str, source: Source, output_dir: str) -> Tuple[List[DatasetRecord], dict]:
    cache = os.path.join(output_dir, "_downloads")
    with open(_download(source.metadata_url, cache), "r", encoding="utf-8") as f:
        texts = ljspeech_texts(f.read())
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    records = []
    for url in source.urls:
        clip_id = os.path.splitext(os.path.basename(url))[0]
        if clip_id not in texts:
            raise KeyError(f"{clip_id} is not in {source.metadata_url}")
        audio = load_mono(_download(url, cache), SPEC.output_sample_rate)
        rel = f"clips/{name}_{clip_id}.wav"
        sf.write(os.path.join(output_dir, rel), audio, SPEC.output_sample_rate, subtype="PCM_16")
        records.append(DatasetRecord(audio=rel, text=texts[clip_id], lang=source.lang, speaker=source.speaker,
                                     duration_s=round(len(audio) / SPEC.output_sample_rate, 4),
                                     sr=SPEC.output_sample_rate, transcript_source="dataset", source=clip_id))
    return records, {"input": name, "transcript_source": "dataset", "clips": len(records),
                     "metadata": source.metadata_url}


def prepare_sources(names: List[str], output_dir: str, aligner: Optional[CtcAligner] = None,
                    config: Optional[PrepConfig] = None, max_seconds: Optional[float] = None) -> str:
    records, reports = [], []
    aligner = aligner or CtcAligner()
    for name in names:
        source = SOURCES[name]
        print(f"Preparing '{name}': {source.description}")
        if source.kind == "corpus_clips":
            recs, report = prepare_corpus_clips(name, source, output_dir)
        else:
            recs, report = [], None
            for url in source.urls:
                path = _download(url, os.path.join(output_dir, "_downloads"))
                recs_i, report = prepare_recording(path, output_dir, lang=source.lang, speaker=source.speaker,
                                                   transcript=source.transcript, aligner=aligner, config=config,
                                                   clip_prefix=name, source=url, max_seconds=max_seconds)
                recs += recs_i
        records += recs
        reports.append(report)
    return write_prepared(output_dir, records, reports)


def validate_prepared(dataset_path: str, vocab_path: str, extra_languages: List[str] = ()) -> None:
    """Phase 6 step 5: a prepared dataset must pass Phase 3 validation; report next to the dataset."""
    report = validate_dataset(dataset_path, vocab_path, extra_languages)
    with open(os.path.join(os.path.dirname(dataset_path), "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report.to_json(), f, indent=2, ensure_ascii=False)
    for warning in report.warnings:
        print(f"WARNING {warning}")
    report.raise_for_errors()
    print(f"Validated {report.stats['clips']} clips ({report.stats['hours'] * 60:.1f} min): {dataset_path}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Prepare aligned speech data for XTTS training")
    parser.add_argument("--source", default=None, choices=sorted(SOURCES) + sorted(SOURCE_SETS),
                        help="Public source (default: sample = LJSpeech clips + aligned JFK)")
    parser.add_argument("--input_file", "-i", default=None, help="Your own recording to transcribe, align and cut")
    parser.add_argument("--lang", default=None, help="Language of --input_file (required with it)")
    parser.add_argument("--speaker_name", default=None, help="Speaker id for --input_file (required with it)")
    parser.add_argument("--transcript_file", default=None,
                        help="Human transcript of the whole --input_file (up to 60 s); default: Whisper")
    parser.add_argument("--output_dir", default="sample_audio")
    parser.add_argument("--max_minutes", type=float, default=None, help="Only prepare the first N minutes")
    parser.add_argument("--fix_pauses", action="store_true", help="Add commas at long unpunctuated pauses (ablation A6)")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="Tokenizer vocab used to validate the result")
    parser.add_argument("--languages", nargs="+", default=[], help="Registered extra language codes (validation)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.input_file and args.source:
        parser.error("use either --source or --input_file")
    if not args.input_file and (args.lang or args.speaker_name or args.transcript_file):
        parser.error("--lang, --speaker_name and --transcript_file apply only to --input_file; "
                     "public sources declare their own language and speaker")
    if args.input_file and not (args.lang and args.speaker_name):
        parser.error("--input_file needs --lang and --speaker_name (no defaults)")
    if not os.path.exists(args.vocab):
        parser.error(f"{args.vocab} not found; fetch it with: python scripts/fetch_base_artifacts.py --only vocab.json")

    config = PrepConfig(fix_pauses=args.fix_pauses)
    aligner = CtcAligner(device=args.device)
    max_seconds = args.max_minutes * 60 if args.max_minutes else None
    if args.input_file:
        transcript = None
        if args.transcript_file:
            with open(args.transcript_file, "r", encoding="utf-8") as f:
                transcript = " ".join(f.read().split())
        records, report = prepare_recording(args.input_file, args.output_dir, lang=args.lang.lower(),
                                            speaker=args.speaker_name, transcript=transcript, aligner=aligner,
                                            config=config, max_seconds=max_seconds)
        dataset_path = write_prepared(args.output_dir, records, [report])
    else:
        source = args.source or "sample"
        dataset_path = prepare_sources(SOURCE_SETS.get(source, [source]), args.output_dir, aligner, config, max_seconds)
    validate_prepared(dataset_path, args.vocab, args.languages)


if __name__ == "__main__":
    main()
