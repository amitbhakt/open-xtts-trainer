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
import hashlib
import io
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import soundfile as sf

from xtts_trainer.data import DatasetRecord, validate_dataset
from xtts_trainer.prepare.align import CtcAligner
from xtts_trainer.prepare.pipeline import PrepConfig, load_mono, prepare_recording, write_prepared
from xtts_trainer.spec import SPEC

DEFAULT_VOCAB = "xtts_base/vocab.json"
# Pinned to commits (the branches move) and checked by SHA-256; values and commands in docs/evidence/source_pins.md
LJSPEECH_BASE = "https://raw.githubusercontent.com/coqui-ai/TTS/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/tests/data/ljspeech"
LJSPEECH_SHA256 = {
    f"{LJSPEECH_BASE}/metadata.csv": "2727a6428189a0213d480b2de5eceb159877c2b99b0027b22115d076c868938d",
    f"{LJSPEECH_BASE}/wavs/LJ001-0001.wav": "4d2bd83ff0f7fe33491b03193abdf623f1b4cc2d8103972b33638d3607d86dc5",
    f"{LJSPEECH_BASE}/wavs/LJ001-0002.wav": "6563390fa42121eeeab15f49fa91fd26afe000022bfdaaa882f06224ad549599",
    f"{LJSPEECH_BASE}/wavs/LJ001-0003.wav": "ad97822fa512b4897a01557423ffff7cc1fc763c9b9d5d502a5a1bc566f7440b",
    f"{LJSPEECH_BASE}/wavs/LJ001-0004.wav": "3c9cce186f26b6690b1d96a6a640c3f1bdb10e1f206738344190fa264955a7af",
    f"{LJSPEECH_BASE}/wavs/LJ001-0005.wav": "0d95df8cd780f0ed4f71d74559d42b33cc98ae9e7b398e0eceb17428dfacb9ca",
    f"{LJSPEECH_BASE}/wavs/LJ001-0006.wav": "4b866eebe1e7cb6ded4bcb27caa0638b8ade67a6a8a2f7132952b716dc7006db",
}
JFK_URL = "https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/tests/jfk.flac"
JFK_SHA256 = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"


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
    # Why we may train on it, with where that is stated
    license: str = ""
    # url -> SHA-256 for every file the source downloads (audio and metadata_url), checked after every download.
    # Required: an unpinned URL is an error, not a skipped check.
    sha256: Dict[str, str] = field(default_factory=dict)

    def pinned(self, url: str) -> str:
        if url not in self.sha256:
            raise KeyError(f"{url} has no SHA-256 pin; every downloaded file must be pinned "
                           f"(docs/evidence/source_pins.md)")
        return self.sha256[url]


# One solo reader, public domain. Pinned by SHA-256 of files whose SHA-1 first matched the value archive.org
# publishes (https://archive.org/metadata/art_of_war_librivox); archive.org publishes no SHA-256.
# Replaced reach-vb/random-audios, which has no license, no dataset card and an unknown speaker.
ARCHIVE_ART_OF_WAR = "https://archive.org/download/art_of_war_librivox"
ART_OF_WAR = [(f"{ARCHIVE_ART_OF_WAR}/{name}", sha256) for name, sha256 in [
    ("art_of_war_01-02_sun_tzu_64kb.mp3", "6cb289fa9a13cfc94b18e8a89159da687c60db88c2b6377311f9bc5664d5492f"),
    ("art_of_war_03-04_sun_tzu_64kb.mp3", "57e879fe7822578fd22010f1280185799a119a3482e025d9a71f49878ff45ba7"),
    ("art_of_war_05-06_sun_tzu_64kb.mp3", "ace85bbe5e784c9dc622b9ddb346b513afa7cdb4f9176c45d2fd9df16794bb58"),
    ("art_of_war_07-08_sun_tzu_64kb.mp3", "23bf3869fc869942cd68812417927fc350dee33e114bbe445f41d5157d559aad"),
    ("art_of_war_09-10_sun_tzu_64kb.mp3", "49dcdde816c1e1152c1ff1721171ebe5817d0db5ffd2c8970c2c20511b4043ad"),
    ("art_of_war_11_sun_tzu_64kb.mp3", "031b2426f8a67f7cda7e3a38ed13b061876f7610e078fb8395962669803c27d8"),
    ("art_of_war_12-13_sun_tzu_64kb.mp3", "61acc0e274eef3f740ec4ad4c5c4a18907e35832d0581d1bf9dfdc3b031afafa"),
]]
ART_OF_WAR_LICENSE = ("Public domain: LibriVox recordings are public-domain dedications; archive.org item "
                      "art_of_war_librivox, licenseurl http://creativecommons.org/licenses/publicdomain/")

SOURCES: Dict[str, Source] = {
    "ljspeech_sample": Source(
        lang="en", speaker="ljspeech", kind="corpus_clips",
        description="LJSpeech LJ001-0001..0006 (public domain); text from metadata.csv, normalized column",
        license="Public domain (LJ Speech Dataset, https://keithito.com/LJ-Speech-Dataset/)",
        urls=[f"{LJSPEECH_BASE}/wavs/LJ001-000{i}.wav" for i in range(1, 7)],
        metadata_url=f"{LJSPEECH_BASE}/metadata.csv",
        sha256=LJSPEECH_SHA256,
    ),
    "jfk": Source(
        lang="en", speaker="jfk",
        description="JFK inaugural address excerpt (public domain, Whisper test clip), human transcript, aligned",
        urls=[JFK_URL], sha256={JFK_URL: JFK_SHA256},
        transcript="And so, my fellow Americans: ask not what your country can do for you, "
                   "ask what you can do for your country.",
        license="Public domain: 1961 inaugural address, a work of the US government; clip from the openai/whisper "
                "test data",
    ),
    "art_of_war_short": Source(
        lang="en", speaker="moira_fogarty",
        description="LibriVox: The Art of War (tr. Lionel Giles), read by Moira Fogarty, chapters 1-2 (8.4 min); "
                    "Whisper transcript, aligned",
        urls=[ART_OF_WAR[0][0]], sha256=dict(ART_OF_WAR[:1]),
        license=ART_OF_WAR_LICENSE,
    ),
    "art_of_war": Source(
        lang="en", speaker="moira_fogarty",
        description="LibriVox: The Art of War (tr. Lionel Giles), read by Moira Fogarty, all 13 chapters (72 min); "
                    "Whisper transcript, aligned",
        urls=[u for u, _ in ART_OF_WAR], sha256=dict(ART_OF_WAR),
        license=ART_OF_WAR_LICENSE,
    ),
}
# Named bundles of sources
SOURCE_SETS = {"sample": ["ljspeech_sample", "jfk"]}


def _download(url: str, dest_dir: str, sha256: str) -> str:
    """Download once into dest_dir; the file, fresh or cached, must match its SHA-256 pin."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256 or ""):
        raise ValueError(f"{url}: a SHA-256 pin (64 hex digits) is required, got {sha256!r}")
    sha256 = sha256.lower()
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "open-xtts-trainer"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise RuntimeError(f"{path} has SHA-256 {digest.hexdigest()}, expected {sha256} for {url}; delete it and retry")
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
    with open(_download(source.metadata_url, cache, source.pinned(source.metadata_url)), "r", encoding="utf-8") as f:
        texts = ljspeech_texts(f.read())
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    records = []
    for url in source.urls:
        clip_id = os.path.splitext(os.path.basename(url))[0]
        if clip_id not in texts:
            raise KeyError(f"{clip_id} is not in {source.metadata_url}")
        audio = load_mono(_download(url, cache, source.pinned(url)), SPEC.output_sample_rate)
        rel = f"clips/{name}_{clip_id}.wav"
        sf.write(os.path.join(output_dir, rel), audio, SPEC.output_sample_rate, subtype="PCM_16")
        records.append(DatasetRecord(audio=rel, text=texts[clip_id], lang=source.lang, speaker=source.speaker,
                                     duration_s=round(len(audio) / SPEC.output_sample_rate, 4),
                                     sr=SPEC.output_sample_rate, transcript_source="dataset", source=clip_id))
    return records, {"input": name, "transcript_source": "dataset", "clips": len(records),
                     "metadata": source.metadata_url}


def prepare_sources(names: List[str], output_dir: str, aligner: Optional[CtcAligner] = None,
                    config: Optional[PrepConfig] = None, max_seconds: Optional[float] = None,
                    device: str = "cpu") -> str:
    records, reports = [], []
    aligner = aligner or CtcAligner(device=device)
    for name in names:
        source = SOURCES[name]
        print(f"Preparing '{name}': {source.description}")
        if source.kind == "corpus_clips":
            recs, report = prepare_corpus_clips(name, source, output_dir)
        else:
            recs = []
            for k, url in enumerate(source.urls):
                path = _download(url, os.path.join(output_dir, "_downloads"), source.pinned(url))
                # One prefix per file: clip numbering restarts for every recording
                prefix = name if len(source.urls) == 1 else f"{name}_{k + 1:02d}"
                recs_k, report = prepare_recording(path, output_dir, lang=source.lang, speaker=source.speaker,
                                                   transcript=source.transcript, aligner=aligner, config=config,
                                                   clip_prefix=prefix, source=url, max_seconds=max_seconds,
                                                   device=device)
                report["license"] = source.license
                reports.append(report)
                recs += recs_k
            records += recs
            continue
        report["license"] = source.license
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
                                            config=config, max_seconds=max_seconds, device=args.device)
        dataset_path = write_prepared(args.output_dir, records, [report])
    else:
        source = args.source or "sample"
        dataset_path = prepare_sources(SOURCE_SETS.get(source, [source]), args.output_dir, aligner, config, max_seconds, device=args.device)
    validate_prepared(dataset_path, args.vocab, args.languages)


if __name__ == "__main__":
    main()
