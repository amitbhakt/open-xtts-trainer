"""Convert a legacy pipe-separated metadata file to dataset.jsonl, with every column declared explicitly.

Nothing is guessed: --columns names each column in order (audio and text are required); speaker and lang
come from columns or from --speaker / --lang, never from filenames or column heuristics.

Examples:
    # LJSpeech-style: audio|text|normalized_text
    python scripts/convert_metadata_csv.py LJSpeech-1.1/metadata.csv --audio_dir LJSpeech-1.1/wavs \\
        --columns audio,ignore,text --speaker ljspeech --lang en --transcript_source dataset
    # this repo's old 4-column format: audio|text|speaker|lang
    python scripts/convert_metadata_csv.py sample_audio/metadata.csv --columns audio,text,speaker,lang \\
        --transcript_source human
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import soundfile as sf

from xtts_trainer.data import DATASET_FILENAME, DatasetRecord, write_dataset

KNOWN_COLUMNS = {"audio", "text", "speaker", "lang", "ignore"}


def convert(metadata_path, columns, audio_dir=None, output=None, speaker=None, lang=None,
            transcript_source="dataset", audio_ext=".wav", delimiter="|"):
    unknown = [c for c in columns if c not in KNOWN_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown column names {unknown}; use {sorted(KNOWN_COLUMNS)}")
    for required in ("audio", "text"):
        if columns.count(required) != 1:
            raise ValueError(f"--columns must name '{required}' exactly once")
    if "speaker" not in columns and not speaker:
        raise ValueError("No speaker column: pass --speaker")
    if "lang" not in columns and not lang:
        raise ValueError("No lang column: pass --lang")

    audio_dir = audio_dir or os.path.dirname(os.path.abspath(metadata_path))
    output = output or os.path.join(audio_dir, DATASET_FILENAME)
    out_dir = os.path.dirname(os.path.abspath(output))

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(delimiter)
            if len(parts) != len(columns):
                raise ValueError(f"{metadata_path} line {line_no}: {len(parts)} fields, --columns declares {len(columns)}")
            row = dict(zip(columns, parts))
            name = row["audio"] if os.path.splitext(row["audio"])[1] else row["audio"] + audio_ext
            path = os.path.join(audio_dir, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{metadata_path} line {line_no}: audio not found: {path}")
            info = sf.info(path)
            records.append(DatasetRecord(
                audio=os.path.relpath(path, out_dir),
                text=row["text"],
                lang=row.get("lang", lang),
                speaker=row.get("speaker", speaker),
                duration_s=round(info.frames / info.samplerate, 4),
                sr=info.samplerate,
                transcript_source=transcript_source,
                source=f"{os.path.basename(metadata_path)}:{line_no}",
            ))
    return write_dataset(output, records)


def main():
    parser = argparse.ArgumentParser(description="Convert a legacy metadata.csv to dataset.jsonl (explicit columns).")
    parser.add_argument("metadata")
    parser.add_argument("--columns", required=True, help="Comma-separated column names in order, e.g. audio,text,speaker,lang")
    parser.add_argument("--audio_dir", default=None, help="Where audio paths are resolved (default: metadata's directory)")
    parser.add_argument("--output", default=None, help=f"Output path (default: <audio_dir>/{DATASET_FILENAME})")
    parser.add_argument("--speaker", default=None, help="Speaker for every row (when there is no speaker column)")
    parser.add_argument("--lang", default=None, help="Language for every row (when there is no lang column)")
    parser.add_argument("--transcript_source", required=True, help="dataset | human | asr:<model id>")
    parser.add_argument("--delimiter", default="|")
    args = parser.parse_args()

    path = convert(args.metadata, args.columns.split(","), args.audio_dir, args.output, args.speaker, args.lang,
                   args.transcript_source, delimiter=args.delimiter)
    print(f"Wrote {path}. Validate it with: python -m xtts_trainer.data.validate {path} --vocab xtts_base/vocab.json")


if __name__ == "__main__":
    main()
