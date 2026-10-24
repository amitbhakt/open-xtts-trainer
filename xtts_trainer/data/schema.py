"""The one dataset format: `dataset.jsonl`, versioned, with no positional guessing.

Line 1 is a header: {"format": "open-xtts-dataset", "version": 1}
Every other line is one clip:
    {"audio": "clips/0001.wav", "text": "...", "lang": "en", "speaker": "spk1",
     "duration_s": 6.21, "sr": 24000, "transcript_source": "dataset"}
Optional per-clip keys: "source" (provenance string), "single_clip_speaker" (true to allow a speaker with
only one clip), "alignment" (when cut from a longer recording by forced alignment:
{"source_audio", "start_s", "end_s", "aligner", "words": [{"w", "start_s", "end_s", "score"}]}, word times
relative to the clip; every key optional except start_s/end_s).
`audio` is relative to the dataset file's directory. Unknown keys are an error, so typos can't pass silently.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DATASET_FILENAME = "dataset.jsonl"
FORMAT_NAME = "open-xtts-dataset"
FORMAT_VERSION = 1

REQUIRED_FIELDS = {
    "audio": str,
    "text": str,
    "lang": str,
    "speaker": str,
    "duration_s": (int, float),
    "sr": int,
    "transcript_source": str,
}
OPTIONAL_FIELDS = {
    "source": str,
    "single_clip_speaker": bool,
    "alignment": dict,
}
# Where a transcript came from: the corpus itself, a human, or an ASR model ("asr:<model id>")
TRANSCRIPT_SOURCES = ("dataset", "human")
ASR_SOURCE_PREFIX = "asr:"


ALIGNMENT_KEYS = {"source_audio": str, "start_s": (int, float), "end_s": (int, float), "aligner": str, "words": list}
ALIGNMENT_WORD_KEYS = {"w": str, "start_s": (int, float), "end_s": (int, float), "score": (int, float)}


class DatasetFormatError(ValueError):
    """The file is not a valid dataset.jsonl (structure, not content)."""


@dataclass
class DatasetRecord:
    audio: str
    text: str
    lang: str
    speaker: str
    duration_s: float
    sr: int
    transcript_source: str
    source: Optional[str] = None
    single_clip_speaker: bool = False
    alignment: Optional[Dict[str, Any]] = None
    line: int = field(default=0, compare=False)  # 1-based line in the file, for error messages

    def audio_path(self, dataset_dir: str) -> str:
        return os.path.normpath(os.path.join(dataset_dir, self.audio))

    def to_json(self) -> Dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "line"}
        for key in ("source", "alignment"):
            if data[key] is None:
                del data[key]
        if not data["single_clip_speaker"]:
            del data["single_clip_speaker"]
        return data


def _parse_record(obj: Any, line_no: int) -> DatasetRecord:
    if not isinstance(obj, dict):
        raise DatasetFormatError(f"line {line_no}: expected a JSON object, got {type(obj).__name__}")
    unknown = sorted(set(obj) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise DatasetFormatError(f"line {line_no}: unknown keys {unknown}")
    missing = sorted(k for k in REQUIRED_FIELDS if k not in obj)
    if missing:
        raise DatasetFormatError(f"line {line_no}: missing required keys {missing}")
    for key, value in obj.items():
        expected = REQUIRED_FIELDS.get(key) or OPTIONAL_FIELDS[key]
        if isinstance(value, bool) and expected is not bool:
            raise DatasetFormatError(f"line {line_no}: '{key}' must not be a boolean")
        if not isinstance(value, expected):
            raise DatasetFormatError(f"line {line_no}: '{key}' has type {type(value).__name__}")
    if "alignment" in obj:
        _check_alignment(obj["alignment"], line_no)
    source = obj["transcript_source"]
    if source not in TRANSCRIPT_SOURCES and not (source.startswith(ASR_SOURCE_PREFIX) and len(source) > len(ASR_SOURCE_PREFIX)):
        raise DatasetFormatError(
            f"line {line_no}: transcript_source '{source}' must be one of {TRANSCRIPT_SOURCES} or 'asr:<model id>'"
        )
    return DatasetRecord(line=line_no, duration_s=float(obj["duration_s"]),
                         **{k: v for k, v in obj.items() if k != "duration_s"})


def _check_alignment(alignment: Dict[str, Any], line_no: int) -> None:
    def fail(msg: str):
        raise DatasetFormatError(f"line {line_no}: alignment {msg}")

    unknown = sorted(set(alignment) - set(ALIGNMENT_KEYS))
    if unknown:
        fail(f"has unknown keys {unknown}")
    for key in ("start_s", "end_s"):
        if key not in alignment:
            fail(f"is missing '{key}'")
    for key, value in alignment.items():
        if isinstance(value, bool) or not isinstance(value, ALIGNMENT_KEYS[key]):
            fail(f"'{key}' has type {type(value).__name__}")
    if alignment["end_s"] <= alignment["start_s"]:
        fail("end_s must be after start_s")
    prev_end = 0.0
    for i, word in enumerate(alignment.get("words", [])):
        if not isinstance(word, dict) or set(word) != set(ALIGNMENT_WORD_KEYS):
            fail(f"word {i} must have exactly the keys {sorted(ALIGNMENT_WORD_KEYS)}")
        for key, expected in ALIGNMENT_WORD_KEYS.items():
            if isinstance(word[key], bool) or not isinstance(word[key], expected):
                fail(f"word {i} '{key}' has type {type(word[key]).__name__}")
        if not prev_end - 1e-3 <= word["start_s"] <= word["end_s"]:
            fail(f"word {i} times are not ordered ({word['start_s']}..{word['end_s']} after {prev_end})")
        prev_end = word["end_s"]


def load_dataset(path: str) -> List[DatasetRecord]:
    """Parse dataset.jsonl strictly. Raises DatasetFormatError on any structural problem."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        raise DatasetFormatError(f"{path} is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as e:
        raise DatasetFormatError(f"{path} line 1: header is not valid JSON ({e})") from e
    if header != {"format": FORMAT_NAME, "version": FORMAT_VERSION}:
        raise DatasetFormatError(
            f"{path} line 1: expected header {{\"format\": \"{FORMAT_NAME}\", \"version\": {FORMAT_VERSION}}}, got {header}"
        )
    records = []
    for i, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            raise DatasetFormatError(f"{path} line {i}: blank line")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise DatasetFormatError(f"{path} line {i}: invalid JSON ({e})") from e
        records.append(_parse_record(obj, i))
    if not records:
        raise DatasetFormatError(f"{path} has a header but no clips")
    return records


def write_dataset(path: str, records: List[DatasetRecord]) -> str:
    """Write records (header first). Every record is re-parsed, so writing invalid structure is impossible."""
    for i, record in enumerate(records, start=2):
        _parse_record(record.to_json(), i)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"format": FORMAT_NAME, "version": FORMAT_VERSION}) + "\n")
        for record in records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path
