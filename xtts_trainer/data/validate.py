"""validate_dataset(): runs before any training and rejects bad data with every offending line listed.

Checks (each error carries the dataset line and a stable code):
- audio:        file exists, decodes, declared sample rate and duration match the file
- duration:     within upstream's training limits (0.5 s minimum; max_wav_length from GPTArgs)
- text:         non-empty, tokenizes with zero [UNK], within max_text_length (from GPTArgs)
- language:     registered (XTTS base languages + extra_languages) and has a [lang] token in the vocab
- consistency:  characters per second within bounds (catches a long sentence paired with a short clip);
                optional ASR cross-check of the transcript; aligned words (when present) spell the text
                and lie inside the clip; optional re-alignment check (catches a shifted cut)
- pauses:       punctuation agrees with aligned pauses (warnings; errors with strict_pauses)
- speakers:     every speaker has >= 2 clips unless the record sets single_clip_speaker
- leakage:      no duplicate audio; no overlap with the eval set's sentences or reference-voice clips
Warnings (non-blocking) report coverage the recipe cares about, e.g. how much short-utterance data exists.

CLI:
    python -m xtts_trainer.data.validate path/to/dataset.jsonl --vocab xtts_base/vocab.json [--report out.json]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import soundfile as sf

from xtts_trainer.data.schema import ASR_SOURCE_PREFIX, DatasetRecord, load_dataset
from xtts_trainer.spec.languages import XTTS_BASE_LANGUAGES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EVAL_SET = os.path.join(REPO_ROOT, "eval", "dataset", "eval_set.json")
DEFAULT_REFERENCE_LOCK = os.path.join(REPO_ROOT, "eval", "dataset", "reference_voices.lock.json")
# Written by scripts/build_target_voices.py: the target speaker's held-out evaluation clips (eval/target_voices.py)
DEFAULT_TARGET_LOCK = os.path.join(REPO_ROOT, "eval", "target_voices", "target_voices.lock.json")


def target_holdout_hashes(lock_path: str) -> Dict[str, str]:
    """SHA-256 -> voice id for every reference clip and held-out source clip in a target-voice lock."""
    with open(lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    hashes = {}
    for vid, entry in lock["voices"].items():
        hashes[entry["clip_sha256"]] = vid
        for source in entry["source_clips"]:
            hashes[source["sha256"]] = vid
    return hashes


def _upstream_training_limits() -> Tuple[int, int]:
    """(max_wav_length in samples at the GPT rate, max_text_length in tokens) from upstream GPTArgs."""
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs

    args = GPTArgs()
    return args.max_wav_length, args.max_text_length


@dataclass
class ValidationConfig:
    # Upstream XTTSDataset rejects clips under 0.5 s (tts/layers/xtts/trainer/dataset.py:116).
    min_duration_s: float = 0.5
    # None = take upstream GPTArgs.max_wav_length / max_text_length.
    max_duration_s: Optional[float] = None
    max_text_tokens: Optional[int] = None
    duration_tolerance_s: float = 0.05
    # Characters (excluding spaces) per second. Placeholders until ablation A1/A6 measures them; scripts
    # that pack a syllable into one character get their own range.
    cps_bounds: Tuple[float, float] = (4.0, 30.0)
    cps_bounds_by_lang: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"zh-cn": (1.5, 12.0), "zh": (1.5, 12.0), "ja": (1.5, 15.0), "ko": (1.5, 15.0)}
    )
    min_clips_per_speaker: int = 2
    # Coverage warnings (docs/recipe_design.md §3, ablation A1): short clips and single words.
    short_clip_s: float = 1.5
    min_short_fraction: float = 0.05
    # Optional ASR cross-check: max character error rate between transcript and a fresh ASR transcript.
    asr_cer_max: Optional[float] = None
    # Punctuation/pause consistency from stored alignment. [E4]: warnings unless strict_pauses (ablation A6).
    pause_min_gap_s: float = 0.15
    pause_unpunctuated_gap_s: float = 0.3
    strict_pauses: bool = False


@dataclass
class Issue:
    line: int
    audio: str
    code: str
    message: str


@dataclass
class ValidationReport:
    dataset: str
    errors: List[Issue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, record: Optional[DatasetRecord], code: str, message: str) -> None:
        self.errors.append(Issue(record.line if record else 0, record.audio if record else "", code, message))

    def to_json(self) -> Dict[str, Any]:
        return {"dataset": self.dataset, "ok": self.ok, "errors": [asdict(e) for e in self.errors],
                "warnings": self.warnings, "stats": self.stats}

    def raise_for_errors(self, max_listed: int = 50) -> None:
        if self.ok:
            return
        lines = [f"  line {e.line} [{e.code}] {e.audio}: {e.message}" for e in self.errors[:max_listed]]
        more = len(self.errors) - max_listed
        if more > 0:
            lines.append(f"  ... and {more} more")
        raise DatasetValidationError(f"{self.dataset} failed validation with {len(self.errors)} error(s):\n" + "\n".join(lines))


class DatasetValidationError(ValueError):
    pass


def normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace (for overlap and ASR comparisons)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def _cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_for_match(reference).replace(" ", ""), normalize_for_match(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(ref)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer(vocab_path: str, extra_languages: Sequence[str] = ()):
    """The project tokenizer (F4), with every registered extra language that has a [lang] vocab token.

    Languages without one are left unregistered so the record gets its own language_not_in_vocab error.
    """
    from tokenizers import Tokenizer

    from xtts_trainer.frontend.text import ProjectTokenizer

    vocab = Tokenizer.from_file(vocab_path)
    langs = [l for l in extra_languages if vocab.token_to_id(f"[{l.split('-')[0]}]") is not None]
    return ProjectTokenizer(vocab_path, langs)


def validate_dataset(
    dataset_path: str,
    vocab_path: str,
    extra_languages: Sequence[str] = (),
    eval_set_path: Optional[str] = DEFAULT_EVAL_SET,
    reference_lock_path: Optional[str] = DEFAULT_REFERENCE_LOCK,
    target_lock_path: Optional[str] = DEFAULT_TARGET_LOCK,
    config: Optional[ValidationConfig] = None,
    transcriber: Optional[Callable[[str, DatasetRecord], str]] = None,
    require_text: bool = True,
    realigner: Optional[Callable[[str, DatasetRecord], List[str]]] = None,
) -> ValidationReport:
    """Validate a dataset.jsonl. Structural problems raise DatasetFormatError; content problems are collected.

    eval_set_path / reference_lock_path / target_lock_path: leakage checks run against them when they exist
    (target_lock_path: the target speaker's held-out evaluation clips, scripts/build_target_voices.py); the report's
    stats say whether they ran. transcriber(audio_path, record) -> text is used when config.asr_cer_max is set.
    require_text=False validates audio only (Stage 1 DVAE and Stage 3 vocoder don't use transcripts);
    vocab_path may then be None. realigner(audio_path, record) -> problems re-aligns the text to the audio
    (see realignment_checker); any problem is an alignment_mismatch error.
    """
    config = config or ValidationConfig()
    records = load_dataset(dataset_path)
    dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
    report = ValidationReport(dataset=dataset_path)

    max_wav_samples, upstream_max_text = _upstream_training_limits()
    from xtts_trainer.spec import SPEC

    max_duration_s = config.max_duration_s or max_wav_samples / SPEC.gpt_sample_rate
    max_text_tokens = config.max_text_tokens or upstream_max_text

    if require_text:
        tokenizer = _load_tokenizer(vocab_path, [l.lower() for l in extra_languages])
        vocab = tokenizer.tokenizer.get_vocab(with_added_tokens=True)
        unk_id = tokenizer.tokenizer.token_to_id("[UNK]")
    allowed_langs = set(XTTS_BASE_LANGUAGES) | {l.lower() for l in extra_languages}

    eval_texts, reference_hashes = {}, {}
    if eval_set_path and os.path.exists(eval_set_path):
        with open(eval_set_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        sentences = list(eval_data["english_sentences"])
        for sents in eval_data["target_languages"].values():
            sentences.extend(sents)
        eval_texts = {normalize_for_match(s["text"]): s["id"] for s in sentences}
    if reference_lock_path and os.path.exists(reference_lock_path):
        with open(reference_lock_path, "r", encoding="utf-8") as f:
            reference_hashes = {v["clip_sha256"]: k for k, v in json.load(f)["voices"].items()}
    target_hashes = target_holdout_hashes(target_lock_path) if target_lock_path and os.path.exists(target_lock_path) else {}

    audio_hashes: Dict[str, DatasetRecord] = {}
    speakers = Counter(r.speaker for r in records)
    single_ok = {r.speaker for r in records if r.single_clip_speaker}
    durations, n_single_word = [], 0
    pause_counts: Counter = Counter()
    pause_examples: List[str] = []

    for r in records:
        path = r.audio_path(dataset_dir)
        # --- audio
        if not os.path.isfile(path):
            report.add(r, "audio_missing", f"file not found: {path}")
            continue
        try:
            info = sf.info(path)
        except RuntimeError as e:
            report.add(r, "audio_unreadable", str(e))
            continue
        actual_s = info.frames / info.samplerate
        durations.append(actual_s)
        if info.samplerate != r.sr:
            report.add(r, "sr_mismatch", f"declared sr {r.sr}, file has {info.samplerate}")
        if abs(actual_s - r.duration_s) > config.duration_tolerance_s:
            report.add(r, "duration_mismatch", f"declared {r.duration_s:.3f}s, file is {actual_s:.3f}s")
        if actual_s < config.min_duration_s:
            report.add(r, "too_short", f"{actual_s:.2f}s < {config.min_duration_s}s")
        if actual_s > max_duration_s:
            report.add(r, "too_long", f"{actual_s:.2f}s > {max_duration_s:.2f}s (upstream max_wav_length)")

        file_hash = _sha256_file(path)
        if file_hash in audio_hashes:
            report.add(r, "duplicate_audio", f"same audio as line {audio_hashes[file_hash].line}")
        else:
            audio_hashes[file_hash] = r
        if file_hash in reference_hashes:
            report.add(r, "eval_overlap_audio", f"is eval reference voice '{reference_hashes[file_hash]}'")
        if file_hash in target_hashes:
            report.add(r, "target_holdout_overlap",
                       f"is held out for evaluating '{target_hashes[file_hash]}'; train on the dataset_train.jsonl "
                       f"written by scripts/build_target_voices.py")

        if not require_text:
            continue

        # --- text and language
        text = r.text.strip()
        if not text:
            report.add(r, "empty_text", "transcript is empty")
            continue
        if len(text.split()) == 1:
            n_single_word += 1
        lang = r.lang.lower()
        if lang not in allowed_langs:
            report.add(r, "unknown_language", f"'{r.lang}' is not an XTTS base language or in extra_languages")
        elif f"[{lang}]" not in vocab:
            report.add(r, "language_not_in_vocab", f"vocab has no [{lang}] token; extend the vocab first")
        else:
            try:
                tokens = tokenizer.encode(text, lang)
            except NotImplementedError as e:
                report.add(r, "tokenizer_unsupported_language", f"{e} (register it with extra_languages, docs/recipe_design.md F4)")
            else:
                n_unk = sum(1 for t in tokens if t == unk_id)
                if n_unk:
                    report.add(r, "unk_tokens", f"{n_unk} [UNK] token(s): characters missing from the vocab")
                if len(tokens) > max_text_tokens:
                    report.add(r, "text_too_long", f"{len(tokens)} tokens > {max_text_tokens}")

        if eval_texts:
            hit = eval_texts.get(normalize_for_match(text))
            if hit:
                report.add(r, "eval_overlap_text", f"same sentence as eval item {hit}")

        # --- text/audio consistency
        n_chars = len(re.sub(r"\s+", "", normalize_for_match(text)))
        if actual_s > 0 and n_chars:
            lo, hi = config.cps_bounds_by_lang.get(lang, config.cps_bounds)
            cps = n_chars / actual_s
            if not lo <= cps <= hi:
                report.add(r, "chars_per_second", f"{cps:.1f} chars/s outside [{lo}, {hi}]: text and audio likely mismatch")
        if config.asr_cer_max is not None:
            if transcriber is None:
                raise ValueError("config.asr_cer_max is set but no transcriber was given")
            cer = _cer(text, transcriber(path, r))
            if cer > config.asr_cer_max:
                report.add(r, "asr_mismatch", f"CER {cer:.2f} vs ASR > {config.asr_cer_max}")
        words = (r.alignment or {}).get("words")
        if words:
            _check_aligned_words(r, text, words, actual_s, config, report, pause_counts, pause_examples)
        if realigner is not None:
            problems = realigner(path, r)
            if problems:
                report.add(r, "alignment_mismatch", "; ".join(problems))

    # --- speakers
    for r in records:
        if speakers[r.speaker] < config.min_clips_per_speaker and r.speaker not in single_ok:
            report.add(r, "single_clip_speaker",
                       f"speaker '{r.speaker}' has {speakers[r.speaker]} clip(s); set single_clip_speaker to allow it")

    # --- stats and coverage warnings
    n = len(records)
    short = sum(1 for d in durations if d < config.short_clip_s)
    report.stats = {
        "clips": n,
        "hours": round(sum(durations) / 3600, 4),
        "speakers": dict(speakers),
        "languages": dict(Counter(r.lang for r in records)),
        "transcript_sources": dict(Counter(r.transcript_source.split(":")[0] for r in records)),
        "duration_s": {
            "min": round(min(durations), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
            "histogram": dict(Counter(f"{int(d)}-{int(d) + 1}s" for d in durations)),
        },
        "short_clip_fraction": round(short / n, 4) if n else 0.0,
        "single_word_fraction": round(n_single_word / n, 4) if n else 0.0,
        "eval_overlap_checked": bool(eval_texts),
        "reference_overlap_checked": bool(reference_hashes),
        "target_holdout_checked": bool(target_hashes),
        "asr_checked": config.asr_cer_max is not None,
        "realigned": realigner is not None,
        "aligned_clips": sum(1 for r in records if (r.alignment or {}).get("words")),
        "pause_issues": dict(pause_counts),
        "text_checked": require_text,
        "limits": {"min_duration_s": config.min_duration_s, "max_duration_s": round(max_duration_s, 3),
                   "max_text_tokens": max_text_tokens},
    }
    if require_text and n and short / n < config.min_short_fraction:
        report.warnings.append(
            f"only {short}/{n} clips are under {config.short_clip_s}s; short-utterance coverage helps avoid "
            f"short-text hallucination (docs/recipe_design.md §3, ablation A1)"
        )
    if pause_counts and not config.strict_pauses:
        report.warnings.append(
            f"punctuation/pause disagreements {dict(pause_counts)} (e.g. {'; '.join(pause_examples[:3])}); "
            f"[E4] check, set strict_pauses to make it blocking (docs/recipe_design.md A6)"
        )
    if not eval_texts or not reference_hashes:
        report.warnings.append("eval-set leakage checks did not run fully (eval set or reference lock not found)")
    if require_text and any(r.transcript_source.startswith(ASR_SOURCE_PREFIX) for r in records) and config.asr_cer_max is None:
        report.warnings.append("dataset has ASR transcripts but no ASR cross-check was run (set asr_cer_max)")
    return report


def _check_aligned_words(r: DatasetRecord, text: str, words: List[Dict[str, Any]], clip_s: float,
                         config: ValidationConfig, report: ValidationReport, pause_counts: Counter,
                         pause_examples: List[str]) -> None:
    """Stored alignment must spell the transcript and fit the clip; punctuation should match its pauses."""
    from xtts_trainer.prepare.pauses import PauseConfig, pause_issues

    spelled = " ".join(w["w"] for w in words)
    if normalize_for_match(spelled) != normalize_for_match(text):
        report.add(r, "alignment_text_mismatch", f"aligned words {spelled!r} do not spell the transcript")
    if words[0]["start_s"] < -config.duration_tolerance_s or words[-1]["end_s"] > clip_s + config.duration_tolerance_s:
        report.add(r, "alignment_outside_clip",
                   f"aligned words span {words[0]['start_s']}..{words[-1]['end_s']}s, clip is {clip_s:.2f}s")
    issues = pause_issues([(w["w"], w["start_s"], w["end_s"]) for w in words],
                          PauseConfig(config.pause_min_gap_s, config.pause_unpunctuated_gap_s))
    for issue in issues:
        message = f"after '{issue['after']}' gap {issue['gap_s']}s"
        if config.strict_pauses:
            report.add(r, issue["kind"], message)
        else:
            pause_counts[issue["kind"]] += 1
            if len(pause_examples) < 5:
                pause_examples.append(f"line {r.line} {issue['kind']} {message}")


def realignment_checker(aligner=None, quality=None) -> Callable[[str, DatasetRecord], List[str]]:
    """A realigner for validate_dataset: force-align each clip's text to its audio and apply the prep checks.

    This is what catches a cut moved into the neighbouring words: the extra speech has no text to align to.
    """
    from xtts_trainer.prepare.align import AlignmentError, CtcAligner
    from xtts_trainer.prepare.pipeline import load_mono
    from xtts_trainer.prepare.segment import Segment, segment_problems
    from xtts_trainer.spec import SPEC

    aligner = aligner or CtcAligner()

    def check(path: str, record: DatasetRecord) -> List[str]:
        audio = load_mono(path, SPEC.output_sample_rate)
        try:
            window = aligner.align(audio, SPEC.output_sample_rate, record.text)
        except AlignmentError as e:
            return [str(e)]
        clip_s = len(audio) / SPEC.output_sample_rate
        segment = Segment(0.0, clip_s, window.words)
        return segment_problems(segment, window.unexplained_speech_s(0.0, clip_s), quality)

    return check


def load_validated_dataset(
    dataset_path: str,
    vocab_path: Optional[str],
    extra_languages: Sequence[str] = (),
    require_text: bool = True,
    logger=None,
    **kwargs,
) -> Tuple[List[DatasetRecord], str, ValidationReport]:
    """What every stage calls before training: validate, refuse on any error, log warnings.

    Returns (records, dataset_dir, report).
    """
    report = validate_dataset(dataset_path, vocab_path, extra_languages, require_text=require_text, **kwargs)
    report.raise_for_errors()
    for warning in report.warnings:
        (logger.warning if logger else print)(f"Dataset warning: {warning}")
    return load_dataset(dataset_path), os.path.dirname(os.path.abspath(dataset_path)), report


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Validate a dataset.jsonl before training.")
    parser.add_argument("dataset")
    parser.add_argument("--vocab", default=None, help="Tokenizer vocab.json (base or extended); required unless --audio_only")
    parser.add_argument("--audio_only", action="store_true", help="Skip transcript checks (Stage 1 / Stage 3 data)")
    parser.add_argument("--languages", nargs="+", default=[], help="Registered extra language codes")
    parser.add_argument("--eval_set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--reference_lock", default=DEFAULT_REFERENCE_LOCK)
    parser.add_argument("--report", default=None, help="Write the JSON report here")
    parser.add_argument("--strict_pauses", action="store_true", help="Punctuation/pause disagreements are errors")
    parser.add_argument("--realign", action="store_true",
                        help="Re-align every clip's text to its audio with MMS_FA (1.26 GB download, CC BY-NC 4.0)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if not args.audio_only and not args.vocab:
        parser.error("--vocab is required unless --audio_only")
    realigner = None
    if args.realign:
        from xtts_trainer.prepare.align import CtcAligner

        realigner = realignment_checker(CtcAligner(device=args.device))
    report = validate_dataset(args.dataset, args.vocab, args.languages, args.eval_set, args.reference_lock,
                              config=ValidationConfig(strict_pauses=args.strict_pauses),
                              require_text=not args.audio_only, realigner=realigner)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report.to_json(), f, indent=2, ensure_ascii=False)
    for issue in report.errors:
        print(f"ERROR line {issue.line} [{issue.code}] {issue.audio}: {issue.message}")
    for warning in report.warnings:
        print(f"WARNING {warning}")
    print(json.dumps(report.stats, indent=2, ensure_ascii=False))
    print("VALID" if report.ok else f"INVALID ({len(report.errors)} errors)")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
