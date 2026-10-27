#!/usr/bin/env python3
"""Held-out validation of the alignment rejection thresholds (QualityConfig), on speakers and text not used to set them.

The thresholds were calibrated in-sample on 8 recordings (docs/evidence/phase6, [E5]). This script measures them,
frozen, on held-out speakers, so the clean false-positive rate and the detection rates are out-of-sample.
The analysis is pre-registered in docs/evidence/phase6_heldout/README.md; commit that file and this script before
the run.

What it measures, per clean clip:
- clean acceptance: preparation keeps the segment, and re-alignment finds no problem (the validator's check);
- a mutation suite with labels defined by speech actually removed or added, relative to the aligned words (each clip
  keeps edge_pad_s of silence at both ends, so a cut measured from the clip edge first removes only silence):
  controls that only trim silence (must pass), speech cuts at the tail and head, transcript insertions, and
  spillover of the next utterance's speech;
- per-word scores and unexplained speech for every clean and mutated clip, so threshold trade-offs can be studied.

Sources:
- CMU ARCTIC rms (US English) and jmk (Canadian English): neither speaker is in the calibration set; utterances
  a0003 onwards (a0001/a0002 were the calibration voices' text); transcripts from the corpus's own prompt file.
- LibriVox French and German readings: transcripts must be checked against the audio before a run counts. Run
  `--check_transcripts` first: it prints Whisper's transcript of each window next to the given one. After listening,
  set `transcript_verified: True` for that source. A run refuses unverified sources unless
  `--allow_unverified_transcripts` is passed, and then records that in the output.

Usage:
    python scripts/validate_alignment_heldout.py --check_transcripts      # once, then verify by listening
    python scripts/validate_alignment_heldout.py --device cuda             # the run (locked environment)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xtts_trainer.prepare.segment import QualityConfig, SegmentConfig  # noqa: E402
from xtts_trainer.spec import SPEC  # noqa: E402

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "phase6_heldout"
DEFAULT_OUTPUT = EVIDENCE_DIR / "heldout_alignment_validation.json"
PINS_PATH = EVIDENCE_DIR / "source_pins.json"
DEFAULT_VOCAB = REPO_ROOT / "xtts_base" / "vocab.json"
RATE = SPEC.output_sample_rate

# Frozen from the in-sample calibration (docs/evidence/phase6). Not to be changed during or after this run.
FROZEN_QUALITY = QualityConfig(min_word_score=0.05, weak_word_score=0.20, max_weak_words=3,
                               max_unexplained_speech_s=0.10)
FROZEN_SEGMENT = dict(min_gap_s=0.15, edge_pad_s=0.20)

# Pre-registered suite sizes
ARCTIC_UTTERANCES = [f"arctic_a{n:04d}" for n in range(3, 23)]  # 20 per speaker, a0001/a0002 excluded
MUTATED_CLIPS_PER_SOURCE = 5
TAIL_SPEECH_CUTS_S = (0.05, 0.15, 0.30, 0.60)
HEAD_SPEECH_CUTS_S = (0.15, 0.30, 0.60)
SPILLOVER_S = {"spillover_0.05s": (0.05, True), "spillover_0.50s": (0.50, False)}
MIN_REMAINING_S = 0.5
INSERTION_WORDS = {"en": ("Moreover", "extraordinary"), "fr": ("Cependant", "extraordinaire"),
                   "de": ("Jedoch", "außergewöhnlich")}

HELDOUT_SOURCES: List[Dict[str, Any]] = [
    {
        "id": "arctic_rms", "lang": "en", "kind": "arctic",
        "description": "CMU ARCTIC rms (US English male)",
        "url": "http://festvox.org/cmu_arctic/packed/cmu_us_rms_arctic.tar.bz2", "corpus_speaker": "rms",
        "license": "CMU ARCTIC license (licenses/rms_COPYING, downloaded and pinned by the run)",
    },
    {
        "id": "arctic_jmk", "lang": "en", "kind": "arctic",
        "description": "CMU ARCTIC jmk (Canadian English male)",
        "url": "http://festvox.org/cmu_arctic/packed/cmu_us_jmk_arctic.tar.bz2", "corpus_speaker": "jmk",
        "license": "CMU ARCTIC license (licenses/jmk_COPYING, downloaded and pinned by the run)",
    },
    {
        "id": "librivox_fr_lafontaine", "lang": "fr", "kind": "recording",
        "description": "LibriVox, Fables de La Fontaine, opening announcement (French)",
        "url": "https://archive.org/download/fables_lafontaine_01_librivox/fables_01_00_lafontaine_64kb.mp3",
        # Ends mid-pause (14.12-14.80 s) after "librivox point org", before the reader credit ("Enregistré par ...")
        "license": "Public domain (archive.org licenseurl; licenses/librivox_archive_org_license.md)",
        "max_seconds": 14.46,
        "transcript": ("Ceci est un enregistrement LibriVox. Tous nos enregistrements appartiennent au domaine public. "
                       "Pour vous renseigner à notre sujet ou pour participer, rendez-vous sur librivox point org."),
        "transcript_verified": True,
    },
    {
        "id": "librivox_de_grimm", "lang": "de", "kind": "recording",
        "description": "LibriVox, Märchen der Gebrüder Grimm, Spindel, Weberschiffchen und Nadel: announcement (German)",
        # grimm_114 (the first draft's choice) has a noise floor 15 dB below peak and no pauses, so the reader
        # credit can't be cut off; chapter 214 is clean (floor -64 dB). Ends mid-pause (11.9-15.6 s) before the title.
        "url": "https://archive.org/download/grimm_maerchen_1_librivox/grimm_214_spindel_weberschiffchen_und_nadel_64kb.mp3",
        "license": "Public domain (archive.org licenseurl; licenses/librivox_archive_org_license.md)",
        "max_seconds": 13.75,
        "transcript": ("Dies ist eine LibriVox-Aufnahme. Alle LibriVox-Aufnahmen sind lizenzfrei und in öffentlichem "
                       "Besitz. Weitere Informationen und Hinweise zur Beteiligung an diesem Projekt gibt es bei "
                       "librivox point org."),
        "transcript_verified": True,
    },
]


# --------------------------------------------------------------------------- provenance

def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pins(path: Optional[Path] = None) -> Dict[str, str]:
    path = path or PINS_PATH  # resolved at call time, so reading and writing use the same file
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_pin(url: str, digest: str, pins: Dict[str, str]) -> str:
    """'verified' when the download matches its pin, 'recorded' when it has none yet; raises on a mismatch."""
    if url not in pins:
        return "recorded"
    if pins[url] != digest:
        raise RuntimeError(f"{url} has SHA-256 {digest}, but its pin in {PINS_PATH} is {pins[url]}")
    return "verified"


def download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers={"User-Agent": "open-xtts-trainer/heldout-validator"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest + ".part", "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
        os.replace(dest + ".part", dest)
    return dest


def locked_versions(lock_path: Path = REPO_ROOT / "uv.lock", names=("torch", "torchaudio")) -> Dict[str, str]:
    text = lock_path.read_text(encoding="utf-8")
    found = {}
    for name in names:
        m = re.search(rf'\[\[package\]\]\nname = "{re.escape(name)}"\nversion = "([^"]+)"', text)
        if m:
            found[name] = m.group(1)
    return found


def environment_problems(installed: Dict[str, str], locked: Dict[str, str]) -> List[str]:
    """Installed versions (local build tags like +cu130 ignored) that differ from uv.lock."""
    return [f"{name} {installed.get(name)} installed, {want} locked" for name, want in locked.items()
            if (installed.get(name) or "").split("+")[0] != want.split("+")[0]]


# --------------------------------------------------------------------------- transcripts

def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(1, len(ref))


# --------------------------------------------------------------------------- mutations (pure, unit-tested)

def word_bounds(words: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Start of the first and end of the last aligned word, in clip seconds."""
    return float(words[0]["start_s"]), float(words[-1]["end_s"])


def build_mutations(clip: np.ndarray, words: List[Dict[str, Any]], text: str, lang: str,
                    next_speech: Optional[np.ndarray], rate: int = RATE) -> List[Dict[str, Any]]:
    """The mutation suite for one clean clip. Every label is defined by speech removed or added, measured from the
    aligned words, not from the clip edges (which carry edge_pad_s of silence)."""
    first, last = word_bounds(words)
    n = len(clip)
    s = lambda t: int(round(t * rate))  # noqa: E731
    out = []

    def add(name, audio, mutated_text, expected_pass, kind, speech_removed_s=0.0, speech_added_s=0.0):
        out.append(dict(name=name, audio=audio, text=mutated_text, expected_pass=expected_pass, kind=kind,
                        speech_removed_s=round(speech_removed_s, 3), speech_added_s=round(speech_added_s, 3)))

    # Controls: trim silence only, keeping 50 ms beyond the words; nothing was lost, so they must pass
    add("control_trim_tail_silence", clip[: min(n, s(last + 0.05))], text, True, "control")
    add("control_trim_head_silence", clip[max(0, s(first - 0.05)):], text, True, "control")
    for cut in TAIL_SPEECH_CUTS_S:
        if last - cut - first >= MIN_REMAINING_S:
            add(f"tail_speech_cut_{int(cut * 1000)}ms", clip[: s(last - cut)], text, False, "tail_cut", cut)
    for cut in HEAD_SPEECH_CUTS_S:
        if last - (first + cut) >= MIN_REMAINING_S:
            add(f"head_speech_cut_{int(cut * 1000)}ms", clip[s(first + cut):], text, False, "head_cut", cut)
    head_word, content_word = INSERTION_WORDS[lang]
    tokens = text.split()
    mid = max(1, len(tokens) // 2)
    add("insert_head_word", clip, f"{head_word} {text}", False, "insertion")
    add("insert_mid_word", clip, " ".join(tokens[:mid] + [content_word] + tokens[mid:]), False, "insertion")
    add("insert_tail_word", clip, f"{text} {content_word}", False, "insertion")
    if next_speech is not None:
        tail = clip[: min(n, s(last + 0.05))]
        for name, (secs, expected) in SPILLOVER_S.items():
            extra = next_speech[: s(secs)]
            if len(extra) == s(secs):
                add(name, np.concatenate([tail, extra]), text, expected, "spillover", speech_added_s=secs)
    return out


# --------------------------------------------------------------------------- alignment checks

def realign(aligner, audio: np.ndarray, text: str, quality: QualityConfig) -> Dict[str, Any]:
    """The validator's re-alignment check (validate.realignment_checker), keeping the per-word scores."""
    from xtts_trainer.prepare.align import AlignmentError
    from xtts_trainer.prepare.segment import Segment, segment_problems

    clip_s = len(audio) / RATE
    try:
        window = aligner.align(audio, RATE, text)
    except AlignmentError as e:
        return {"problems": [str(e)], "scores": [], "unexplained_speech_s": None}
    unexplained = window.unexplained_speech_s(0.0, clip_s)
    return {"problems": segment_problems(Segment(0.0, clip_s, window.words), unexplained, quality),
            "scores": [round(float(w.score), 4) for w in window.words],
            "unexplained_speech_s": round(float(unexplained), 4)}


def clean_clips(source: Dict[str, Any], work_dir: str, aligner, pins: Dict[str, str]) -> Dict[str, Any]:
    """Prepare every clean clip of one source with the frozen config; returns clips plus provenance."""
    from xtts_trainer.prepare import PrepConfig, prepare_recording
    from xtts_trainer.prepare.pipeline import load_mono

    src_dir = os.path.join(work_dir, source["id"])
    archive = download(source["url"], os.path.join(src_dir, os.path.basename(source["url"])))
    digest = sha256_file(archive)
    provenance = {"url": source["url"], "sha256": digest, "pin": check_pin(source["url"], digest, pins)}
    recordings = []  # (name, path, transcript, max_seconds)
    if source["kind"] == "arctic":
        from eval.dataset import CMU_ARCTIC_LICENSE_URL
        from eval.reference_voices import read_from_tarball

        # Each speaker's own license, as the reference voices do; no license, no clips
        license_url = CMU_ARCTIC_LICENSE_URL.format(speaker=source["corpus_speaker"])
        license_path = download(license_url, str(EVIDENCE_DIR / "licenses" / f"{source['corpus_speaker']}_COPYING"))
        license_sha = sha256_file(license_path)
        provenance["license"] = {"url": license_url, "sha256": license_sha, "pin": check_pin(license_url, license_sha, pins)}

        arrays, sample_rate, transcripts = read_from_tarball(archive, source["corpus_speaker"], ARCTIC_UTTERANCES)
        for utt, audio in zip(ARCTIC_UTTERANCES, arrays):
            wav = os.path.join(src_dir, f"{utt}.wav")
            sf.write(wav, audio, sample_rate)
            recordings.append((utt, wav, transcripts[utt], None))
        provenance["transcripts"] = "corpus prompt file etc/txt.done.data"
    else:
        recordings.append((source["id"], archive, source["transcript"], source.get("max_seconds")))
        provenance["transcripts"] = "verified by listening" if source["transcript_verified"] else "UNVERIFIED"

    seg_cfg = SegmentConfig(min_s=1.0, max_s=11.0, **FROZEN_SEGMENT)
    clips = []
    for name, path, transcript, max_seconds in recordings:
        out = os.path.join(src_dir, "prepared", name)
        records, report = prepare_recording(path, out, lang=source["lang"], speaker=source["id"], transcript=transcript,
                                            aligner=aligner, config=PrepConfig(segment=seg_cfg, quality=FROZEN_QUALITY),
                                            max_seconds=max_seconds)
        candidates = report.get("candidates", [])
        rejected = [c for c in candidates if not c.get("kept")]
        for c in rejected:  # dropped by preparation: a clean clip rejected, i.e. a false positive
            clips.append({"recording": name, "text": c["text"], "prep_kept": False, "prep_problems": c.get("problems"),
                          "realign": None, "accepted": False, "audio": None, "words": None})
        for r in records:
            audio = load_mono(os.path.join(out, r.audio), RATE)
            check = realign(aligner, audio, r.text, FROZEN_QUALITY)
            clips.append({"recording": name, "text": r.text, "prep_kept": True, "prep_problems": [],
                          "realign": check, "accepted": not check["problems"], "audio": audio,
                          "words": r.alignment["words"]})
    return {"provenance": provenance, "clips": clips}


# --------------------------------------------------------------------------- run

def check_transcripts(work_dir: str, device: str) -> None:
    """Print Whisper's timestamped transcript around each LibriVox window next to the given one, for a listener.

    Whisper reads 8 s past the window, with timestamps: on a window cut exactly at a pause it can drop a whole
    sentence (observed on the German window at 13.75 s, not at 12.6 s or 24 s), and the extra seconds show what the
    window excludes. CER is computed on the segments that end inside the window.
    """
    from xtts_trainer.asr import _generate_kwargs, default_asr_model, load_asr
    from xtts_trainer.prepare.pipeline import load_mono

    for src in HELDOUT_SOURCES:
        if src["kind"] != "recording":
            continue
        path = download(src["url"], os.path.join(work_dir, src["id"], os.path.basename(src["url"])))
        end = src["max_seconds"]
        audio = load_mono(path, RATE)[: int((end + 8.0) * RATE)]  # the pipeline resamples for Whisper
        model = default_asr_model(src["lang"])
        chunks = load_asr(model, device)({"raw": audio, "sampling_rate": RATE}, return_timestamps=True,
                                         generate_kwargs=_generate_kwargs(model, src["lang"]))["chunks"]
        inside = " ".join(c["text"].strip() for c in chunks if c["timestamp"][1] is not None and c["timestamp"][1] <= end)
        print(f"\n== {src['id']} (window 0-{end} s; CER of the window vs given: {cer(src['transcript'], inside):.3f})")
        print(f"given  : {src['transcript']}")
        for c in chunks:
            where = "inside " if c["timestamp"][1] is not None and c["timestamp"][1] <= end else "OUTSIDE"
            print(f"  {where} {c['timestamp']}: {c['text'].strip()}")
    print("\nListen to each window; correct the transcript if needed and set transcript_verified: True.")


def run(device: str, output: Path, work_dir: str, allow_unverified: bool, allow_unlocked_env: bool) -> Dict[str, Any]:
    import torch
    import torchaudio

    from xtts_trainer.data import DatasetRecord, validate_dataset, write_dataset
    from xtts_trainer.data.validate import realignment_checker
    from xtts_trainer.prepare import CtcAligner

    env = {"torch": torch.__version__, "torchaudio": torchaudio.__version__}
    env_issues = environment_problems(env, locked_versions())
    if env_issues and not allow_unlocked_env:
        raise SystemExit(f"not the locked environment ({'; '.join(env_issues)}); install with `uv sync --frozen`")
    unverified = [s["id"] for s in HELDOUT_SOURCES if s["kind"] == "recording" and not s["transcript_verified"]]
    if unverified and not allow_unverified:
        raise SystemExit(f"transcripts not verified for {unverified}; run --check_transcripts and listen first")
    if not DEFAULT_VOCAB.exists():
        raise SystemExit(f"{DEFAULT_VOCAB} missing: run scripts/fetch_base_artifacts.py first")

    aligner = CtcAligner(device=device)
    pins = load_pins()
    results = []
    for src in HELDOUT_SOURCES:
        print(f"\n== {src['id']} ({src['lang']})")
        prepared = clean_clips(src, work_dir, aligner, pins)
        clips = prepared["clips"]
        kept = [c for c in clips if c["prep_kept"]]
        # Production check on the kept clips too: the validator with re-alignment, as training would run it
        for i, c in enumerate(kept):
            d = os.path.join(work_dir, src["id"], "validate", f"clip_{i:03d}")
            os.makedirs(d, exist_ok=True)
            sf.write(os.path.join(d, "clip.wav"), c["audio"], RATE, subtype="PCM_16")
            write_dataset(os.path.join(d, "dataset.jsonl"), [DatasetRecord(
                audio="clip.wav", text=c["text"], lang=src["lang"], speaker=src["id"],
                duration_s=round(len(c["audio"]) / RATE, 4), sr=RATE, transcript_source="human",
                single_clip_speaker=True)])
            report = validate_dataset(os.path.join(d, "dataset.jsonl"), str(DEFAULT_VOCAB), eval_set_path=None,
                                      reference_lock_path=None, target_lock_path=None,
                                      realigner=realignment_checker(aligner, FROZEN_QUALITY))
            c["validator_ok"] = report.ok
            c["validator_errors"] = [f"{e.code}: {e.message}" for e in report.errors]
        mutations = []
        for i, c in enumerate(kept[:MUTATED_CLIPS_PER_SOURCE]):
            nxt = kept[i + 1] if i + 1 < len(kept) else None
            next_speech = None
            if nxt is not None:
                start = int(round(word_bounds(nxt["words"])[0] * RATE))
                next_speech = nxt["audio"][start:]
            for m in build_mutations(c["audio"], c["words"], c["text"], src["lang"], next_speech):
                check = realign(aligner, m["audio"], m["text"], FROZEN_QUALITY)
                actual_pass = not check["problems"]
                mutations.append({k: v for k, v in m.items() if k != "audio"} | {
                    "clip": i, "actual_pass": actual_pass, "correct": actual_pass == m["expected_pass"], **check})
        results.append({
            "id": src["id"], "lang": src["lang"], "description": src["description"], "license": src["license"],
            "provenance": prepared["provenance"],
            "clean": [{k: v for k, v in c.items() if k != "audio"} for c in clips],
            "mutations": mutations,
        })
        n_ok = sum(c["accepted"] for c in clips)
        print(f"   clean accepted {n_ok}/{len(clips)}; mutations correct "
              f"{sum(m['correct'] for m in mutations)}/{len(mutations)}")

    report = {
        "format": "open-xtts-heldout-alignment-validation", "version": 1,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "environment": env | {"python": sys.version.split()[0], "device": device,
                              "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                              "locked": locked_versions(), "environment_issues": env_issues},
        "unverified_transcripts_allowed": bool(unverified),
        "frozen_quality_config": FROZEN_QUALITY.__dict__, "frozen_segment_config": FROZEN_SEGMENT,
        "results": results,
        "summary": summarize(results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    new_pins = {r["provenance"]["url"]: r["provenance"]["sha256"] for r in results}
    new_pins |= {r["provenance"]["license"]["url"]: r["provenance"]["license"]["sha256"]
                 for r in results if "license" in r["provenance"]}
    if any(url not in pins for url in new_pins):
        PINS_PATH.write_text(json.dumps(pins | new_pins, indent=2) + "\n", encoding="utf-8")
        print(f"recorded new download pins in {PINS_PATH}; commit them so later runs verify against them")
    print(f"\nwritten: {output}")
    print(json.dumps(report["summary"], indent=2))
    return report


def clopper_pearson(k: int, n: int, level: float = 0.95) -> List[Optional[float]]:
    """Exact binomial interval for k successes out of n."""
    from scipy.stats import beta

    if n == 0:
        return [None, None]
    a = (1 - level) / 2
    low = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [round(low, 4), round(high, 4)]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered summary: clean acceptance overall and per language, and rates per mutation."""
    clean = [(r["lang"], c) for r in results for c in r["clean"]]
    by_lang: Dict[str, List[int]] = {}
    for lang, c in clean:
        by_lang.setdefault(lang, [0, 0])
        by_lang[lang][0] += c["accepted"]
        by_lang[lang][1] += 1
    by_mutation: Dict[str, List[int]] = {}
    for r in results:
        for m in r["mutations"]:
            slot = by_mutation.setdefault(m["name"], [0, 0, m["expected_pass"]])
            slot[0] += m["actual_pass"]
            slot[1] += 1
    accepted = sum(c["accepted"] for _, c in clean)
    return {
        "clean_accepted": f"{accepted}/{len(clean)}",
        "clean_accepted_ci95": clopper_pearson(accepted, len(clean)),
        "clean_accepted_by_language": {k: {"accepted": f"{a}/{n}", "ci95": clopper_pearson(a, n)}
                                       for k, (a, n) in sorted(by_lang.items())},
        "mutations": {k: {"passed": f"{p}/{n}", "expected": "pass" if e else "reject",
                          "rejected_ci95": clopper_pearson(n - p, n)}
                      for k, (p, n, e) in sorted(by_mutation.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work_dir", default=os.path.join("/tmp", "xtts_heldout_alignment"))
    parser.add_argument("--check_transcripts", action="store_true", help="print Whisper vs given transcripts and stop")
    parser.add_argument("--allow_unverified_transcripts", action="store_true")
    parser.add_argument("--allow_unlocked_env", action="store_true", help="for development only; recorded in output")
    args = parser.parse_args()
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.check_transcripts:
        check_transcripts(args.work_dir, args.device)
    else:
        run(args.device, args.output, args.work_dir, args.allow_unverified_transcripts, args.allow_unlocked_env)


if __name__ == "__main__":
    main()
