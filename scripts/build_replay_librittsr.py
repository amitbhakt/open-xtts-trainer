"""Build M1's same-language replay set from LibriTTS-R dev-clean (docs/evidence/m1_stage2_preservation/PROTOCOL.md).

Replay for speaker adaptation (candidate C2) must come from other English speakers, recorded unlike the five ARCTIC
preservation voices, so that a pass can't come from familiarity with their recording setup. LibriTTS-R (Koizumi
et al. 2023; https://www.openslr.org/141/) is a restored, multi-speaker audiobook corpus at 24 kHz.

Selection, fixed before any audio was looked at:
- archives: dev_clean.tar.gz and doc.tar.gz, each checked against the MD5 OpenSLR publishes and a SHA-256 pin;
- speakers: dev-clean readers (speakers.tsv) minus EXCLUDED_READERS; female and male ids, each sorted and shuffled
  with SELECTION_SEED into a queue; slots are filled alternately (female first) until N_SPEAKERS, each from the
  front of its gender's queue;
- clips: each reader's utterances, sorted and shuffled with a per-reader seed; the first CLIPS_PER_SPEAKER that
  last MIN_S to MAX_S and pass the project's dataset validator (vocabulary, eval-set sentences, reference-voice
  audio, duration rules). A reader with fewer is skipped (LibriTTS-R drops noisy lines, so a few readers have
  little audio) and the next reader of the same gender takes the slot. Skipped readers and rejected candidates
  are recorded with their reasons.

Writes <output_dir>/dataset.jsonl, the clips, and replay_selection.json (pins, license, rule, every choice).

    python scripts/build_replay_librittsr.py --output_dir runs/replay_librittsr \\
        [--evidence docs/evidence/m1_stage2_preservation/replay_selection.json]
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import soundfile as sf

from xtts_trainer.data import DatasetRecord, validate_dataset, write_dataset
from xtts_trainer.prepare.pipeline import load_mono
from xtts_trainer.spec import SPEC

URL_BASE = "https://www.openslr.org/resources/141/"
# MD5: OpenSLR's published md5sum.txt (fetched 2026-10-31). SHA-256: the files whose MD5 matched it.
ARCHIVES = {
    "doc.tar.gz": {"md5": "dad90f1b357625ce93d695b930977f7f", "sha256": "b59b03462911e559a4ab84f7229752b3578b715d534a31d17b88054eb3cc7389"},
    "dev_clean.tar.gz": {"md5": "2c1f5312914890634cc2d15783032ff3", "sha256": "02ea52de47ca670d7ad86e714a6c58b207660a3f8a371ad102156776d2262a0c"},
}
LICENSE = ("CC BY 4.0: \"The LibriTTS-R corpus is made available by Google LLC under a Creative Commons Attribution "
           "4.0 International License.\" (LICENSE.txt in doc.tar.gz)")
CITATION = ("Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani, Y. Zhang, W. Han and "
            "A. Bapna, \"LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus,\" arXiv, 2023.")
# Readers never selected, with why. 209 is in train-clean-360, not dev-clean; listed so the rule holds whichever
# subset is used.
EXCLUDED_READERS = {"209": "Moira Fogarty, the M1 target speaker"}
SUBSET = "dev-clean"
SELECTION_SEED = 20261031
N_SPEAKERS = 15
CLIPS_PER_SPEAKER = 5
MIN_S, MAX_S = 3.0, 10.0
UTTERANCE = re.compile(r"^LibriTTS_R/dev-clean/(\d+)/(\d+)/(\d+_\d+_\d+_\d+)\.(wav|normalized\.txt)$")


def file_digests(path: str) -> Tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def fetch(name: str, cache_dir: str, url_base: str = URL_BASE) -> str:
    """Download once into cache_dir; the file must match the published MD5 and the SHA-256 pin."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        req = urllib.request.Request(url_base + name, headers={"User-Agent": "open-xtts-trainer"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(path + ".part", "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
        os.replace(path + ".part", path)
    md5, sha = file_digests(path)
    want = ARCHIVES[name]
    if md5 != want["md5"]:
        raise RuntimeError(f"{path}: MD5 {md5} is not the published {want['md5']}; delete it and retry")
    if sha != want["sha256"]:
        raise RuntimeError(f"{path}: SHA-256 {sha} is not the pinned {want['sha256']}")
    return path


def read_speakers(doc_tar: str, subset: str = SUBSET) -> Dict[str, str]:
    """reader id -> gender (F/M) for one subset, from speakers.tsv."""
    with tarfile.open(doc_tar, "r:gz") as tar:
        member = tar.getmember("LibriTTS_R/speakers.tsv")
        text = tar.extractfile(member).read().decode("utf-8")
    speakers = {}
    for line in text.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 3 and cols[2].strip() == subset:
            speakers[cols[0].strip()] = cols[1].strip()
    if not speakers:
        raise ValueError(f"no {subset} readers in speakers.tsv")
    return speakers


def reader_queues(speakers: Dict[str, str], seed: Optional[int] = None,
                  excluded: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """gender -> reader ids in selection order (sorted, then shuffled with the seed), excluded readers removed."""
    # Defaults resolved at call time, so the module constants are the single source
    excluded = EXCLUDED_READERS if excluded is None else excluded
    rng = random.Random(SELECTION_SEED if seed is None else seed)
    queues = {}
    for g in ("F", "M"):
        ids = sorted((s for s, sg in speakers.items() if sg == g and s not in excluded), key=int)
        rng.shuffle(ids)
        queues[g] = ids
    return queues


def index_and_extract(tar_path: str, speakers: List[str], dest: str) -> Dict[str, List[str]]:
    """Extract the chosen speakers' utterances (wav + normalized text) into dest; return speaker -> utterance ids.

    Only regular files whose names match UTTERANCE are read, and each is written to dest/<utterance>.<ext>, so an
    archive member can't write outside dest.
    """
    wanted = set(speakers)
    utts: Dict[str, set] = {s: set() for s in speakers}
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            m = UTTERANCE.match(member.name)
            if not m or not member.isreg() or m.group(1) not in wanted:
                continue
            utt, ext = m.group(3), m.group(4)
            with open(os.path.join(dest, f"{utt}.{ext}"), "wb") as out:
                out.write(tar.extractfile(member).read())
            if ext == "wav":
                utts[m.group(1)].add(utt)
    return {s: sorted(u) for s, u in utts.items()}


def candidate_order(utterances: List[str], speaker: str) -> List[str]:
    order = sorted(utterances)
    random.Random(f"{SELECTION_SEED}:{speaker}").shuffle(order)
    return order


def usable_clips(spk: str, utts: List[str], raw: str, output_dir: str, vocab_path: str
                 ) -> Tuple[List[DatasetRecord], List[Dict[str, Any]]]:
    """The first CLIPS_PER_SPEAKER clips of one reader, in its seeded order, that pass the duration rule and the
    validator; and every candidate rejected on the way. Audio of clips not chosen is removed."""
    rate = SPEC.output_sample_rate
    rejected: List[Dict[str, Any]] = []
    candidates: List[DatasetRecord] = []
    for utt in candidate_order(utts, spk):
        # Enough candidates that the validator can drop some and still leave CLIPS_PER_SPEAKER
        if len(candidates) >= 3 * CLIPS_PER_SPEAKER:
            break
        info = sf.info(os.path.join(raw, f"{utt}.wav"))
        duration = info.frames / info.samplerate
        if not MIN_S <= duration <= MAX_S:
            rejected.append({"utterance": utt, "speaker": spk, "reason": f"duration {duration:.2f} s"})
            continue
        with open(os.path.join(raw, f"{utt}.normalized.txt"), encoding="utf-8") as f:
            text = " ".join(f.read().split())
        audio = load_mono(os.path.join(raw, f"{utt}.wav"), rate)
        rel = f"clips/librittsr_{utt}.wav"
        sf.write(os.path.join(output_dir, rel), audio, rate, subtype="PCM_16")
        candidates.append(DatasetRecord(
            audio=rel, text=text, lang="en", speaker=f"librittsr_{spk}", duration_s=round(len(audio) / rate, 4),
            sr=rate, transcript_source="dataset", source=f"LibriTTS-R {SUBSET} {utt}"))
    chosen: List[DatasetRecord] = []
    if candidates:
        cand_path = write_dataset(os.path.join(output_dir, f"candidates_{spk}.jsonl"), candidates)
        bad: Dict[str, List[str]] = {}
        for e in validate_dataset(cand_path, vocab_path).errors:  # keyed by audio path, unlike line numbers
            bad.setdefault(e.audio, []).append(f"{e.code}: {e.message}")
        os.remove(cand_path)
        for rec in candidates:
            if rec.audio in bad:
                rejected.append({"utterance": rec.source.split()[-1], "speaker": spk,
                                 "reason": "; ".join(bad[rec.audio])})
            elif len(chosen) < CLIPS_PER_SPEAKER:
                chosen.append(rec)
    for rec in candidates:
        if rec not in chosen or len(chosen) < CLIPS_PER_SPEAKER:
            os.remove(os.path.join(output_dir, rec.audio))
    return (chosen if len(chosen) == CLIPS_PER_SPEAKER else []), rejected


def build(output_dir: str, cache_dir: str, vocab_path: str, url_base: str = URL_BASE) -> Dict[str, Any]:
    doc = fetch("doc.tar.gz", cache_dir, url_base)
    dev = fetch("dev_clean.tar.gz", cache_dir, url_base)
    genders = read_speakers(doc)
    queues = reader_queues(genders)
    raw = os.path.join(cache_dir, "dev_clean_extracted")
    utts = index_and_extract(dev, queues["F"] + queues["M"], raw)
    os.makedirs(os.path.join(output_dir, "clips"), exist_ok=True)

    speakers: List[str] = []
    chosen: List[DatasetRecord] = []
    rejected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    slot = 0
    while len(speakers) < N_SPEAKERS:
        gender = ("F", "M")[slot % 2]
        if not queues[gender]:
            raise ValueError(f"no {gender} reader left with {CLIPS_PER_SPEAKER} usable clips; "
                             f"{len(speakers)} of {N_SPEAKERS} chosen")
        spk = queues[gender].pop(0)
        clips, rej = usable_clips(spk, utts[spk], raw, output_dir, vocab_path)
        rejected += rej
        if not clips:
            skipped.append({"reader": spk, "gender": gender,
                            "reason": f"fewer than {CLIPS_PER_SPEAKER} usable clips ({len(utts[spk])} utterances)"})
            continue
        speakers.append(spk)
        chosen += clips
        slot += 1

    dataset = write_dataset(os.path.join(output_dir, "dataset.jsonl"), chosen)
    final = validate_dataset(dataset, vocab_path)
    if not final.ok:
        raise ValueError(f"the selected replay set fails validation: {[e.message for e in final.errors]}")

    def sha256(rel):
        return file_digests(os.path.join(output_dir, rel))[1]
    selection = {
        "corpus": "LibriTTS-R", "subset": SUBSET, "url_base": url_base, "license": LICENSE, "citation": CITATION,
        "archives": {n: dict(v) for n, v in ARCHIVES.items()},
        "rule": {"seed": SELECTION_SEED, "n_speakers": N_SPEAKERS, "clips_per_speaker": CLIPS_PER_SPEAKER,
                 "duration_s": [MIN_S, MAX_S], "excluded_readers": EXCLUDED_READERS},
        "speakers": [{"reader": s, "gender": genders[s]} for s in speakers],
        "skipped_readers": skipped,
        "clips": [{"utterance": r.source.split()[-1], "speaker": r.speaker, "duration_s": r.duration_s,
                   "text": r.text, "sha256": sha256(r.audio)} for r in chosen],
        "rejected": rejected,
    }
    with open(os.path.join(output_dir, "replay_selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)
    return selection


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/open_xtts_trainer/librittsr"))
    parser.add_argument("--vocab", default="xtts_base/vocab.json")
    parser.add_argument("--evidence", default=None, help="Also write replay_selection.json here")
    args = parser.parse_args(argv)
    selection = build(args.output_dir, args.cache_dir, args.vocab)
    if args.evidence:
        with open(args.evidence, "w", encoding="utf-8") as f:
            json.dump(selection, f, indent=2, ensure_ascii=False)
    print(f"{len(selection['clips'])} clips from {len(selection['speakers'])} readers "
          f"({sum(s['gender'] == 'F' for s in selection['speakers'])} F); "
          f"{len(selection['rejected'])} candidates rejected")


if __name__ == "__main__":
    main()
