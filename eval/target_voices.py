"""Target-speaker reference voices: held-out clips of the speaker(s) you fine-tune on.

The A/B question for voice adaptation is "does the fine-tune sound more like the target speaker than stock
cloning does?", so evaluation needs reference clips of that speaker that training never saw. This module:
- holds out whole clips per speaker from a dataset.jsonl (seeded), until they make a 6-10 s reference clip
  (same assembly as the CMU ARCTIC voices: whole utterances, edges trimmed, fixed gaps, never truncated);
- writes the clips and a lock with the SHA-256 of every reference clip *and* every held-out source clip;
- writes the remaining clips as the training dataset (dataset_train.jsonl next to the original).
xtts_trainer.data.validate refuses any training clip whose audio matches the lock (target_holdout_overlap),
so training on the original, un-split dataset by mistake is caught. Matching is by file hash: a re-encoded
copy of a held-out clip would not be caught, which the split here avoids by construction.

Voice ids are "target_<speaker>"; use them in eval/gates.json's target-speaker gate.

The lock always lives at xtts_trainer.data.validate.DEFAULT_TARGET_LOCK, the one place training checks: a lock
written elsewhere would let training on held-out clips pass unnoticed, so the builder refuses other locations
(tests opt out with allow_unchecked_location).
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from eval.reference_voices import MAX_CLIP_S, MIN_CLIP_S, assemble_clip, sha256_file
from xtts_trainer.data import DatasetRecord, load_validated_dataset, write_dataset
from xtts_trainer.data.validate import DEFAULT_TARGET_LOCK
from xtts_trainer.spec import SPEC

DEFAULT_TARGET_DIR = os.path.dirname(DEFAULT_TARGET_LOCK)
SPEAKER_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*")  # becomes a file name: no separators, no leading dot
LOCK_FILENAME = "target_voices.lock.json"
TRAIN_DATASET_FILENAME = "dataset_train.jsonl"
LOCK_FORMAT = {"format": "open-xtts-target-voices", "version": 1}
VOICE_PREFIX = "target_"


def voice_id(speaker: str) -> str:
    return f"{VOICE_PREFIX}{speaker}"


def _hold_out(records: List[DatasetRecord], dataset_dir: str, rng: random.Random
              ) -> Tuple[List[DatasetRecord], np.ndarray, List[int]]:
    """Shuffled clips short enough to fit are loaded one at a time until they assemble into a 6-10 s clip;
    those are held out. Clips load lazily, so a speaker with hours of audio costs a few clips' memory, and
    trimming or skipping never ends the search while eligible clips remain."""
    from xtts_trainer.prepare.pipeline import load_mono

    order = [r for r in records if r.duration_s <= MAX_CLIP_S]
    rng.shuffle(order)
    rate = SPEC.output_sample_rate
    loaded: List[np.ndarray] = []
    for r in order:
        loaded.append(load_mono(r.audio_path(dataset_dir), rate))
        if sum(len(a) for a in loaded) < MIN_CLIP_S * rate:
            continue
        try:
            clip, used = assemble_clip(loaded, rate)
        except ValueError:
            continue  # trimmed too short so far: keep adding clips
        return order[:len(loaded)], clip, used
    speaker = records[0].speaker if records else "?"
    raise ValueError(f"speaker '{speaker}': not enough audio for a {MIN_CLIP_S}-{MAX_CLIP_S} s reference clip "
                     f"from {len(order)} clips of at most {MAX_CLIP_S} s")


def build_target_voices(dataset_path: str, out_dir: str = DEFAULT_TARGET_DIR, speakers: Optional[List[str]] = None,
                        seed: int = 0, min_train_clips: int = 2, allow_unchecked_location: bool = False
                        ) -> Dict[str, Any]:
    """Hold out reference clips per speaker; write clips, the lock and dataset_train.jsonl. Returns the lock.

    Rerunning replaces the previous holdout. allow_unchecked_location: tests only (training doesn't check
    locks outside DEFAULT_TARGET_DIR).
    """
    if os.path.abspath(out_dir) != os.path.abspath(DEFAULT_TARGET_DIR) and not allow_unchecked_location:
        raise ValueError(f"target voices must be built in {DEFAULT_TARGET_DIR}: it is the only lock location "
                         f"training checks, so a holdout elsewhere could be trained on unnoticed")
    # The source dataset contains this speaker's clips by definition, including any previous holdout, so it is
    # validated without the target-holdout check (every other check still applies)
    records, dataset_dir, _ = load_validated_dataset(dataset_path, None, require_text=False, target_lock_path=None)
    by_speaker: Dict[str, List[DatasetRecord]] = {}
    for r in records:
        by_speaker.setdefault(r.speaker, []).append(r)
    speakers = speakers or sorted(by_speaker)
    unknown = sorted(set(speakers) - set(by_speaker))
    if unknown:
        raise KeyError(f"speakers {unknown} are not in {dataset_path}")
    unsafe = [s for s in speakers if not SPEAKER_ID.fullmatch(s)]
    if unsafe:
        raise ValueError(f"speaker ids {unsafe} can't be file names (letters, digits, '_', '-', '.'; no leading '.')")

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    voices, held_out = {}, set()
    for speaker in speakers:
        clips = sorted(by_speaker[speaker], key=lambda r: r.audio)
        order, clip, used = _hold_out(clips, dataset_dir, rng)
        if len(clips) - len(used) < min_train_clips:
            raise ValueError(f"speaker '{speaker}' would keep only {len(clips) - len(used)} training clips; "
                             f"needs at least {min_train_clips} besides the held-out reference")
        vid = voice_id(speaker)
        path = os.path.join(out_dir, f"{vid}.wav")
        sf.write(path, clip, SPEC.output_sample_rate, subtype="PCM_16")
        sources = [order[i] for i in used]
        held_out.update(r.audio for r in sources)
        voices[vid] = {
            "speaker": speaker,
            "clip_sha256": sha256_file(path),
            "duration_s": round(len(clip) / SPEC.output_sample_rate, 2),
            "source_clips": [{"audio": r.audio, "sha256": sha256_file(r.audio_path(dataset_dir))} for r in sources],
            "transcript": " ".join(r.text for r in sources),
        }

    train = [r for r in records if r.audio not in held_out]
    train_path = write_dataset(os.path.join(dataset_dir, TRAIN_DATASET_FILENAME), train)
    lock = {**LOCK_FORMAT, "dataset": os.path.abspath(dataset_path), "seed": seed,
            "train_dataset": os.path.abspath(train_path), "voices": voices}
    with open(os.path.join(out_dir, LOCK_FILENAME), "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)
    return lock


def load_lock(out_dir: str = DEFAULT_TARGET_DIR) -> Dict[str, Any]:
    path = os.path.join(out_dir, LOCK_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    if {k: lock.get(k) for k in LOCK_FORMAT} != LOCK_FORMAT:
        raise ValueError(f"{path} is not a target-voice lock")
    return lock


def verify_target_voices(out_dir: str = DEFAULT_TARGET_DIR) -> Dict[str, str]:
    """{voice_id: clip path} after checking every clip against the lock; raises on any mismatch."""
    paths = {}
    for vid, entry in load_lock(out_dir)["voices"].items():
        path = os.path.join(out_dir, f"{vid}.wav")
        if not os.path.exists(path) or sha256_file(path) != entry["clip_sha256"]:
            raise RuntimeError(f"{path} is missing or doesn't match the target-voice lock; rebuild it")
        paths[vid] = path
    return paths


def holdout_hashes(lock_path: str) -> Dict[str, str]:
    """SHA-256 -> voice id for every reference clip and held-out source clip (shared with the validator)."""
    from xtts_trainer.data.validate import target_holdout_hashes

    return target_holdout_hashes(lock_path)
