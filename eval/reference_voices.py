"""Build and verify the eval set's reference voice clips from their source corpus (CMU ARCTIC).

Every step is deterministic and hash-locked:
- source tarballs are verified against `reference_voices.lock.json` (written on the first run, since
  festvox.org serves plain HTTP and publishes no checksums; review and keep that lock file);
- each clip is rebuilt from the listed utterances and must reproduce the locked clip hash.
"""

import datetime
import hashlib
import io
import json
import os
import re
import tarfile
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from eval.dataset import CMU_ARCTIC_LICENSE_URL, CMU_ARCTIC_URL, DEFAULT_EVAL_SET_PATH, load_eval_set

# Absolute (repo-root based), so the defaults don't depend on the working directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "eval", "reference_voices")
DEFAULT_LOCK_PATH = os.path.join(_REPO_ROOT, "eval", "dataset", "reference_voices.lock.json")
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/open_xtts_trainer/cmu_arctic")
MIN_CLIP_S = 6.0
MAX_CLIP_S = 10.0
GAP_S = 0.25
TRIM_DB = -40.0  # edge frames this far below the utterance peak (RMS) are trimmed
NOTICE_FILENAME = "MODIFICATIONS.txt"
TRANSCRIPT_LINE = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)\s*$')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: str) -> str:
    """Download url to dest (atomically). Returns the file's SHA-256."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "open-xtts-trainer/fetch_reference_voices"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)
            digest.update(chunk)
    os.replace(tmp, dest)
    return digest.hexdigest()


def trim_edges(wav: np.ndarray, sample_rate: int, db: float = TRIM_DB) -> np.ndarray:
    """Trim leading/trailing frames whose RMS is `db` below the loudest frame (10 ms frames)."""
    frame = max(1, int(0.01 * sample_rate))
    n_frames = len(wav) // frame
    if n_frames == 0:
        return wav
    rms = np.sqrt(np.mean(wav[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms / rms.max()) > db)[0]
    if len(loud) == 0:
        return wav
    return wav[loud[0] * frame : (loud[-1] + 1) * frame]


def assemble_clip(utterances: List[np.ndarray], sample_rate: int):
    """Join whole, edge-trimmed utterances (in list order, with fixed gaps) until the clip reaches MIN_CLIP_S.

    An utterance that would push the clip past MAX_CLIP_S is skipped rather than truncated, so the clip's
    transcript is always exactly its utterances' text. Returns (clip, indices of the utterances used).
    """
    gap = np.zeros(int(GAP_S * sample_rate), dtype=np.float32)
    parts: List[np.ndarray] = []
    used: List[int] = []
    total = 0
    for i, utt in enumerate(utterances):
        if total >= MIN_CLIP_S * sample_rate:
            break
        trimmed = trim_edges(utt.astype(np.float32), sample_rate)
        added = len(trimmed) + (len(gap) if parts else 0)
        if total + added > MAX_CLIP_S * sample_rate:
            continue
        if parts:
            parts.append(gap)
        parts.append(trimmed)
        used.append(i)
        total += added
    if total < MIN_CLIP_S * sample_rate:
        raise ValueError(f"Reference clip is only {total / sample_rate:.2f}s; list more utterances.")
    return np.concatenate(parts), used


def read_from_tarball(tar_path: str, corpus_speaker: str, utterances: List[str]):
    """Return ([utterance arrays], sample_rate, {utterance: transcript}) from a CMU ARCTIC tarball."""
    root = f"cmu_us_{corpus_speaker}_arctic"
    wanted = {f"{root}/wav/{u}.wav": u for u in utterances}
    transcript_member = f"{root}/etc/txt.done.data"
    found: Dict[str, bytes] = {}
    transcript_data: Optional[bytes] = None
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name in wanted:
                found[wanted[member.name]] = tar.extractfile(member).read()
            elif member.name == transcript_member:
                transcript_data = tar.extractfile(member).read()
    missing = [u for u in utterances if u not in found]
    if missing:
        raise FileNotFoundError(f"{tar_path} is missing utterances {missing} under {root}/wav/")
    if transcript_data is None:
        raise FileNotFoundError(f"{tar_path} has no {transcript_member}")
    transcripts = {}
    for line in transcript_data.decode("utf-8").splitlines():
        match = TRANSCRIPT_LINE.match(line.strip())
        if match and match.group(1) in utterances:
            transcripts[match.group(1)] = match.group(2)
    no_text = [u for u in utterances if u not in transcripts]
    if no_text:
        raise ValueError(f"{transcript_member} has no transcript for {no_text}")

    arrays, rates = [], set()
    for u in utterances:
        data, sr = sf.read(io.BytesIO(found[u]), dtype="float32", always_2d=True)
        arrays.append(data.mean(axis=1))
        rates.add(sr)
    if len(rates) != 1:
        raise ValueError(f"Mixed sample rates in {tar_path}: {sorted(rates)}")
    return arrays, rates.pop(), transcripts


def write_modification_notice(out_dir: str, lock_voices: Dict[str, Any]) -> str:
    """Mark the clips as modified works, as the CMU ARCTIC license requires (condition 2)."""
    lines = [
        "These reference clips are MODIFIED excerpts of the CMU ARCTIC speech databases",
        "(Language Technologies Institute, Carnegie Mellon University; http://www.festvox.org/cmu_arctic/).",
        "The original copyright notice and license for each speaker is in licenses/.",
        "",
        "Modifications, made by eval/reference_voices.py in open-xtts-trainer:",
        "- leading and trailing silence trimmed from each utterance;",
        "- whole utterances concatenated in the listed order with 0.25 s of inserted silence;",
        "- re-encoded as 16-bit PCM WAV at the original sample rate.",
        "",
    ]
    for speaker_id, entry in sorted(lock_voices.items()):
        lines.append(f"{speaker_id}.wav: {' + '.join(entry['utterances_used'])} (from {entry['source_url']})")
    path = os.path.join(out_dir, NOTICE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build_reference_voices(
    out_dir: str = DEFAULT_OUT_DIR,
    lock_path: str = DEFAULT_LOCK_PATH,
    cache_dir: str = DEFAULT_CACHE_DIR,
    eval_set_path: str = DEFAULT_EVAL_SET_PATH,
    url_template: str = CMU_ARCTIC_URL,
    license_url_template: str = CMU_ARCTIC_LICENSE_URL,
    update_lock: bool = False,
) -> Dict[str, Any]:
    """Download, verify and build every reference clip and its license. Returns the lock contents.

    With an existing lock, every tarball, clip and license hash must match it (unless update_lock=True).
    Without one, a new lock is written and must be reviewed and kept with the repo.
    """
    eval_data = load_eval_set(eval_set_path)
    lock: Optional[Dict[str, Any]] = None
    if os.path.exists(lock_path) and not update_lock:
        with open(lock_path, "r", encoding="utf-8") as f:
            lock = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    new_lock: Dict[str, Any] = {
        "eval_set_sha256": eval_data.get("sha256", ""),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "voices": {},
    }
    for voice in eval_data["reference_voices"]:
        speaker_id = voice["speaker_id"]
        url = url_template.format(speaker=voice["corpus_speaker"])
        tar_path = os.path.join(cache_dir, os.path.basename(url))
        locked = lock["voices"].get(speaker_id) if lock else None
        if locked and "license_sha256" not in locked:
            raise RuntimeError(
                f"{lock_path} predates license and transcript capture; rebuild it with --update_lock."
            )

        if os.path.exists(tar_path):
            tar_sha = sha256_file(tar_path)
        else:
            print(f"Downloading {url} ...")
            tar_sha = download(url, tar_path)
        if locked and tar_sha != locked["tarball_sha256"]:
            raise RuntimeError(
                f"{tar_path} SHA-256 {tar_sha} != locked {locked['tarball_sha256']}. "
                f"Delete it to re-download, or rerun with update_lock after checking the source."
            )

        arrays, sr, transcripts = read_from_tarball(tar_path, voice["corpus_speaker"], voice["utterances"])
        clip, used_idx = assemble_clip(arrays, sr)
        used = [voice["utterances"][i] for i in used_idx]
        clip_path = os.path.join(out_dir, f"{speaker_id}.wav")
        buf = io.BytesIO()
        sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
        clip_bytes = buf.getvalue()
        clip_sha = sha256_bytes(clip_bytes)
        if locked and clip_sha != locked["clip_sha256"]:
            raise RuntimeError(f"Rebuilt {clip_path} SHA-256 {clip_sha} != locked {locked['clip_sha256']}.")
        with open(clip_path, "wb") as f:
            f.write(clip_bytes)

        license_url = license_url_template.format(speaker=voice["corpus_speaker"])
        license_path = os.path.join(out_dir, "licenses", f"{voice['corpus_speaker']}_COPYING")
        license_sha = download(license_url, license_path)  # raises if unavailable: no license, no clip
        if locked and license_sha != locked["license_sha256"]:
            raise RuntimeError(f"{license_url} changed (SHA-256 {license_sha} != locked {locked['license_sha256']}).")

        new_lock["voices"][speaker_id] = {
            "source_url": url,
            "tarball_sha256": tar_sha,
            "utterances": voice["utterances"],
            "utterances_used": used,
            "transcript": " ".join(transcripts[u] for u in used),
            "sample_rate": sr,
            "duration_s": round(len(clip) / sr, 3),
            "clip_sha256": clip_sha,
            "license_url": license_url,
            "license_file": os.path.relpath(license_path, out_dir),
            "license_sha256": license_sha,
        }
        print(f"  [OK] {speaker_id}: {len(clip) / sr:.2f}s -> {clip_path}")

    write_modification_notice(out_dir, new_lock["voices"])
    if lock is None:
        os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(new_lock, f, indent=2)
        print(f"Wrote new lock file {lock_path}. Review it and keep it with the repo.")
        return new_lock
    return lock


def verify_reference_voices(out_dir: str, lock_path: str = DEFAULT_LOCK_PATH) -> Dict[str, str]:
    """Check every built clip against the lock. Returns {speaker_id: clip_path}; raises on any mismatch."""
    if not os.path.exists(lock_path):
        raise FileNotFoundError(f"No reference-voice lock at {lock_path}; run scripts/fetch_reference_voices.py first.")
    with open(lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    paths = {}
    for speaker_id, entry in lock["voices"].items():
        path = os.path.join(out_dir, f"{speaker_id}.wav")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing reference clip {path}")
        if sha256_file(path) != entry["clip_sha256"]:
            raise RuntimeError(f"{path} does not match its locked SHA-256; rebuild it.")
        if "license_sha256" not in entry:
            raise RuntimeError(f"{lock_path} predates license capture; rebuild it with --update_lock.")
        license_path = os.path.join(out_dir, entry["license_file"])
        if not os.path.exists(license_path) or sha256_file(license_path) != entry["license_sha256"]:
            raise RuntimeError(f"License {license_path} is missing or changed; rebuild the reference voices.")
        paths[speaker_id] = path
    if not os.path.exists(os.path.join(out_dir, NOTICE_FILENAME)):
        raise FileNotFoundError(f"Missing {NOTICE_FILENAME} in {out_dir} (the license requires marking modifications)")
    return paths
