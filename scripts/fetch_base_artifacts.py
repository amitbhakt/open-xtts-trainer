"""Fetch and verify upstream XTTS-v2 artifacts listed in artifacts.lock.json.

Every file is streamed to disk while its SHA-256 is computed, then checked against the
lockfile's sha256 and size_bytes. Nothing is trusted from memory: a mismatch is an error.

Optionally runs the mel_stats three-way check (packaged copy vs downloaded mel_stats.pth vs
the mel_stats tensor stored inside model.pth) and writes a JSON evidence report.

Usage:
    python scripts/fetch_base_artifacts.py --dest xtts_base
    python scripts/fetch_base_artifacts.py --dest xtts_base --only mel_stats.pth config.json vocab.json
    python scripts/fetch_base_artifacts.py --dest xtts_base --verify_only
    python scripts/fetch_base_artifacts.py --dest xtts_base --check_mel_stats --report docs/evidence/artifact_verification.json
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOCKFILE = os.path.join(REPO_ROOT, "artifacts.lock.json")
PACKAGED_MEL_STATS = os.path.join(REPO_ROOT, "xtts_trainer", "mel_stats.pth")
CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_of_file(path: str) -> str:
    """Compute the SHA-256 of a file by streaming it."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lockfile(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    for name, entry in lock.items():
        for key in ("url", "sha256", "size_bytes"):
            if key not in entry:
                raise ValueError(f"Lockfile entry '{name}' is missing required field '{key}'.")
    return lock


def verify_file(path: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Check a local file against its lockfile entry. Returns a result dict; never raises on mismatch."""
    size = os.path.getsize(path)
    file_hash = sha256_of_file(path)
    return {
        "path": path,
        "size_bytes": size,
        "sha256": file_hash,
        "size_ok": size == entry["size_bytes"],
        "sha256_ok": file_hash == entry["sha256"],
    }


def download_file(url: str, dest: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Stream a file to dest + '.part', hashing as it downloads, and only rename into place if it verifies."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp_path = dest + ".part"
    digest = hashlib.sha256()
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "open-xtts-trainer/fetch_base_artifacts"})
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out:
        for chunk in iter(lambda: resp.read(CHUNK_SIZE), b""):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)

    file_hash = digest.hexdigest()
    result = {
        "path": dest,
        "size_bytes": size,
        "sha256": file_hash,
        "size_ok": size == entry["size_bytes"],
        "sha256_ok": file_hash == entry["sha256"],
    }
    if result["size_ok"] and result["sha256_ok"]:
        os.replace(tmp_path, dest)
    else:
        os.remove(tmp_path)
    return result


def check_mel_stats(dest_dir: str) -> Dict[str, Any]:
    """Compare the packaged mel_stats.pth, the downloaded one, and the tensor stored inside model.pth."""
    import torch

    def _load_tensor(path: str) -> "torch.Tensor":
        return torch.load(path, map_location="cpu", weights_only=True)

    sources: Dict[str, Optional["torch.Tensor"]] = {}
    notes: List[str] = []

    sources["packaged"] = _load_tensor(PACKAGED_MEL_STATS) if os.path.exists(PACKAGED_MEL_STATS) else None

    downloaded_path = os.path.join(dest_dir, "mel_stats.pth")
    sources["downloaded"] = _load_tensor(downloaded_path) if os.path.exists(downloaded_path) else None

    model_path = os.path.join(dest_dir, "model.pth")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        key = next((k for k in ("mel_stats", "xtts.mel_stats") if k in state), None)
        sources["model_pth"] = state[key] if key else None
        if key is None:
            notes.append("model.pth has no 'mel_stats' key.")
    else:
        sources["model_pth"] = None
        notes.append("model.pth not present; three-way comparison incomplete.")

    available = {name: t for name, t in sources.items() if t is not None}
    shapes = {name: list(t.shape) for name, t in available.items()}
    names = list(available)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = available[a], available[b]
            equal = ta.shape == tb.shape and torch.equal(ta, tb)
            max_abs_diff = float((ta - tb).abs().max()) if ta.shape == tb.shape else None
            pairwise[f"{a}_vs_{b}"] = {"equal": equal, "max_abs_diff": max_abs_diff}

    complete = len(available) == 3
    all_equal = bool(pairwise) and all(p["equal"] for p in pairwise.values())
    return {
        "sources_found": names,
        "shapes": shapes,
        "pairwise": pairwise,
        "complete": complete,
        "all_equal": all_equal,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify upstream XTTS-v2 artifacts from artifacts.lock.json.")
    parser.add_argument("--dest", type=str, default="xtts_base", help="Directory to download artifacts into")
    parser.add_argument("--lockfile", type=str, default=DEFAULT_LOCKFILE, help="Path to artifacts.lock.json")
    parser.add_argument("--only", type=str, nargs="+", default=None, help="Only these artifact names (e.g. mel_stats.pth config.json)")
    parser.add_argument("--verify_only", action="store_true", help="Do not download; only verify files already in --dest")
    parser.add_argument("--check_mel_stats", action="store_true", help="Run the packaged/downloaded/model.pth mel_stats comparison")
    parser.add_argument("--report", type=str, default=None, help="Write a JSON evidence report to this path")
    args = parser.parse_args()

    lock = load_lockfile(args.lockfile)
    names = args.only or list(lock)
    unknown = [n for n in names if n not in lock]
    if unknown:
        raise SystemExit(f"Not in lockfile: {unknown}")

    report: Dict[str, Any] = {
        "lockfile": os.path.relpath(args.lockfile, REPO_ROOT),
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "artifacts": {},
    }
    failures = []

    for name in names:
        entry = lock[name]
        dest_path = os.path.join(args.dest, name)
        if os.path.exists(dest_path):
            result = verify_file(dest_path, entry)
            result["action"] = "verified_existing"
        elif args.verify_only:
            result = {"path": dest_path, "action": "missing", "size_ok": False, "sha256_ok": False}
        else:
            print(f"Downloading {name} ({entry['size_bytes']:,} bytes) from {entry['url']} ...")
            result = download_file(entry["url"], dest_path, entry)
            result["action"] = "downloaded"

        ok = result["size_ok"] and result["sha256_ok"]
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: action={result['action']} size_ok={result['size_ok']} sha256_ok={result['sha256_ok']}")
        if not ok:
            failures.append(name)
        result["path"] = os.path.relpath(result["path"], REPO_ROOT) if os.path.isabs(result["path"]) else result["path"]
        report["artifacts"][name] = result

    if args.check_mel_stats:
        mel_report = check_mel_stats(args.dest)
        report["mel_stats_check"] = mel_report
        print(f"  mel_stats sources: {mel_report['sources_found']} | all_equal={mel_report['all_equal']} | complete={mel_report['complete']}")
        for pair, res in mel_report["pairwise"].items():
            print(f"    {pair}: equal={res['equal']} max_abs_diff={res['max_abs_diff']}")
        for note in mel_report["notes"]:
            print(f"    note: {note}")
        if mel_report["pairwise"] and not mel_report["all_equal"]:
            failures.append("mel_stats_check")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote evidence report to {args.report}")

    if failures:
        print(f"Verification FAILED for: {failures}")
        sys.exit(1)
    print("All requested artifacts verified.")


if __name__ == "__main__":
    main()
