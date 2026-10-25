"""Provenance checks for baseline manifests written in pieces (one per language, possibly across sessions).

A long baseline is run one language at a time so a lost session only loses the language in progress. Resuming
must not adopt a manifest from different inputs: expected_baseline_inputs() states what every piece must have
been measured on, from this checkout's committed pins (not from whatever happens to be on disk), and
manifest_mismatches() lists every difference. write_baseline_index() then combines the pieces into one index
that records each file's SHA-256, so the per-language manifests stay the immutable sources.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from eval.dataset import DEFAULT_EVAL_SET_PATH, load_eval_set
from eval.harness import METRICS_VERSION, get_code_fingerprint
from eval.reference_voices import DEFAULT_LOCK_PATH

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_LOCK = os.path.join(REPO_ROOT, "artifacts.lock.json")
CHECKPOINT_FILES = ("model.pth", "config.json", "vocab.json")  # what run_eval_baseline hashes
# Inputs every piece of one baseline shares; "languages" differs per piece and is checked separately
SHARED_KEYS = ("commit_sha", "code_fingerprint", "eval_set_sha256", "checkpoint_sha256", "reference_voices",
               "voices", "label", "asr_model", "seeds", "subset_size", "metrics_version")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_sha256_file = sha256_file


def expected_baseline_inputs(commit_sha: str, seeds: Sequence[int], subset_size: Optional[int], label: str = "base",
                             asr_model: str = "openai/whisper-large-v3", eval_set_path: str = DEFAULT_EVAL_SET_PATH,
                             reference_lock_path: str = DEFAULT_LOCK_PATH, artifacts_lock_path: str = ARTIFACTS_LOCK,
                             repo_root: str = REPO_ROOT) -> Dict[str, Any]:
    """The inputs a stock-model baseline on the eval set's ARCTIC voices must record, from committed pins."""
    with open(artifacts_lock_path, "r", encoding="utf-8") as f:
        artifacts = json.load(f)
    with open(reference_lock_path, "r", encoding="utf-8") as f:
        voices = json.load(f)["voices"]
    return {
        "commit_sha": commit_sha,
        "code_fingerprint": get_code_fingerprint(repo_root)["sha256"],
        "eval_set_sha256": load_eval_set(eval_set_path)["sha256"],
        "checkpoint_sha256": {name: artifacts[name]["sha256"] for name in CHECKPOINT_FILES},
        "reference_voices": {vid: v["clip_sha256"] for vid, v in voices.items()},
        "voices": "arctic",
        "label": label,
        "asr_model": asr_model,
        "seeds": list(seeds),
        "subset_size": subset_size,
        "metrics_version": METRICS_VERSION,
    }


def recorded_inputs(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The same fields as expected_baseline_inputs, read from a manifest."""
    return {
        "commit_sha": (manifest.get("git") or {}).get("commit_sha"),
        "code_fingerprint": (manifest.get("code_fingerprint") or {}).get("sha256"),
        "eval_set_sha256": manifest.get("eval_set_sha256"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "reference_voices": {v["speaker_id"]: v["sha256"] for v in manifest.get("reference_voices", [])},
        "voices": manifest.get("voices", "arctic"),  # manifests before the field existed only measured ARCTIC
        "label": manifest.get("label"),
        "asr_model": manifest.get("asr_model"),
        "seeds": list(manifest.get("seeds") or []),
        "subset_size": manifest.get("subset_size"),
        "metrics_version": manifest.get("metrics_version"),
    }


def manifest_mismatches(manifest: Dict[str, Any], expected: Dict[str, Any], language: str) -> List[str]:
    """Every way the manifest differs from the expected inputs; empty means it may be reused for `language`."""
    got = recorded_inputs(manifest)
    problems = [f"{k}: recorded {got[k]!r}, expected {expected[k]!r}" for k in SHARED_KEYS if got[k] != expected[k]]
    if (manifest.get("git") or {}).get("is_dirty") is not False:
        problems.append(f"git.is_dirty: recorded {(manifest.get('git') or {}).get('is_dirty')!r}, expected False")
    if manifest.get("languages") != [language]:
        problems.append(f"languages: recorded {manifest.get('languages')!r}, expected {[language]!r}")
    if language not in (manifest.get("summary_by_language") or {}):
        problems.append(f"summary_by_language has no {language!r}: the run did not finish")
    return problems


def write_baseline_index(manifest_paths: Dict[str, str], out_path: str) -> Dict[str, Any]:
    """One index over per-language manifests that share every input; raises if any shared input differs."""
    shared, files, summaries = None, {}, {}
    for language, path in sorted(manifest_paths.items()):
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        inputs = {k: v for k, v in recorded_inputs(manifest).items() if k in SHARED_KEYS}
        if shared is None:
            shared = inputs
        elif inputs != shared:
            diff = sorted(k for k in SHARED_KEYS if inputs[k] != shared[k])
            raise ValueError(f"{path} was measured on different inputs from the other pieces: {diff}")
        files[language] = {"file": os.path.basename(path), "sha256": _sha256_file(path)}
        summaries[language] = manifest["summary_by_language"][language]
    index = {"format": "open-xtts-baseline-index", "version": 1, "inputs": shared, "files": files,
             "summary_by_language": summaries}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index


def index_manifest_mismatches(manifest_path: str, index: Dict[str, Any], language: str) -> List[str]:
    """Check that a manifest file matches the file name, SHA-256 and summary recorded in an existing index."""
    entry = (index.get("files") or {}).get(language)
    if not entry:
        return [f"index does not record language {language!r}"]
    expected_file = os.path.basename(manifest_path)
    problems = []
    if entry.get("file") != expected_file:
        problems.append(f"file: index records {entry.get('file')!r}, got {expected_file!r}")
    actual_sha = sha256_file(manifest_path)
    if actual_sha != entry.get("sha256"):
        problems.append(f"sha256: file has {actual_sha}, index records {entry.get('sha256')}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    actual_summary = (manifest.get("summary_by_language") or {}).get(language)
    index_summary = (index.get("summary_by_language") or {}).get(language)
    if actual_summary != index_summary:
        problems.append("summary_by_language: manifest summary does not match index summary")
    return problems
