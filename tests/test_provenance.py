"""Baseline pieces are reused on resume only if they were measured on exactly the expected inputs."""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from eval.provenance import expected_baseline_inputs, index_manifest_mismatches, manifest_mismatches, write_baseline_index

COMMIT = "a" * 40


def _manifest(expected, language):
    """A finished manifest shaped like scripts/run_eval_baseline.py's, measured on the expected inputs."""
    return {
        "label": expected["label"],
        "git": {"commit_sha": expected["commit_sha"], "is_dirty": False},
        "code_fingerprint": {"sha256": expected["code_fingerprint"], "n_files": 1},
        "eval_set_sha256": expected["eval_set_sha256"],
        "checkpoint_sha256": dict(expected["checkpoint_sha256"]),
        "voices": "arctic",
        "reference_voices": [{"speaker_id": k, "sha256": v} for k, v in expected["reference_voices"].items()],
        "asr_model": expected["asr_model"],
        "seeds": list(expected["seeds"]),
        "subset_size": expected["subset_size"],
        "languages": [language],
        "metrics_version": expected["metrics_version"],
        "summary_by_language": {language: {"wer": {"mean": 0.05}}},
        "runs": [],
    }


@pytest.fixture
def expected():
    return expected_baseline_inputs(COMMIT, seeds=[42, 123, 999], subset_size=12)


def test_expected_inputs_come_from_committed_pins(expected):
    with open(os.path.join(os.path.dirname(__file__), "..", "artifacts.lock.json"), encoding="utf-8") as f:
        artifacts = json.load(f)
    assert expected["checkpoint_sha256"]["model.pth"] == artifacts["model.pth"]["sha256"]
    assert len(expected["reference_voices"]) == 5 and len(expected["eval_set_sha256"]) == 64


def test_matching_piece_is_reusable(expected):
    assert manifest_mismatches(_manifest(expected, "es"), expected, "es") == []


@pytest.mark.parametrize("edit, field", [
    (lambda m: m["git"].update(commit_sha="b" * 40), "commit_sha"),
    (lambda m: m["git"].update(is_dirty=True), "git.is_dirty"),
    (lambda m: m["code_fingerprint"].update(sha256="0" * 64), "code_fingerprint"),
    (lambda m: m.update(eval_set_sha256="0" * 64), "eval_set_sha256"),
    (lambda m: m["checkpoint_sha256"].update({"model.pth": "0" * 64}), "checkpoint_sha256"),
    (lambda m: m["reference_voices"].pop(), "reference_voices"),
    (lambda m: m.update(seeds=[42, 123]), "seeds"),
    (lambda m: m.update(subset_size=None), "subset_size"),
    (lambda m: m.update(metrics_version=1), "metrics_version"),
    (lambda m: m.update(asr_model="openai/whisper-tiny"), "asr_model"),
    (lambda m: m.update(voices="target"), "voices"),
    (lambda m: m.update(languages=["fr"]), "languages"),
    (lambda m: m.update(summary_by_language={}), "summary_by_language"),
])
def test_stale_or_unfinished_piece_is_refused(expected, edit, field):
    manifest = _manifest(expected, "es")
    edit(manifest)
    problems = manifest_mismatches(manifest, expected, "es")
    assert problems and any(p.startswith(field) for p in problems), problems


def test_index_records_each_piece_and_refuses_mixed_inputs(expected, tmp_path):
    paths = {}
    for lang in ("en", "es"):
        paths[lang] = str(tmp_path / f"{lang}.json")
        with open(paths[lang], "w", encoding="utf-8") as f:
            json.dump(_manifest(expected, lang), f)
    index = write_baseline_index(paths, str(tmp_path / "index.json"))
    assert sorted(index["files"]) == ["en", "es"] and all(len(e["sha256"]) == 64 for e in index["files"].values())
    assert index["inputs"]["commit_sha"] == COMMIT and sorted(index["summary_by_language"]) == ["en", "es"]

    other = copy.deepcopy(_manifest(expected, "es"))
    other["seeds"] = [1]
    with open(paths["es"], "w", encoding="utf-8") as f:
        json.dump(other, f)
    with pytest.raises(ValueError, match="seeds"):
        write_baseline_index(paths, str(tmp_path / "index.json"))


def test_index_manifest_mismatches_accepts_matching_file(expected, tmp_path):
    paths = {"es": str(tmp_path / "base_xtts_s12_es.json")}
    with open(paths["es"], "w", encoding="utf-8") as f:
        json.dump(_manifest(expected, "es"), f)
    index = write_baseline_index(paths, str(tmp_path / "index.json"))
    assert index_manifest_mismatches(paths["es"], index, "es") == []


def test_index_manifest_mismatches_detects_edited_manifest(expected, tmp_path):
    paths = {"es": str(tmp_path / "base_xtts_s12_es.json")}
    manifest = _manifest(expected, "es")
    with open(paths["es"], "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    index = write_baseline_index(paths, str(tmp_path / "index.json"))

    # Modify summary
    manifest["summary_by_language"]["es"]["wer"]["mean"] = 0.99
    with open(paths["es"], "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    problems = index_manifest_mismatches(paths["es"], index, "es")
    assert any("sha256:" in p for p in problems)
    assert any("summary_by_language:" in p for p in problems)


def test_index_manifest_mismatches_detects_missing_language_and_wrong_file(expected, tmp_path):
    paths = {"es": str(tmp_path / "base_xtts_s12_es.json")}
    with open(paths["es"], "w", encoding="utf-8") as f:
        json.dump(_manifest(expected, "es"), f)
    index = write_baseline_index(paths, str(tmp_path / "index.json"))

    assert "does not record" in index_manifest_mismatches(paths["es"], index, "fr")[0]

    renamed = tmp_path / "wrong_name.json"
    renamed.write_text((tmp_path / "base_xtts_s12_es.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert any("file:" in p for p in index_manifest_mismatches(str(renamed), index, "es"))

