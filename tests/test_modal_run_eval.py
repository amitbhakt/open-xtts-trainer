"""scripts/modal_run_eval.py: which files a remote run brings back. Modal itself isn't installed, so it is stubbed."""

import importlib
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load():
    with mock.patch.dict(sys.modules, {"modal": mock.MagicMock()}):
        sys.modules.pop("scripts.modal_run_eval", None)
        return importlib.import_module("scripts.modal_run_eval")


def test_private_clips_audio_and_weights_never_come_back(tmp_path):
    runner = _load()
    files = {
        "runs/m1/c1/ab_report_target.json": "{}",
        "runs/m1/c1/evidence/training_summary.json": "{}",
        "runs/m1/c1/evidence/target_voices/target_voices.lock.json": "{}",  # the held-out speaker's folder
        "runs/m1/c1/evidence/target_voices/clip_000.wav": "audio",
        "runs/m1/c1/samples/take.WAV": "audio",
        "runs/m1/c1/samples/take.m4a": "audio",
        "runs/m1/c1/samples/take.opus": "audio",
        "runs/m1/c1/export/model.pth": "weights",
        "runs/m1/c1/export/model.bin": "weights",
        "runs/m1/c1/export/model.onnx": "weights",
        "runs/m1/c1/export/tensors.npy": "tensors",
        "runs/m1/c1/export/archive.tar.bz2": "archive",
        "runs/m1/c2/ab_report_target.json": "{}",  # another candidate: not under the chosen folder
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    # Symlink pointing to external private file should never come back
    secret = tmp_path / "secret_clip.wav"
    secret.write_text("private")
    symlink = tmp_path / "runs/m1/c1/evidence/leak_report.json"
    symlink.symlink_to(secret)

    got = runner.evidence_digests(tmp_path, "runs/m1/c1")
    assert set(got) == {"runs/m1/c1/ab_report_target.json", "runs/m1/c1/evidence/training_summary.json"}


def test_default_folder_is_still_the_held_out_evidence(tmp_path):
    runner = _load()
    assert runner.DEFAULT_EVIDENCE_DIR == "docs/evidence/phase6_heldout"
    (tmp_path / "docs/evidence/phase6_heldout").mkdir(parents=True)
    (tmp_path / "docs/evidence/phase6_heldout/report.json").write_text("{}")
    assert list(runner.evidence_digests(tmp_path)) == ["docs/evidence/phase6_heldout/report.json"]
    assert runner.evidence_digests(tmp_path, "runs/missing") == {}


def test_tracked_evidence_files_tracks_deletions_for_all_suffixes(tmp_path):
    runner = _load()
    dir_path = tmp_path / "evidence"
    dir_path.mkdir()
    (dir_path / "report.json").write_text("{}")
    (dir_path / "data.npz").write_text("data")
    (dir_path / "weights.bin").write_text("bin")
    tracked = runner.tracked_evidence_files(tmp_path, "evidence")
    assert tracked == {"evidence/report.json", "evidence/data.npz", "evidence/weights.bin"}
