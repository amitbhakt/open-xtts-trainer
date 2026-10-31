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
        "runs/m1/c1/export/model.pth": "weights",
        "runs/m1/c2/ab_report_target.json": "{}",  # another candidate: not under the chosen folder
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    got = runner.evidence_digests(tmp_path, "runs/m1/c1")
    assert set(got) == {"runs/m1/c1/ab_report_target.json", "runs/m1/c1/evidence/training_summary.json"}


def test_default_folder_is_still_the_held_out_evidence(tmp_path):
    runner = _load()
    assert runner.DEFAULT_EVIDENCE_DIR == "docs/evidence/phase6_heldout"
    (tmp_path / "docs/evidence/phase6_heldout").mkdir(parents=True)
    (tmp_path / "docs/evidence/phase6_heldout/report.json").write_text("{}")
    assert list(runner.evidence_digests(tmp_path)) == ["docs/evidence/phase6_heldout/report.json"]
    assert runner.evidence_digests(tmp_path, "runs/missing") == {}
