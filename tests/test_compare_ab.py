"""Phase 8: paired A/B comparison with pre-registered gates (scripts/compare_ab.py).

The stock side is Colab run 2's real pilot manifest (120 utterances); the "tuned" side is a doctored copy.
"""

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from scripts.compare_ab import compare_ab

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = json.loads(Path(REPO_ROOT, "eval", "gates.json").read_text(encoding="utf-8"))


@pytest.fixture
def base():
    with open(os.path.join(REPO_ROOT, "docs", "evidence", "colab_run2", "pilot_en_metrics_v2.json"), encoding="utf-8") as f:
        return json.load(f)


def tuned_from(base, edit=None, voice=None):
    tuned = copy.deepcopy(base)
    tuned["checkpoint_sha256"] = {"model.pth": "tuned"}
    tuned["label"] = "tuned"
    for run in tuned["runs"]:
        if voice and run["speaker_id"] != voice:
            continue
        for results in run["per_seed_results"].values():
            for r in results:
                if edit:
                    edit(r)
    return tuned


def gate(report, metric):
    return next(g for g in report["gates"] if g["metric"] == metric)


def test_identical_outputs_pass_not_worse_and_fail_better(base):
    report = compare_ab(base, tuned_from(base), GATES)
    assert report["pairs"] == 120 and report["passed"]
    assert all(s["diff"] == 0 for s in report["overall"].values())
    assert report["overall"]["secs"]["n"] == 100  # the 20 sub-2 s short clips are not headline SECS
    better = compare_ab(base, tuned_from(base), {"gates": [{"metric": "secs", "rule": "better", "margin": 0.0}]})
    assert not better["passed"]  # no change is not an improvement


def test_regressions_fail_their_gate(base):
    worse_wer = compare_ab(base, tuned_from(base, lambda r: r.update(wer=r["wer"] + 0.1)), GATES)
    assert not gate(worse_wer, "wer")["passed"] and not worse_wer["passed"]
    worse_mos = compare_ab(base, tuned_from(base, lambda r: r.update(utmos=r["utmos"] - 0.5)), GATES)
    assert not gate(worse_mos, "utmos")["passed"] and gate(worse_mos, "wer")["passed"]
    better_wer = compare_ab(base, tuned_from(base, lambda r: r.update(wer=max(0.0, r["wer"] - 0.05))), GATES)
    assert gate(better_wer, "wer")["passed"]


def test_failure_count_gate(base):
    def babble_once(r, done=[]):
        if not done:
            r["failures"]["is_too_long"] = True
            done.append(1)

    report = compare_ab(base, tuned_from(base, babble_once), GATES)
    assert gate(report, "failures") == {**gate(report, "failures"), "base": 0, "tuned": 1, "passed": False}


def test_failure_count_does_not_double_count_overlapping_flags(base):
    def multiple_failures_on_single_utterance(r, done=[]):
        if not done:
            r["failures"]["is_too_long"] = True
            r["failures"]["is_empty"] = True
            r["failures"]["has_nan"] = True
            done.append(1)

    report = compare_ab(base, tuned_from(base, multiple_failures_on_single_utterance), GATES)
    assert gate(report, "failures") == {**gate(report, "failures"), "base": 0, "tuned": 1, "passed": False}


def test_target_speaker_gate_counts_only_its_voices(base):
    gates = {"gates": [{"metric": "secs", "rule": "better", "margin": 0.0, "voices": ["arctic_slt"]}]}
    tuned = tuned_from(base, lambda r: r.update(secs=r["secs"] + 0.05), voice="arctic_slt")
    report = compare_ab(base, tuned, gates)
    assert report["passed"] and gate(report, "secs")["n"] == 20  # slt: 10 long sentences x 2 seeds
    assert report["by_voice"]["arctic_awb"]["secs"]["diff"] == 0
    assert report["by_voice"]["arctic_slt"]["secs"]["diff"] == pytest.approx(0.05)
    assert "short" in report["by_category"]


def test_comparisons_need_identical_inputs(base):
    tuned = tuned_from(base)
    tuned["seeds"] = [1, 2]
    with pytest.raises(ValueError, match="seeds"):
        compare_ab(base, tuned)
    base["voices"] = "arctic"  # as run_eval_baseline records it (the run-2 manifest predates the field)
    tuned = tuned_from(base)
    tuned["voices"] = "target"  # measured on the target speaker, base on the eval voices
    with pytest.raises(ValueError, match="voices"):
        compare_ab(base, tuned)
    # A manifest from before the field was always measured on the ARCTIC voices, so it pairs with "arctic"
    legacy = copy.deepcopy(base)
    del legacy["voices"]
    assert compare_ab(legacy, tuned_from(base))["pairs"] > 0
    with pytest.raises(ValueError, match="voices"):
        compare_ab(legacy, tuned)  # ...and still not with a target-speaker run
    with pytest.raises(ValueError, match="same checkpoint"):
        compare_ab(base, copy.deepcopy(base))
    tuned = tuned_from(base)
    tuned["runs"][0]["per_seed_results"]["42"].pop()
    with pytest.raises(ValueError, match="unpaired"):
        compare_ab(base, tuned)
    tuned_batched = tuned_from(base)
    tuned_batched["batch_size"] = 16  # base has no batch_size (legacy sequential = 1)
    with pytest.raises(ValueError, match="batch_size"):
        compare_ab(base, tuned_batched)


def test_gates_file_is_pre_registered_and_well_formed():
    assert GATES["version"] == 1 and GATES["gates"]
    for g in GATES["gates"]:
        assert g["rule"] in ("not_worse", "better") and "margin" in g and g["why"]


def test_pre_registered_ablations_file_is_well_formed():
    path = Path(REPO_ROOT, "eval", "pre_registered_ablations.json")
    assert path.is_file(), "eval/pre_registered_ablations.json must exist"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "date" in data and data["description"]
    assert "calibration" in data and "global_guardrails" in data
    guardrails = {g["metric"]: g for g in data["global_guardrails"]}
    assert set(guardrails.keys()) == {"wer", "cer", "secs", "utmos", "failures"}
    for g in guardrails.values():
        assert g["rule"] == "not_worse" and "margin" in g and g["why"]

    expected_ids = {f"A{i}" for i in range(1, 9)}
    assert set(data["ablations"].keys()) == expected_ids
    for aid, item in data["ablations"].items():
        assert item["id"] == aid
        assert item["title"] and item["question"]
        assert "setup" in item and item["primary_metrics"]
        assert item["decision_rule"]
        assert isinstance(item["citations"], list) and len(item["citations"]) > 0
        assert item["status"] == "pre_registered"


def test_load_ablation_gates():
    from scripts.compare_ab import load_ablation_gates

    gates = load_ablation_gates("A3")
    assert gates["ablation_id"] == "A3"
    assert "Conditioning Masking Impact" in gates["title"]
    assert len(gates["gates"]) >= 5
    with pytest.raises(ValueError, match="Unknown ablation ID 'INVALID'"):
        load_ablation_gates("INVALID")


def test_compare_ab_with_ablation_gates(base):
    from scripts.compare_ab import load_ablation_gates

    tuned = tuned_from(base)
    gates = load_ablation_gates("A3")
    report = compare_ab(base, tuned, gates)
    assert report["passed"]
    assert report["gates_version"] == 1


def test_run_ablation_runner(base, tmp_path):
    from scripts.run_ablation import run_ablation

    tuned = tuned_from(base)
    base_file = tmp_path / "base.json"
    tuned_file = tmp_path / "tuned.json"
    base_file.write_text(json.dumps(base), encoding="utf-8")
    tuned_file.write_text(json.dumps(tuned), encoding="utf-8")

    # Dry run
    dry_report = run_ablation("A3", str(tuned_file), str(base_file), output_dir=str(tmp_path), dry_run=True)
    assert dry_report["dry_run"] is True and dry_report["passed"] is True

    # Full execution
    report = run_ablation("A3", str(tuned_file), str(base_file), output_dir=str(tmp_path))
    assert report["passed"] is True
    assert (tmp_path / "ab_report.json").is_file()


def test_failures_gate_requires_metadata_and_counts_utterance_once():
    from scripts.compare_ab import failures

    # Clean utterance
    clean = [{"failures": {"nan_audio": False, "empty_audio": False}}]
    assert failures(clean) == 0

    # Utterance with multiple failure flags is only counted once
    multiple_flags = [{"failures": {"nan_audio": True, "empty_audio": True}}]
    assert failures(multiple_flags) == 1

    # Missing failure metadata must raise KeyError rather than silently counting as clean
    missing_failures = [{}]
    with pytest.raises(KeyError):
        failures(missing_failures)
