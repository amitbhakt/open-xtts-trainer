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
    assert data["version"] == 2
    assert "date" in data and data["description"]
    assert "history" in data and len(data["history"]) >= 2
    for h in data["history"]:
        assert "version" in h and "date" in h and "summary" in h
    assert "calibration" in data and "global_guardrails" in data
    guardrails_list = data["global_guardrails"]
    assert len(guardrails_list) == len({g["metric"] for g in guardrails_list})
    guardrails = {g["metric"]: g for g in guardrails_list}
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
        assert "gates" in item and isinstance(item["gates"], list) and len(item["gates"]) > 0
        for g in item["gates"]:
            assert g["metric"] in {"wer", "cer", "secs", "utmos", "duration_ratio", "failures"}
            assert g["rule"] in ("not_worse", "better")
            assert "margin" in g and "why" in g


def test_gate_category_filtering(base):
    from scripts.compare_ab import check_gate, utterances

    b_utts = utterances(base)
    pairs = [(b_utts[k], b_utts[k]) for k in sorted(b_utts)]

    # Valid category slice - only scores matching pairs
    short_gate = {"metric": "cer", "rule": "not_worse", "margin": 0.01, "category": "short"}
    res = check_gate(short_gate, pairs)
    assert res["passed"]
    assert res["n"] == 20  # 10 short sentences x 2 seeds

    # Unknown category raises ValueError
    unknown_gate = {"metric": "cer", "rule": "not_worse", "margin": 0.01, "category": "nonexistent_category"}
    with pytest.raises(ValueError, match="not in the manifests"):
        check_gate(unknown_gate, pairs)


def test_load_ablation_gates():
    from scripts.compare_ab import load_ablation_gates

    gates = load_ablation_gates("A3")
    assert gates["ablation_id"] == "A3"
    assert "Conditioning Masking Impact" in gates["title"]
    # 5 global guardrails + 2 ablation-specific gates
    assert len(gates["gates"]) == 7
    with pytest.raises(ValueError, match="Unknown ablation ID 'INVALID'"):
        load_ablation_gates("INVALID")


def test_compare_ab_with_ablation_gates(base):
    from scripts.compare_ab import load_ablation_gates

    tuned = tuned_from(base)
    gates = load_ablation_gates("A3")
    report = compare_ab(base, tuned, gates)
    assert report["passed"]
    assert report["gates_version"] == 2


def test_compare_ab_cli_rejects_gates_and_ablation_together(base, tmp_path):
    import subprocess

    base_file = tmp_path / "base.json"
    tuned_file = tmp_path / "tuned.json"
    base_file.write_text(json.dumps(base), encoding="utf-8")
    tuned_file.write_text(json.dumps(tuned_from(base)), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/compare_ab.py",
        str(base_file),
        str(tuned_file),
        "--gates",
        "eval/gates.json",
        "--ablation",
        "A3",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Cannot specify both --gates and --ablation" in res.stderr


def test_gate_validation_and_empty_gates(base):
    from scripts.compare_ab import check_gate, utterances, validate_gate

    # Unknown key rejected
    with pytest.raises(ValueError, match="Unknown keys in gate"):
        validate_gate({"metric": "cer", "rule": "not_worse", "margin": 0.01, "invalid_key": True})

    # Unknown metric rejected
    with pytest.raises(ValueError, match="Invalid or missing metric in gate"):
        validate_gate({"metric": "unknown_metric", "rule": "not_worse", "margin": 0.01})

    # Unknown rule rejected
    with pytest.raises(ValueError, match="Invalid or missing rule in gate"):
        validate_gate({"metric": "cer", "rule": "unknown_rule", "margin": 0.01})

    # Empty gates list rejected in compare_ab
    with pytest.raises(ValueError, match="non-empty 'gates' list"):
        compare_ab(base, tuned_from(base), {"gates": []})


def test_gate_languages_filtering(base):
    from scripts.compare_ab import check_gate, utterances

    b_utts = utterances(base)
    pairs = [(b_utts[k], b_utts[k]) for k in sorted(b_utts)]

    # English gate on English pairs passes
    en_gate = {"metric": "cer", "rule": "not_worse", "margin": 0.01, "languages": ["en"]}
    res = check_gate(en_gate, pairs)
    assert res["passed"]
    assert res["n"] == 120

    # Scalar string language selector is normalized and passes
    scalar_en_gate = {"metric": "cer", "rule": "not_worse", "margin": 0.01, "languages": "en"}
    scalar_res = check_gate(scalar_en_gate, pairs)
    assert scalar_res["passed"]
    assert scalar_res["n"] == 120

    # Non-matching language raises ValueError
    de_gate = {"metric": "cer", "rule": "not_worse", "margin": 0.01, "languages": ["de"]}
    with pytest.raises(ValueError, match="not in the manifests"):
        check_gate(de_gate, pairs)


def test_run_ablation_find_default_baseline():
    from scripts.run_ablation import find_default_baseline

    # Existing baselines in eval/baselines/ verify language metadata and filename
    for lang in ("en", "de", "es", "fr", "hi"):
        path = find_default_baseline(12, lang)
        assert path is not None and path.endswith(f"base_xtts_s12_{lang}.json")
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["languages"] == [lang]
        assert meta["subset_size"] == 12

    path_s20 = find_default_baseline(20, "en")
    assert path_s20 is not None and path_s20.endswith("base_xtts_s20_en.json")
    with open(path_s20, "r", encoding="utf-8") as f:
        meta_s20 = json.load(f)
    assert meta_s20["languages"] == ["en"]
    assert meta_s20["subset_size"] == 20

    # Missing subset size or unsupported language returns None without falling back to a different subset
    assert find_default_baseline(99, "en") is None
    assert find_default_baseline(12, "nonexistent_lang") is None


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


def test_run_ablation_runner_dry_run_rejects_invalid_inputs(base, tmp_path):
    from scripts.run_ablation import run_ablation

    base_file = tmp_path / "base.json"
    base_file.write_text(json.dumps(base), encoding="utf-8")

    # 1. Identical checkpoint rejection
    identical_candidate = tmp_path / "identical.json"
    identical_candidate.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="same checkpoint"):
        run_ablation("A3", str(identical_candidate), str(base_file), output_dir=str(tmp_path), dry_run=True)

    # 2. Mismatched input keys (e.g. batch_size)
    mismatched_tuned = tuned_from(base)
    mismatched_tuned["batch_size"] = 99
    mismatched_file = tmp_path / "mismatched.json"
    mismatched_file.write_text(json.dumps(mismatched_tuned), encoding="utf-8")
    with pytest.raises(ValueError, match="manifests were measured on different inputs"):
        run_ablation("A3", str(mismatched_file), str(base_file), output_dir=str(tmp_path), dry_run=True)

    # 3. Mismatched / unpaired utterances
    unpaired_tuned = tuned_from(base)
    unpaired_tuned["runs"] = unpaired_tuned["runs"][:-1]
    unpaired_file = tmp_path / "unpaired.json"
    unpaired_file.write_text(json.dumps(unpaired_tuned), encoding="utf-8")
    with pytest.raises(ValueError, match="utterance sets differ"):
        run_ablation("A3", str(unpaired_file), str(base_file), output_dir=str(tmp_path), dry_run=True)

    # 4. Multi-language candidate auto-detection rejection
    multi_lang_tuned = tuned_from(base)
    multi_lang_tuned["languages"] = ["en", "es"]
    multi_lang_file = tmp_path / "multi_lang.json"
    multi_lang_file.write_text(json.dumps(multi_lang_tuned), encoding="utf-8")
    with pytest.raises(ValueError, match="only supports single-language candidates"):
        run_ablation("A3", str(multi_lang_file), output_dir=str(tmp_path), dry_run=True)


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


def test_failures_gate_rejects_empty_selection():
    from scripts.compare_ab import check_gate

    f_gate = {"metric": "failures", "rule": "not_worse", "margin": 0}
    with pytest.raises(ValueError, match="matched 0 utterances; cannot evaluate failures on an empty selection"):
        check_gate(f_gate, [])


def test_run_ablation_scalar_selectors(base, tmp_path, monkeypatch):
    from scripts.run_ablation import run_ablation

    tuned = tuned_from(base)
    base_file = tmp_path / "base.json"
    tuned_file = tmp_path / "tuned.json"
    base_file.write_text(json.dumps(base), encoding="utf-8")
    tuned_file.write_text(json.dumps(tuned), encoding="utf-8")

    speaker_id = base["runs"][0]["speaker_id"]
    custom_prereg = tmp_path / "prereg.json"
    custom_prereg.write_text(json.dumps({
        "version": 2,
        "date": "2026-10-25",
        "description": "test",
        "calibration": {},
        "global_guardrails": [],
        "ablations": {
            "CUSTOM": {
                "id": "CUSTOM",
                "title": "Custom Scalar",
                "decision_rule": "test",
                "gates": [
                    {"metric": "cer", "rule": "not_worse", "margin": 0.01, "languages": "en", "voices": speaker_id}
                ]
            }
        }
    }), encoding="utf-8")

    monkeypatch.setattr("scripts.run_ablation.PREREG_PATH", str(custom_prereg))
    res = run_ablation("CUSTOM", str(tuned_file), str(base_file), output_dir=str(tmp_path), dry_run=True)
    assert res["dry_run"] is True and res["passed"] is True
