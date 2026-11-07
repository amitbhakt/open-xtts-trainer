"""O1 (finishing ablation A3): the runner's locked inputs and the analysis's locked decision rules."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import analyze_o1 as o1  # noqa: E402
from scripts import run_o1  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A3 = ROOT / "docs" / "evidence" / "ablation_a3"
VOICES = ("arctic_slt", "arctic_clb", "arctic_bdl", "arctic_awb", "arctic_ksp")
SEEDS = (1, 2, 3)
OLD = sorted(o1.A3_SENTENCE_IDS)
NEW = [f"en_{i:02d}" for i in range(1, 61) if f"en_{i:02d}" not in o1.A3_SENTENCE_IDS]


def _take(sid, seed, wer=0.0, cer=0.0, category="harvard"):
    return {"id": sid, "lang": "en", "seed": seed, "category": category, "wer": wer, "cer": cer, "secs": 0.6,
            "utmos": 4.1, "duration_ratio": 1.0, "duration_s": 3.0, "asr_text": "text",
            "failures": {"empty": False, "too_long": False, "nan": False}}


def _manifest(label, sha, runaways=(), bad_short=()):
    """60 sentences x 5 voices x 3 seeds; `runaways` and `bad_short` are (voice, sentence, seed) keys."""
    runs = []
    for v in VOICES:
        per_seed = {}
        for s in SEEDS:
            per_seed[str(s)] = [
                _take(sid, s, wer=5.0 if (v, sid, s) in runaways else 0.0,
                      cer=0.9 if (v, sid, s) in bad_short else 0.0,
                      category="short" if sid in ("en_51", "en_54", "en_55") else "harvard")
                for sid in OLD + NEW]
        runs.append({"speaker_id": v, "language": "en", "per_seed_results": per_seed})
    return {"label": label, "checkpoint_sha256": {"model.pth": sha}, "metrics_version": 2, "eval_set_sha256": "x",
            "voices": "arctic", "reference_voices": [], "seeds": list(SEEDS), "subset_size": 60, "languages": ["en"],
            "asr_model": "openai/whisper-large-v3", "batch_size": 16, "runs": runs}


def test_old_sentences_are_exactly_the_twenty_a3_scored():
    manifest = json.loads((A3 / "manifest_upstream_s20_base.json").read_text(encoding="utf-8"))
    scored = {r["id"] for run in manifest["runs"] for rs in run["per_seed_results"].values() for r in rs}
    assert scored == o1.A3_SENTENCE_IDS


def test_old_sentences_are_the_first_twenty_of_the_sixty_o1_scores():
    from eval.dataset import DEFAULT_EVAL_SET_PATH, load_eval_set
    from eval.harness import select_subset
    english = load_eval_set(DEFAULT_EVAL_SET_PATH)["english_sentences"]
    sixty = select_subset(english, run_o1.SUBSET_SIZE)
    assert len(sixty) == 60 and {s["id"] for s in sixty[:20]} == o1.A3_SENTENCE_IDS


def test_runner_expects_the_exact_checkpoints_a3_scored():
    for (arm, seed), sha in run_o1.EXPECTED_MODEL_SHA256.items():
        path = A3 / "per_seed_manifests" / f"manifest_{arm}_base_seed_{seed}_s20.json"
        assert json.loads(path.read_text(encoding="utf-8"))["checkpoint_sha256"]["model.pth"] == sha


def test_runner_refuses_a_checkpoint_that_differs(tmp_path, monkeypatch):
    for seed in SEEDS:
        d = tmp_path / "upstream" / f"seed_{seed}" / "export"
        d.mkdir(parents=True)
        (d / "model.pth").write_bytes(b"not the A3 model")
    with pytest.raises(SystemExit, match="not the"):
        run_o1.verify_checkpoints("upstream", a3_dir=str(tmp_path))


def test_runner_accepts_matching_checkpoints(tmp_path, monkeypatch):
    blobs = {}
    for seed in SEEDS:
        d = tmp_path / "per_sample" / f"seed_{seed}" / "export"
        d.mkdir(parents=True)
        data = f"model {seed}".encode()
        (d / "model.pth").write_bytes(data)
        blobs[("per_sample", seed)] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(run_o1, "EXPECTED_MODEL_SHA256", blobs)
    run_o1.verify_checkpoints("per_sample", a3_dir=str(tmp_path))


def test_eval_command_scores_sixty_sentences_with_the_training_seed():
    cmd = run_o1.eval_command("upstream", 2, "out.json")
    assert cmd[cmd.index("--subset_size") + 1] == "60"
    assert cmd[cmd.index("--seeds") + 1] == "2"
    assert cmd[cmd.index("--checkpoint_dir") + 1].endswith("upstream/seed_2/export")
    assert cmd[cmd.index("--batch_size") + 1] == "16"


def test_both_arms_run_in_one_process_alternating_by_seed():
    # One container for both arms (amendment A1): a container difference can't fall on one arm only
    assert run_o1.run_plan() == [("upstream", 1), ("per_sample", 1), ("upstream", 2), ("per_sample", 2),
                                 ("upstream", 3), ("per_sample", 3)]


@pytest.mark.parametrize("earlier", ["per_seed", "environment"])
def test_runner_refuses_to_overwrite_any_earlier_file(tmp_path, monkeypatch, earlier):
    path = Path(run_o1.per_seed_path(str(tmp_path), "per_sample", 2) if earlier == "per_seed"
                else run_o1.environment_path(str(tmp_path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_o1.py", "--output_dir", str(tmp_path)])
    monkeypatch.setattr(run_o1, "verify_checkpoints", lambda arm: pytest.fail("must refuse before any work"))
    with pytest.raises(SystemExit, match="never overwrites"):
        run_o1.main()


def test_mcnemar_one_sided_values():
    assert o1.mcnemar_one_sided(0, 0) == 1.0
    assert o1.mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert o1.mcnemar_one_sided(4, 0) == pytest.approx(1 / 16)
    assert o1.mcnemar_one_sided(0, 3) == 1.0


def test_decision_tiers():
    assert o1.decide(0.01, 0.002) == "established"
    assert o1.decide(0.01, 0.0) == "supported, concentrated on few sentences"
    assert o1.decide(0.06, 0.002) == "not confirmed"


def test_runaways_spread_over_new_sentences_are_established():
    runaways = {(VOICES[i % 5], NEW[i], 1) for i in range(8)}
    report = o1.analyze(_manifest("up", "a", runaways), _manifest("ps", "b"))
    prim = report["primary_new_sentences"]
    assert prim["pairs"] == 600 and prim["discordant"] == {"upstream_only": 8, "per_sample_only": 0}
    assert prim["decision"] == "established"


def test_runaways_on_one_new_sentence_are_only_supported():
    runaways = {(v, NEW[0], s) for v in VOICES for s in SEEDS[:2]}  # 10 takes, all one sentence
    prim = o1.analyze(_manifest("up", "a", runaways), _manifest("ps", "b"))["primary_new_sentences"]
    assert prim["mcnemar_one_sided_p"] < o1.ALPHA
    assert prim["decision"] == "supported, concentrated on few sentences"


def test_runaways_on_sentences_a3_already_saw_never_count():
    runaways = {(v, sid, s) for v in VOICES for sid in OLD for s in SEEDS}
    prim = o1.analyze(_manifest("up", "a", runaways), _manifest("ps", "b"))["primary_new_sentences"]
    assert prim["runaways"] == {"upstream": 0, "per_sample": 0}
    assert prim["decision"] == "not confirmed"


def test_secondary_counts_only_new_short_prompts():
    bad = {(v, "en_54", 1) for v in VOICES} | {(v, "en_51", 1) for v in VOICES}  # en_51 is an old sentence
    sec = o1.analyze(_manifest("up", "a", bad_short=bad), _manifest("ps", "b"))["secondary_short_new_sentences"]
    assert sec["pairs"] == 2 * 5 * 3 and sec["bad"] == {"upstream": 5, "per_sample": 0}


def test_a3_gates_run_on_all_pairs():
    gates = o1.analyze(_manifest("up", "a"), _manifest("ps", "b"))["a3_gates_all_pairs"]
    assert gates["pairs"] == 900 and "passed" in gates


def test_unpaired_or_incomplete_arms_are_refused():
    up, ps = _manifest("up", "a"), _manifest("ps", "b")
    ps["runs"][0]["per_seed_results"]["1"].pop()
    with pytest.raises(ValueError, match="not paired"):
        o1.analyze(up, ps)
    with pytest.raises(ValueError, match="expected 600 new pairs"):
        o1.new_pairs({("en", "v", "en_54", 1): (None, None)})


# --- The committed O1 evidence (docs/evidence/o1_a3_completion) -----------------------------------------------------
O1 = ROOT / "docs" / "evidence" / "o1_a3_completion"
RUN = O1 / "attempt2"


def _sums():
    out = {}
    for line in (O1 / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        out[rel] = digest
    return out


def test_o1_evidence_matches_its_checksums_and_every_file_is_listed():
    sums = _sums()
    on_disk = {p.relative_to(O1).as_posix() for d in ("attempt1_failed", "attempt2") for p in (O1 / d).rglob("*")
               if p.is_file()}
    assert on_disk == set(sums)
    for rel, digest in sums.items():
        assert hashlib.sha256((O1 / rel).read_bytes()).hexdigest() == digest, rel


def test_o1_inputs_match_their_locks():
    lock = json.loads((ROOT / "eval" / "dataset" / "reference_voices.lock.json").read_text(encoding="utf-8"))
    for path in sorted((RUN / "per_seed").glob("*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        arm = "per_sample" if "per_sample" in path.name else "upstream"
        assert m["checkpoint_sha256"]["model.pth"] == run_o1.EXPECTED_MODEL_SHA256[(arm, m["seeds"][0])]
        assert m["eval_set_sha256"] == lock["eval_set_sha256"]
        assert {v["speaker_id"]: v["sha256"] for v in m["reference_voices"]} == {
            k: v["clip_sha256"] for k, v in lock["voices"].items()}
        assert m["subset_size"] == run_o1.SUBSET_SIZE and not m["git"]["is_dirty"]


def test_o1_report_reproduces_from_the_committed_manifests():
    def load(p):
        return json.loads(p.read_text(encoding="utf-8"))

    again = o1.analyze(load(RUN / "manifest_upstream_s60.json"), load(RUN / "manifest_per_sample_s60.json"),
                       load(A3 / "manifest_upstream_s20_base.json"), load(A3 / "manifest_per_sample_s20_base.json"))
    assert json.loads(json.dumps(again)) == load(RUN / "o1_report.json")
    assert again["primary_new_sentences"]["decision"] == "not confirmed"
