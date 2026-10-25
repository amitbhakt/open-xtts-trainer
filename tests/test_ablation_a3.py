"""Unit tests for Ablation A3 setup and conditioning masking modes."""

import os
import sys
import pytest
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_ablation_a3 import run_ablation_a3, resolve_dataset, merge_manifests
from xtts_trainer.models import build_dvae
from xtts_trainer.stage2_gpt.gpt_trainer import (
    ProjectGPTTrainer,
    build_gpt_trainer_config,
)

REAL_VOCAB = os.path.join(REPO_ROOT, "xtts_base", "vocab.json")
needs_vocab = pytest.mark.skipif(not os.path.exists(REAL_VOCAB), reason="needs xtts_base/vocab.json")
TINY_ARCH = dict(gpt_layers=1, gpt_n_model_channels=64, gpt_n_heads=2, decoder_input_dim=64)


@pytest.fixture
def dummy_dvae(tmp_path):
    p = tmp_path / "dvae.pth"
    torch.save(build_dvae().state_dict(), str(p))
    return str(p)


@needs_vocab
def test_masking_mode_validation(tmp_path, dummy_dvae):
    cfg = build_gpt_trainer_config(
        REAL_VOCAB, dummy_dvae, str(tmp_path / "out"), arch_overrides=TINY_ARCH
    )
    # Valid modes
    m_persample = ProjectGPTTrainer(cfg, masking_mode="per_sample")
    assert m_persample.masking_mode == "per_sample"

    m_upstream = ProjectGPTTrainer(cfg, masking_mode="upstream")
    assert m_upstream.masking_mode == "upstream"

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Invalid masking_mode 'invalid'"):
        ProjectGPTTrainer(cfg, masking_mode="invalid")


@needs_vocab
def test_masking_mode_forward_routing(tmp_path, dummy_dvae, monkeypatch):
    cfg = build_gpt_trainer_config(
        REAL_VOCAB, dummy_dvae, str(tmp_path / "out"), arch_overrides=TINY_ARCH
    )

    # Test upstream routing
    m_upstream = ProjectGPTTrainer(cfg, masking_mode="upstream")
    called_gpt = False

    def mock_gpt(*args, **kwargs):
        nonlocal called_gpt
        called_gpt = True
        return torch.tensor(0.0), torch.tensor(0.0), torch.zeros(1, 10, 1024)

    monkeypatch.setattr(m_upstream.xtts.gpt, "forward", mock_gpt)
    dummy_input = torch.tensor([[1, 2, 3]])
    m_upstream.forward(dummy_input, torch.tensor([3]), dummy_input, torch.tensor([3]), None, None, None)
    assert called_gpt is True

    # Test per_sample routing
    m_persample = ProjectGPTTrainer(cfg, masking_mode="per_sample")
    called_gpt_losses = False

    def mock_gpt_losses(*args, **kwargs):
        nonlocal called_gpt_losses
        called_gpt_losses = True
        return torch.tensor(0.0), torch.tensor(0.0), torch.zeros(1, 10, 1024)

    monkeypatch.setattr("xtts_trainer.stage2_gpt.gpt_trainer.gpt_losses", mock_gpt_losses)
    m_persample.forward(dummy_input, torch.tensor([3]), dummy_input, torch.tensor([3]), None, None, None)
    assert called_gpt_losses is True


def test_ablation_a3_dry_run(tmp_path):
    dummy_ds = tmp_path / "dataset_train.jsonl"
    dummy_ds.write_text('{"audio_path": "a.wav", "text": "hello"}\n', encoding="utf-8")
    report = run_ablation_a3(
        dataset_path=str(dummy_ds),
        base_dir="xtts_base",
        output_dir=str(tmp_path / "runs"),
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["passed"] is True
    plan = report["plan"]
    assert plan["ablation_id"] == "A3"
    assert plan["variants"] == ["upstream", "per_sample"]

    # Verify invalid prerequisite reports passed: false
    failed_report = run_ablation_a3(
        dataset_path=str(tmp_path / "nonexistent.jsonl"),
        base_dir="xtts_base",
        output_dir=str(tmp_path / "runs"),
        dry_run=True,
    )
    assert failed_report["dry_run"] is True
    assert failed_report["passed"] is False
    assert plan["eval_batch_size"] == 16


def test_merge_manifests(tmp_path):
    def make_dummy_manifest(seed: int, cer_val: float, ckpt_sha: str = "base_sha"):
        return {
            "label": f"dummy_s{seed}",
            "checkpoint_sha256": {"model.pth": ckpt_sha},
            "seeds": [seed],
            "runs": [
                {
                    "speaker_id": "spk1",
                    "language": "en",
                    "per_seed_results": {
                        str(seed): [
                            {
                                "id": "utt1",
                                "lang": "en",
                                "category": "short",
                                "cer": cer_val,
                                "wer": cer_val * 1.5,
                                "secs": 0.85,
                                "utmos": 3.8,
                                "duration_s": 2.5,
                                "duration_ratio": 1.0,
                                "failures": {"is_empty": False, "is_too_long": False, "is_nan": False},
                            },
                            {
                                "id": "utt2",
                                "lang": "en",
                                "category": "short",
                                "cer": cer_val + 0.01,
                                "wer": (cer_val + 0.01) * 1.5,
                                "secs": 0.88,
                                "utmos": 3.9,
                                "duration_s": 3.0,
                                "duration_ratio": 1.05,
                                "failures": {"is_empty": False, "is_too_long": False, "is_nan": False},
                            },
                        ]
                    },
                }
            ],
        }

    m1 = make_dummy_manifest(1, 0.05, "sha_base")
    m2 = make_dummy_manifest(2, 0.06, "sha_base")

    out_file = tmp_path / "merged.json"
    merged = merge_manifests([m1, m2], label="merged_test", output_path=str(out_file))

    assert merged["label"] == "merged_test"
    assert merged["seeds"] == [1, 2]
    assert len(merged["runs"]) == 1
    run0 = merged["runs"][0]
    assert "1" in run0["per_seed_results"]
    assert "2" in run0["per_seed_results"]
    assert "summary_by_language" in merged
    assert "en" in merged["summary_by_language"]

    from scripts.compare_ab import compare_ab, load_ablation_gates

    m1_cand = make_dummy_manifest(1, 0.04, "sha_cand")
    m2_cand = make_dummy_manifest(2, 0.05, "sha_cand")
    merged_cand = merge_manifests([m1_cand, m2_cand], label="merged_cand", output_path=str(tmp_path / "cand.json"))

    gates = load_ablation_gates("A3")
    res = compare_ab(merged, merged_cand, gates)
    assert res["pairs"] == 4
    assert res["passed"] is True


def test_resolve_dataset_explicit_validation(tmp_path, monkeypatch):
    ds_file = tmp_path / "dataset_train.jsonl"
    ds_file.write_text("{}", encoding="utf-8")

    # 1. Non-existent explicit dataset raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Supplied explicit --dataset does not exist"):
        resolve_dataset(str(tmp_path / "does_not_exist.jsonl"))

    # 2. Missing target-voice lock raises FileNotFoundError
    monkeypatch.setattr("os.path.exists", lambda p: str(p) == str(ds_file))
    with pytest.raises(FileNotFoundError, match="Target voice lock not found"):
        resolve_dataset(str(ds_file))

    # 3. Target-voice lock verification failure (checksum or malformed lock) raises RuntimeError
    monkeypatch.setattr("os.path.exists", lambda p: True)
    import eval.target_voices as tv_mod

    def mock_verify_fail(path):
        raise RuntimeError("checksum mismatch")

    monkeypatch.setattr(tv_mod, "verify_target_voices", mock_verify_fail)
    with pytest.raises(RuntimeError, match="Target voice lock verification failed"):
        resolve_dataset(str(ds_file))

    # Structural exceptions from malformed lock (KeyError, TypeError) are also caught
    def mock_verify_key_error(path):
        raise KeyError("missing 'voices' key in lock")

    monkeypatch.setattr(tv_mod, "verify_target_voices", mock_verify_key_error)
    with pytest.raises(RuntimeError, match="Target voice lock verification failed"):
        resolve_dataset(str(ds_file))

    def mock_verify_type_error(path):
        raise TypeError("expected dict for voice entry")

    monkeypatch.setattr(tv_mod, "verify_target_voices", mock_verify_type_error)
    with pytest.raises(RuntimeError, match="Target voice lock verification failed"):
        resolve_dataset(str(ds_file))

    # 4. Successful verification returns absolute dataset path
    def mock_verify_ok(path):
        return {"target_moira_fogarty": "clip.wav"}

    monkeypatch.setattr(tv_mod, "verify_target_voices", mock_verify_ok)
    resolved = resolve_dataset(str(ds_file))
    assert resolved == os.path.abspath(str(ds_file))

