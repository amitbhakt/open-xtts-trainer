"""Unit tests for Ablation A3 setup and conditioning masking modes."""

import os
import sys
import pytest
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_ablation_a3 import run_ablation_a3, resolve_dataset
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


def test_ablation_a3_dry_run():
    dataset_path = resolve_dataset(dry_run=True)
    report = run_ablation_a3(
        dataset_path=dataset_path,
        base_dir="xtts_base",
        output_dir="runs/test_a3_dry_run",
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["passed"] is True
    plan = report["plan"]
    assert plan["ablation_id"] == "A3"
    assert plan["variants"] == ["upstream", "per_sample"]
    assert plan["eval_batch_size"] == 16
