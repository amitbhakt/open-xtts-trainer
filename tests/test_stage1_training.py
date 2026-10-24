"""Stage 1 (DVAE): data, validation metrics, collapse gate, frozen codebook, best checkpoint and resume."""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest
import soundfile as sf
import torch

from dataset_helpers import write_test_dataset
from xtts_trainer.config import DVAEConfig
from xtts_trainer.models import REQUIRES_GPT_RETRAIN_KEY, build_dvae, safe_torch_load
from xtts_trainer.spec import SPEC
from xtts_trainer.stage1_dvae.dataset import MelDataset, collate_mels
from xtts_trainer.stage1_dvae.train import CodebookCollapseError, codebook_stats, train_dvae

SR = SPEC.dvae_mel.sample_rate
FACTOR = SPEC.dvae.downsample_factor


def _clips(tmp_path, n=6):
    rng = np.random.RandomState(0)
    paths = []
    for i in range(n):
        t = np.arange(int(SR * (1.5 + 0.3 * i))) / SR
        audio = sum(0.1 * np.sin(2 * np.pi * f * t) for f in rng.uniform(100, 4000, 5)) + 0.01 * rng.randn(len(t))
        path = str(tmp_path / f"c{i}.wav")
        sf.write(path, audio.astype("float32"), SR)
        paths.append(path)
    return paths, write_test_dataset(str(tmp_path), paths)


def _base(tmp_path):
    torch.manual_seed(0)
    path = str(tmp_path / "base_dvae.pth")
    torch.save(build_dvae(num_tokens=64, codebook_dim=32, hidden_dim=32).state_dict(), path)
    return path


def _config(**kw):
    defaults = dict(batch_size=2, epochs=2, codebook_dim=32, hidden_dim=32, val_fraction=0.34, val_every_steps=2,
                    min_active_codes=1, lr_warmup_steps=2, frozen_codebook=True)
    defaults.update(kw)
    return DVAEConfig(**defaults)


def test_dvae_forward_backward():
    dvae = build_dvae(num_tokens=512, codebook_dim=64, hidden_dim=64).train()
    recon, commit, _ = dvae(torch.randn(2, 80, 32))
    (recon.mean() + commit.mean()).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in dvae.parameters())


def test_mels_are_trimmed_not_padded(tmp_path):
    paths, _ = _clips(tmp_path, 3)
    train = MelDataset(paths, max_mel_frames=130)
    lengths = [train[i].shape[-1] for i in range(3)]
    assert all(n % FACTOR == 0 and n <= 128 for n in lengths)
    full = MelDataset(paths, max_mel_frames=10_000, random_crop=False)
    n_frames = full[0].shape[-1]
    assert n_frames % FACTOR == 0 and abs(n_frames - SR * 1.5 / SPEC.dvae_mel.hop_length) < FACTOR + 2
    batch = collate_mels([full[0], full[1]])
    assert batch.shape[-1] == min(full[0].shape[-1], full[1].shape[-1])  # cropped to the shorter, never padded


def test_codebook_stats():
    assert codebook_stats(torch.arange(8).repeat(3), 16) == {"active_codes": 8, "perplexity": pytest.approx(8.0)}
    assert codebook_stats(torch.zeros(10, dtype=torch.long), 16) == {"active_codes": 1, "perplexity": 1.0}


def test_codebook_collapse_fails_the_run(tmp_path):
    """Without a frozen codebook, EMA updates (F1) collapse this small random DVAE to one code in 2 steps."""
    _, dataset = _clips(tmp_path)
    with pytest.raises(CodebookCollapseError, match="codebook collapsed"):
        train_dvae(dataset, str(tmp_path / "out"), _config(frozen_codebook=False), _base(tmp_path), device="cpu")
    metrics = [json.loads(l) for l in Path(tmp_path / "out" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metrics[-1]["active_codes"] == 1


def test_frozen_codebook_best_checkpoint_and_metadata(tmp_path):
    _, dataset = _clips(tmp_path)
    base = _base(tmp_path)
    best_path = train_dvae(dataset, str(tmp_path / "out"), _config(epochs=3), base, device="cpu")
    ckpt = safe_torch_load(best_path)
    assert ckpt[REQUIRES_GPT_RETRAIN_KEY] is True and ckpt["codebook_frozen"] is True
    assert torch.equal(ckpt["dvae"]["codebook.embed"], torch.load(base)["codebook.embed"])  # frozen: unchanged
    metrics = [json.loads(l) for l in Path(tmp_path / "out" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ckpt["val"]["val_recon"] == min(m["val_recon"] for m in metrics)  # best by validation loss
    assert ckpt["reference"]["code_drift"] == 0.0 and all(m["code_drift"] is not None for m in metrics)
    assert ckpt["base_dvae_sha256"]
    assert len({m["step"] for m in metrics}) == len(metrics)  # no step validated twice


def test_resume_matches_an_uninterrupted_run(tmp_path):
    _, dataset = _clips(tmp_path)
    base = _base(tmp_path)
    straight = safe_torch_load(train_dvae(dataset, str(tmp_path / "a"), _config(epochs=3), base, device="cpu"))
    train_dvae(dataset, str(tmp_path / "b"), _config(epochs=2), base, device="cpu")
    resumed = safe_torch_load(train_dvae(dataset, str(tmp_path / "b"), _config(epochs=3), base, resume=True, device="cpu"))
    assert straight["val"] == resumed["val"]
    assert all(torch.equal(straight["dvae"][k], resumed["dvae"][k]) for k in straight["dvae"])


def test_scratch_training_writes_a_loadable_checkpoint(tmp_path):
    _, dataset = _clips(tmp_path)
    cfg = _config(num_tokens=64, frozen_codebook=False, epochs=1)
    ckpt = safe_torch_load(train_dvae(dataset, str(tmp_path / "out"), cfg, device="cpu"))
    assert ckpt["base_dvae_sha256"] is None and ckpt["val"]["code_drift"] is None
    build_dvae(num_tokens=64, codebook_dim=32, hidden_dim=32).load_state_dict(ckpt["dvae"], strict=True)
    assert math.isfinite(ckpt["val"]["val_recon"])


def test_best_checkpoint_is_the_best_validation_not_the_last(tmp_path, monkeypatch):
    import xtts_trainer.stage1_dvae.train as train_module

    scripted = iter([2.0, 1.0, 0.5, 0.9])  # reference, then validations: improves, then gets worse

    def fake_evaluate(dvae, val_mels, base_dvae=None):
        return {"active_codes": 10, "perplexity": 5.0, "val_recon": next(scripted), "code_drift": 0.0}

    monkeypatch.setattr(train_module, "evaluate_dvae", fake_evaluate)
    _, dataset = _clips(tmp_path)
    best = safe_torch_load(train_dvae(dataset, str(tmp_path / "out"), _config(epochs=3), _base(tmp_path), device="cpu"))
    assert best["val"]["val_recon"] == 0.5 and best["val"]["step"] == 4


def test_diagnose_dvae_reports_fit_and_drift(tmp_path):
    from scripts.diagnose_dvae import diagnose

    _, dataset = _clips(tmp_path)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    _, reference = _clips(ref_dir, 3)
    base, other = str(tmp_path / "full_base.pth"), str(tmp_path / "other.pth")  # full size, as load_dvae builds
    torch.manual_seed(4)
    torch.save(build_dvae().state_dict(), base)
    torch.manual_seed(5)
    torch.save(build_dvae().state_dict(), other)
    report = diagnose(dataset, base, reference_dataset=reference, compare=other)
    assert report["target"]["active_codes"] >= 1 and report["target"]["code_drift"] > 0
    assert report["target_vs_reference"]["recon_ratio"] > 0
    same = diagnose(dataset, base, compare=base)
    assert same["target"]["code_drift"] == 0.0
