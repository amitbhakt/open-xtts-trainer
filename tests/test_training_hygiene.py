"""Phase 7 step 2: shared training hygiene (xtts_trainer/training.py)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pytest
import torch

from xtts_trainer.data import DatasetRecord
from xtts_trainer.training import (
    Precision, load_training_state, save_training_state, seed_everything, split_records, warmup_lambda,
)


def _records(spec):
    return [DatasetRecord(audio=f"{spk}_{i}.wav", text="x", lang="en", speaker=spk, duration_s=1.0, sr=24000,
                          transcript_source="human") for spk, n in spec.items() for i in range(n)]


def test_split_is_utterance_level_stratified_and_seeded():
    records = _records({"a": 10, "b": 4, "c": 1})
    train, val = split_records(records, 0.2, seed=7)
    assert {r.audio for r in train}.isdisjoint({r.audio for r in val})
    assert len(train) + len(val) == len(records)
    assert sorted(r.speaker for r in val) == ["a", "a", "b"]  # every speaker with >= 2 clips is validated
    assert [r.audio for r in split_records(records, 0.2, seed=7)[1]] == [r.audio for r in val]
    assert [r.audio for r in split_records(records, 0.2, seed=8)[1]] != [r.audio for r in val]
    with pytest.raises(ValueError, match="no validation clips"):
        split_records(_records({"solo": 1}), 0.2, seed=0)


def test_warmup_then_decay():
    f = warmup_lambda(4, decay_per_step=0.5)
    assert [f(s) for s in range(7)] == [0.25, 0.5, 0.75, 1.0, 1.0, 0.5, 0.25]  # full LR, then decay


def test_precision_is_off_on_cpu():
    p = Precision("cpu", enabled=True)
    assert not p.enabled and not p.scaler().is_enabled()


def _run(model, opt, sched, steps):
    for _ in range(steps):
        x = torch.randn(8, 4)
        loss = model(x).pow(2).mean()  # dropout makes the result depend on the RNG stream
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


def _fresh():
    seed_everything(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Dropout(0.5), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_lambda(2, 0.9))
    return model, opt, sched


def test_resume_is_exact(tmp_path):
    """Stop after 3 steps, restore, run 3 more: identical to 6 uninterrupted steps."""
    model, opt, sched = _fresh()
    _run(model, opt, sched, 6)
    straight = [p.detach().clone() for p in model.parameters()]

    model, opt, sched = _fresh()
    scaler = Precision("cpu", True).scaler()
    _run(model, opt, sched, 3)
    path = str(tmp_path / "state.pth")
    save_training_state(path, models={"m": model}, optimizers={"o": opt}, schedulers={"s": sched},
                        scalers={"amp": scaler}, step=3, epoch=1, best=0.5, extra={"note": 1})

    model, opt, sched = _fresh()
    torch.manual_seed(123)  # a different RNG stream, which the restore must replace
    restored = load_training_state(path, models={"m": model}, optimizers={"o": opt}, schedulers={"s": sched},
                                   scalers={"amp": scaler})
    assert restored == {"step": 3, "epoch": 1, "best": 0.5, "extra": {"note": 1}}
    _run(model, opt, sched, 3)
    assert all(torch.equal(a, b) for a, b in zip(straight, model.parameters()))


def test_resume_refuses_mismatched_runs(tmp_path):
    model, opt, sched = _fresh()
    path = save_training_state(str(tmp_path / "s.pth"), models={"m": model}, optimizers={"o": opt},
                               schedulers={"s": sched}, scalers={}, step=0, epoch=0, best=None)
    with pytest.raises(KeyError, match="models"):
        load_training_state(path, models={"m": model, "extra": model}, optimizers={"o": opt},
                            schedulers={"s": sched}, scalers={})
    other = torch.nn.Linear(4, 2)
    with pytest.raises(RuntimeError):
        load_training_state(path, models={"m": other}, optimizers={"o": opt}, schedulers={"s": sched}, scalers={})
