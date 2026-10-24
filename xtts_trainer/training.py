"""Training hygiene shared by the owned stages, 1 (DVAE) and 3 (HiFi-GAN) (docs/recipe_design.md §6, phase 7).

- split_records: utterance-level train/validation split, stratified by speaker, seeded.
- seed_everything: python, numpy and torch (CPU and CUDA) generators.
- warmup_lambda: linear learning-rate warm-up, then exponential decay per step.
- Precision: autocast plus a GradScaler, enabled only on CUDA.
- save_training_state / load_training_state: full resume (models, optimizers, schedulers, scalers, step,
  epoch, best validation value and every RNG state), loaded strictly.
Stage 2 gets these from the coqui Trainer (stage2_gpt/train.py).
"""

import math
import os
import random
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from xtts_trainer.data import DatasetRecord
from xtts_trainer.models import load_state_strict, safe_torch_load, stamp

TRAINING_STATE_VERSION = 1


def seed_everything(seed: int) -> torch.Generator:
    """Seed every generator a run uses; returns a torch.Generator for DataLoader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def split_records(records: Sequence[DatasetRecord], val_fraction: float, seed: int
                  ) -> Tuple[List[DatasetRecord], List[DatasetRecord]]:
    """Utterance-level split, stratified by speaker: each speaker with >= 2 clips gives
    max(1, round(n * val_fraction)) clips to validation. Same seed and records -> same split."""
    rng = random.Random(seed)
    by_speaker: Dict[str, List[DatasetRecord]] = {}
    for r in records:
        by_speaker.setdefault(r.speaker, []).append(r)
    train, val = [], []
    for speaker in sorted(by_speaker):
        clips = sorted(by_speaker[speaker], key=lambda r: r.audio)
        rng.shuffle(clips)
        n_val = max(1, round(len(clips) * val_fraction)) if len(clips) >= 2 else 0
        n_val = min(n_val, len(clips) - 1)
        val += clips[:n_val]
        train += clips[n_val:]
    if not val:
        raise ValueError("no validation clips: some speaker needs at least 2 clips (the best checkpoint is "
                         "chosen by validation metric)")
    return train, val


def warmup_lambda(warmup_steps: int, decay_per_step: float = 1.0) -> Callable[[int], float]:
    """LR multiplier: linear ramp over warmup_steps, then decay_per_step ** (steps after warm-up)."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return decay_per_step ** (step - warmup_steps)
    return factor


def decay_per_step(decay_per_epoch: float, steps_per_epoch: int) -> float:
    return decay_per_epoch ** (1.0 / max(1, steps_per_epoch))


class Precision:
    """Mixed precision on CUDA only (fp16 autocast + loss scaling); a no-op elsewhere."""

    def __init__(self, device: str, enabled: bool):
        self.enabled = bool(enabled and str(device).startswith("cuda"))
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def autocast(self):
        return torch.autocast(self.device_type, dtype=torch.float16) if self.enabled else nullcontext()

    def scaler(self) -> "torch.amp.GradScaler":
        return torch.amp.GradScaler("cuda", enabled=self.enabled)


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(path: str, *, models: Dict[str, torch.nn.Module], optimizers: Dict[str, Any],
                        schedulers: Dict[str, Any], scalers: Dict[str, Any], step: int, epoch: int,
                        best: Optional[float], extra: Optional[Dict[str, Any]] = None) -> str:
    """Everything needed to continue a run exactly where it stopped (atomic write)."""
    payload = stamp({
        "training_state_version": TRAINING_STATE_VERSION,
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "schedulers": {k: s.state_dict() for k, s in schedulers.items()},
        "scalers": {k: s.state_dict() for k, s in scalers.items()},
        "step": step, "epoch": epoch, "best": best, "rng": rng_state(), "extra": extra or {},
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def load_training_state(path: str, *, models: Dict[str, torch.nn.Module], optimizers: Dict[str, Any],
                        schedulers: Dict[str, Any], scalers: Dict[str, Any]) -> Dict[str, Any]:
    """Restore a save_training_state checkpoint in place; every component must be present.

    Returns {"step", "epoch", "best", "extra"}. RNG states are restored last, so the run continues with the
    same random stream it would have had.
    """
    state = safe_torch_load(path, map_location="cpu")
    if state.get("training_state_version") != TRAINING_STATE_VERSION:
        raise ValueError(f"{path} is not a training state (version {state.get('training_state_version')})")
    for group, live in (("models", models), ("optimizers", optimizers), ("schedulers", schedulers), ("scalers", scalers)):
        if set(state[group]) != set(live):
            raise KeyError(f"{path} {group} {sorted(state[group])} != run's {sorted(live)}")
    for name, module in models.items():
        load_state_strict(module, state["models"][name], f"{name} ({path})")
    for name, opt in optimizers.items():
        opt.load_state_dict(state["optimizers"][name])
    for name, sched in schedulers.items():
        sched.load_state_dict(state["schedulers"][name])
    for name, scaler in scalers.items():
        scaler.load_state_dict(state["scalers"][name])
    set_rng_state(state["rng"])
    return {"step": state["step"], "epoch": state["epoch"], "best": state["best"], "extra": state["extra"]}


def is_improvement(value: float, best: Optional[float]) -> bool:
    return math.isfinite(value) and (best is None or value < best)
