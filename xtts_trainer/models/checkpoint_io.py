"""Checkpoint loading and stamping. Every load is strict; every file this project writes carries spec_hash."""

from typing import Any, Dict, Optional

import torch

from xtts_trainer.spec.xtts_spec import SPEC, spec_hash

SPEC_HASH_KEY = "spec_hash"


def safe_torch_load(path: str, map_location="cpu"):
    """Loads a PyTorch checkpoint safely across PyTorch 2.0 to 2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def stamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add the current spec_hash to a checkpoint or latent payload before saving."""
    payload[SPEC_HASH_KEY] = spec_hash()
    return payload


def check_spec_hash(payload: Any, source: str) -> None:
    """Refuse files written under a different spec. Upstream files carry no spec_hash and are accepted."""
    if isinstance(payload, dict) and SPEC_HASH_KEY in payload and payload[SPEC_HASH_KEY] != spec_hash():
        raise RuntimeError(
            f"{source} was written with spec_hash {payload[SPEC_HASH_KEY]}, but the current spec is {spec_hash()}. "
            f"It was produced with different XTTS constants; regenerate it."
        )


# Running-statistics buffers of DiscretizationLoss that upstream never saved: the released dvae.pth has
# 53 tensors, the model 56, and these three are the difference (verified against the file's index).
DVAE_UNSAVED_BUFFERS = frozenset({
    "discrete_loss.accumulator",
    "discrete_loss.accumulator_filled",
    "discrete_loss.accumulator_index",
})


def load_state_strict(
    module: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    what: str,
    allowed_missing: frozenset = frozenset(),
) -> None:
    """Strict load that reports exactly which keys were missing or unexpected.

    `allowed_missing` names keys that may be absent (and keep their initialized values); nothing else may be.
    """
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(f"{what} strict load failed: missing={list(missing)}, unexpected={list(unexpected)}")


def load_dvae_state(dvae: torch.nn.Module, state: Dict[str, torch.Tensor], what: str) -> None:
    """Strict DVAE load that accepts upstream dvae.pth (missing only DVAE_UNSAVED_BUFFERS)."""
    load_state_strict(dvae, state, what, allowed_missing=DVAE_UNSAVED_BUFFERS)


def resolve_dvae_state(ckpt: Any) -> Dict[str, torch.Tensor]:
    """DVAE state dict from any supported layout: this project's {'dvae': ...}, {'model': ...},
    {'state_dict': ...}, or upstream's raw dvae.pth state dict."""
    if isinstance(ckpt, dict):
        for key in ("dvae", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def dvae_num_tokens(state: Dict[str, torch.Tensor]) -> int:
    """Codebook size of a DVAE state dict (codebook.embed is [codebook_dim, num_tokens])."""
    if "codebook.embed" in state:
        return int(state["codebook.embed"].shape[1])
    return SPEC.dvae.num_tokens


def load_dvae(path: str, device: str = "cpu", num_tokens: Optional[int] = None):
    """Build and strictly load a DVAE from a checkpoint file. Returns the model in eval mode."""
    from xtts_trainer.models.factory import build_dvae

    ckpt = safe_torch_load(path, map_location=device)
    check_spec_hash(ckpt, path)
    state = resolve_dvae_state(ckpt)
    dvae = build_dvae(num_tokens=num_tokens or dvae_num_tokens(state)).to(device)
    load_dvae_state(dvae, state, f"DVAE ({path})")
    dvae.eval()
    return dvae


# A DVAE trained by this project changes code meanings (F1: the codebook moves on every training forward
# pass), so its checkpoint carries this flag and a pretrained GPT may not be paired with it.
REQUIRES_GPT_RETRAIN_KEY = "requires_gpt_retrain"


def dvae_requires_gpt_retrain(path: str) -> bool:
    """True for a DVAE checkpoint written by Stage 1; False for the released dvae.pth (a bare state dict)."""
    ckpt = safe_torch_load(path, map_location="cpu")
    return bool(isinstance(ckpt, dict) and ckpt.get(REQUIRES_GPT_RETRAIN_KEY, False))

