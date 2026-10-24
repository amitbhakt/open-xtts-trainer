"""Model construction and checkpoint I/O shared by every stage."""

from xtts_trainer.models.factory import build_dvae, build_gpt, build_hifigan, hifigan_decoder_input
from xtts_trainer.models.checkpoint_io import (
    SPEC_HASH_KEY,
    safe_torch_load,
    stamp,
    check_spec_hash,
    load_state_strict,
    load_dvae_state,
    DVAE_UNSAVED_BUFFERS,
    resolve_dvae_state,
    dvae_num_tokens,
    load_dvae,
    REQUIRES_GPT_RETRAIN_KEY,
    dvae_requires_gpt_retrain,
)

__all__ = [
    "build_dvae", "build_gpt", "build_hifigan", "hifigan_decoder_input",
    "SPEC_HASH_KEY", "safe_torch_load", "stamp", "check_spec_hash", "load_state_strict", "load_dvae_state", "DVAE_UNSAVED_BUFFERS",
    "resolve_dvae_state", "dvae_num_tokens", "load_dvae", "REQUIRES_GPT_RETRAIN_KEY", "dvae_requires_gpt_retrain",
]
