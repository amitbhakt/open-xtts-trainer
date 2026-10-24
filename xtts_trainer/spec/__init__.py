"""Single source of truth for XTTS-v2 constants, vendored artifacts and the language registry."""

from xtts_trainer.spec.xtts_spec import SPEC, XttsSpec, MelSpec, DVAEArch, GPTArch, load_spec, spec_hash
from xtts_trainer.spec.artifacts import DEFAULT_MEL_STATS_PATH, ensure_mel_stats, verify_vendored_config
from xtts_trainer.spec.languages import (
    XTTS_BASE_LANGUAGES,
    WHISPER_LANG_MAP,
    WHISPER_UNSUPPORTED,
    whisper_language,
)

__all__ = [
    "SPEC", "XttsSpec", "MelSpec", "DVAEArch", "GPTArch", "load_spec", "spec_hash",
    "DEFAULT_MEL_STATS_PATH", "ensure_mel_stats", "verify_vendored_config",
    "XTTS_BASE_LANGUAGES", "WHISPER_LANG_MAP", "WHISPER_UNSUPPORTED", "whisper_language",
]
