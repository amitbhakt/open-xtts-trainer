"""The one language registry: XTTS base languages, known extensions, and Whisper ASR mappings."""

from xtts_trainer.spec.xtts_spec import SPEC

# Languages the stock XTTS-v2 checkpoint supports (from the vendored upstream config). Other languages are
# registered explicitly (vocab extension + extra_languages), never inferred from data.
XTTS_BASE_LANGUAGES = frozenset(SPEC.languages)

# Whisper ASR language codes that differ from ours, and languages Whisper cannot transcribe
WHISPER_LANG_MAP = {
    "zh-cn": "zh",
    "pt-br": "pt",
    "es-la": "es",
}
WHISPER_UNSUPPORTED = {"dv", "eo"}


def whisper_language(lang: str) -> str:
    """Map a project language code to Whisper's code, raising for languages Whisper cannot transcribe."""
    code = lang.strip().lower()
    if code in WHISPER_UNSUPPORTED:
        raise ValueError(
            f"Whisper ASR does not support language '{lang}'. "
            f"Please provide an explicit --transcript or --transcript_file."
        )
    return WHISPER_LANG_MAP.get(code, code)
