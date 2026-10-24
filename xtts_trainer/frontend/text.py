"""The project tokenizer (docs/recipe_design.md F4): XTTS's VoiceBpeTokenizer plus explicitly registered languages.

Upstream preprocess_text raises NotImplementedError for any language outside its list
(tts/layers/xtts/tokenizer.py:651-652), and Xtts.inference indexes char_limits[language], so a model trained
on a new language cannot run through stock inference. This subclass is used everywhere text is tokenized:
validation, Stage 2 training and the exported inference path (stage2_gpt.export.load_for_inference).

Base languages keep upstream's cleaners unchanged. A registered language gets an explicit cleaner (default
basic_cleaner); an unregistered one still raises.
"""

import re
import unicodedata
from typing import Callable, Dict, Iterable, Mapping, Optional, Union

from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

from xtts_trainer.spec.languages import XTTS_BASE_LANGUAGES

Cleaner = Callable[[str], str]
# Upstream's fallback limit for languages missing from char_limits (VoiceBpeTokenizer.check_input_length)
DEFAULT_CHAR_LIMIT = 250


def basic_cleaner(text: str) -> str:
    """NFC, drop double quotes (as upstream's multilingual cleaner does), lowercase, collapse whitespace.

    Numbers and symbols are not expanded: spell them out in the data, or register a language-specific
    cleaner. The validator reports any character the vocab can't encode as [UNK].
    """
    text = unicodedata.normalize("NFC", text).replace('"', "")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _base_code(lang: str) -> str:
    return lang.split("-")[0]


class ProjectTokenizer(VoiceBpeTokenizer):
    def __init__(self, vocab_file: str,
                 extra_languages: Union[Mapping[str, Cleaner], Iterable[str], None] = None,
                 char_limits: Optional[Dict[str, int]] = None):
        super().__init__(vocab_file)
        if extra_languages is None:
            extra_languages = {}
        if not isinstance(extra_languages, Mapping):
            extra_languages = {lang: basic_cleaner for lang in extra_languages}
        self.cleaners: Dict[str, Cleaner] = {}
        base = {_base_code(l) for l in XTTS_BASE_LANGUAGES}
        for lang, cleaner in extra_languages.items():
            code = _base_code(lang.lower())
            if code in base:
                raise ValueError(f"'{lang}' is an XTTS base language; its upstream cleaner can't be replaced")
            if self.tokenizer.token_to_id(f"[{code}]") is None:
                raise ValueError(f"vocab {vocab_file} has no [{code}] token; extend the vocab first")
            self.cleaners[code] = cleaner
            self.char_limits.setdefault(code, DEFAULT_CHAR_LIMIT)
        self.char_limits.update(char_limits or {})

    @property
    def extra_languages(self):
        return sorted(self.cleaners)

    def preprocess_text(self, txt: str, lang: str) -> str:
        code = _base_code(lang)
        if code in self.cleaners:
            return self.cleaners[code](txt)
        return super().preprocess_text(txt, lang)
