"""Punctuation/pause consistency (docs/recipe_design.md §3, [E4]; ablation A6 decides whether it blocks).

XTTS learns pauses from punctuation, so a comma the speaker didn't pause at, or a long pause with no
punctuation, teaches the wrong mapping. Gaps come from forced alignment, never from Whisper timestamps.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

PAUSE_PUNCTUATION = set(",;:.!?…—–")
CLOSERS = "\"')]}”’»"

# (word text, start_s, end_s)
TimedWord = Tuple[str, float, float]


@dataclass
class PauseConfig:
    min_gap_s: float = 0.15  # a punctuation mark should come with at least this much silence
    unpunctuated_gap_s: float = 0.3  # a silence this long should carry punctuation


def ends_with_pause_punctuation(word: str) -> bool:
    stripped = word.rstrip(CLOSERS)
    return bool(stripped) and stripped[-1] in PAUSE_PUNCTUATION


def pause_issues(words: Sequence[TimedWord], config: PauseConfig = PauseConfig()) -> List[dict]:
    """Every internal word boundary where punctuation and pause disagree."""
    issues = []
    for (text, _, end), (_, start, _) in zip(words[:-1], words[1:]):
        gap = start - end
        if ends_with_pause_punctuation(text):
            if gap < config.min_gap_s:
                issues.append({"kind": "punctuation_without_pause", "after": text, "gap_s": round(gap, 3)})
        elif gap >= config.unpunctuated_gap_s:
            issues.append({"kind": "unpunctuated_pause", "after": text, "gap_s": round(gap, 3)})
    return issues


def insert_pause_commas(words: Sequence[TimedWord], config: PauseConfig = PauseConfig()) -> List[str]:
    """Word texts with a comma added after each long unpunctuated pause (the A6 'corrected' arm)."""
    out = [w[0] for w in words]
    for i, ((text, _, end), (_, start, _)) in enumerate(zip(words[:-1], words[1:])):
        if start - end >= config.unpunctuated_gap_s and not ends_with_pause_punctuation(text):
            stripped = text.rstrip(CLOSERS)
            out[i] = stripped + "," + text[len(stripped):]
    return out
