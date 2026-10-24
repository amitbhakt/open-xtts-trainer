"""Cut an aligned recording into training segments at word gaps (docs/recipe_design.md §6, phase 6).

A segment's transcript is exactly the words aligned inside it, so text and audio agree by construction.
Cuts happen only where the gap between two words is at least min_gap_s and neither word is a wildcard
(its boundaries are unreliable). Each segment keeps up to edge_pad_s of the surrounding silence, never more
than half the gap, so it can't reach into a neighbouring word.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from xtts_trainer.prepare.align import AlignedWord


@dataclass
class SegmentConfig:
    # 3-11 s from the plan; the upper bound is also capped by the spec's max_wav_length at the call site.
    min_s: float = 3.0
    max_s: float = 11.0
    min_gap_s: float = 0.15  # [E4] smallsudarshan; ablation A6
    edge_pad_s: float = 0.2
    # Among plans that keep the same speech, prefer cuts at longer pauses (word boundaries there are the
    # least ambiguous for the aligner). Seconds of score per second of pause, counted up to the cap.
    pause_preference: float = 1.0
    pause_cap_s: float = 1.0


@dataclass
class QualityConfig:
    """Alignment checks that reject a segment. [E5] placeholders until calibrated on real data (Colab)."""

    min_word_score: float = 0.2
    # Speech the transcript doesn't account for (WindowAlignment.unexplained): text missing from the transcript
    max_unexplained_speech_s: float = 0.1
    # A word longer than this absorbed speech that isn't in the transcript
    max_word_s: float = 1.5
    max_s_per_char: float = 0.35


@dataclass
class Segment:
    start_s: float
    end_s: float
    words: List[AlignedWord]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class Dropped:
    start_s: float
    end_s: float
    text: str
    reason: str


@dataclass
class SegmentPlan:
    segments: List[Segment] = field(default_factory=list)
    dropped: List[Dropped] = field(default_factory=list)


def _bounds(words: Sequence[AlignedWord], i: int, j: int, audio_s: float, pad: float) -> Tuple[float, float]:
    """Audio span for words[i:j], padded into the gaps on either side by at most half the gap."""
    before = words[i].start_s - (words[i - 1].end_s if i > 0 else 0.0)
    after = (words[j].start_s if j < len(words) else audio_s) - words[j - 1].end_s
    start = words[i].start_s - min(pad, max(before, 0.0) / (2 if i > 0 else 1))
    end = words[j - 1].end_s + min(pad, max(after, 0.0) / (2 if j < len(words) else 1))
    return max(0.0, start), min(audio_s, end)


def cut_points(words: Sequence[AlignedWord], min_gap_s: float) -> List[int]:
    """Word indices k where a cut may go between words[k-1] and words[k] (0 and len(words) always allowed)."""
    cuts = [0]
    for k in range(1, len(words)):
        prev, nxt = words[k - 1], words[k]
        if nxt.start_s - prev.end_s >= min_gap_s and not prev.star and not nxt.star:
            cuts.append(k)
    cuts.append(len(words))
    return cuts


def plan_segments(words: Sequence[AlignedWord], audio_s: float, config: Optional[SegmentConfig] = None) -> SegmentPlan:
    """Choose cuts that keep the most speech in segments of [min_s, max_s], cutting at the longest pauses.

    Dynamic programming over allowed cut points: each step either keeps the stretch between two cuts as a
    segment (if its padded span fits the limits) or drops the stretch between two adjacent cuts. A kept
    segment scores its words' speech time plus pause_preference x its leading pause (capped). Scoring kept
    *audio* instead (the first version) penalized cutting at a long pause, since the pause itself is dropped:
    on JFK it cut at a 0.15 s "do | for" boundary rather than the 1.06 s pause after "not", and the aligner's
    uncertain "for" (score 0.12) then cost the whole second clip (Colab run 3).
    """
    config = config or SegmentConfig()
    if not words:
        return SegmentPlan()
    cuts = cut_points(words, config.min_gap_s)
    n = len(cuts)
    speech_s = [0.0]  # cumulative word time: speech_s[k] = speech in words[:k]
    for w in words:
        speech_s.append(speech_s[-1] + (w.end_s - w.start_s))

    def pause_before(k: int) -> float:
        return 0.0 if k == 0 else min(words[k].start_s - words[k - 1].end_s, config.pause_cap_s)

    best = [float("-inf")] * n
    choice: List[Optional[Tuple[int, bool]]] = [None] * n
    best[0] = 0.0
    for j in range(1, n):
        # Drop the stretch since the previous cut
        best[j], choice[j] = best[j - 1], (j - 1, False)
        for i in range(j - 1, -1, -1):
            if words[cuts[j] - 1].end_s - words[cuts[i]].start_s > config.max_s:
                break
            start, end = _bounds(words, cuts[i], cuts[j], audio_s, config.edge_pad_s)
            if not config.min_s <= end - start <= config.max_s:
                continue
            value = best[i] + speech_s[cuts[j]] - speech_s[cuts[i]] + config.pause_preference * pause_before(cuts[i])
            if value > best[j]:
                best[j], choice[j] = value, (i, True)

    plan = SegmentPlan()
    j = n - 1
    while j > 0:
        i, keep = choice[j]
        a, b = cuts[i], cuts[j]
        start, end = _bounds(words, a, b, audio_s, config.edge_pad_s)
        if keep:
            plan.segments.append(Segment(start, end, list(words[a:b])))
        else:
            reason = "no_pause_within_max_s" if end - start > config.max_s else "shorter_than_min_s"
            plan.dropped.append(Dropped(start, end, " ".join(w.text for w in words[a:b]), reason))
        j = i
    plan.segments.reverse()
    plan.dropped.reverse()
    return plan


def segment_problems(segment: Segment, unexplained_speech_s: float, config: Optional[QualityConfig] = None) -> List[str]:
    """Reasons the alignment inside a segment can't be trusted (empty list = usable)."""
    config = config or QualityConfig()
    problems = []
    worst = min(segment.words, key=lambda w: w.score)
    if worst.score < config.min_word_score:
        problems.append(f"low_alignment_score: '{worst.text}' {worst.score:.2f} < {config.min_word_score}")
    if unexplained_speech_s > config.max_unexplained_speech_s:
        problems.append(f"unexplained_speech: {unexplained_speech_s:.2f}s of speech not in the transcript")
    for w in segment.words:
        limit = max(config.max_word_s, config.max_s_per_char * len(w.text))
        if w.end_s - w.start_s > limit:
            problems.append(f"stretched_word: '{w.text}' spans {w.end_s - w.start_s:.2f}s")
            break
    return problems
