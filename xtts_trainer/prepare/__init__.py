"""Data preparation on forced alignment: transcribe, align, then cut at word gaps (docs/recipe_design.md §6, phase 6)."""

from xtts_trainer.prepare.align import AlignedWord, AlignmentError, CtcAligner, MmsFaEmitter
from xtts_trainer.prepare.pauses import PauseConfig, insert_pause_commas, pause_issues
from xtts_trainer.prepare.pipeline import PrepConfig, prepare_recording, write_prepared
from xtts_trainer.prepare.segment import QualityConfig, SegmentConfig, plan_segments, segment_problems

__all__ = [
    "AlignedWord", "AlignmentError", "CtcAligner", "MmsFaEmitter",
    "PauseConfig", "insert_pause_commas", "pause_issues",
    "PrepConfig", "prepare_recording", "write_prepared",
    "QualityConfig", "SegmentConfig", "plan_segments", "segment_problems",
]
