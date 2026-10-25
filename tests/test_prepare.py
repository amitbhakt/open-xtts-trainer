"""Phase 6: data preparation on forced alignment.

Most tests use a synthetic "tone speech" emitter: each character is a 60 ms sine at its own frequency, and
the emitter decodes 20 ms frames back to characters with an FFT. That exercises the real pipeline (torchaudio
forced_align, cut planning, clip writing, validation, re-alignment) on real audio files without the 1.26 GB
MMS_FA model. Tests marked mms run the real aligner when its weights are already cached (Colab).
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from dataset_helpers import write_test_vocab
from xtts_trainer.data import DatasetFormatError, load_dataset, validate_dataset, write_dataset
from xtts_trainer.data.validate import ValidationConfig, realignment_checker
from xtts_trainer.prepare import (
    AlignedWord, AlignmentError, CtcAligner, PauseConfig, PrepConfig, insert_pause_commas, pause_issues,
    plan_segments, prepare_recording, write_prepared,
)
from xtts_trainer.prepare.align import ALIGN_SAMPLE_RATE, Emitter, romanize_word, transcript_units
from xtts_trainer.prepare.segment import SegmentConfig, cut_points
from xtts_trainer.spec import SPEC

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = tuple(torchaudio.pipelines.MMS_FA.get_labels(star="*"))
CHAR_S, SPACE_S, FRAME_S = 0.06, 0.02, 0.02
TONE_HZ = 150.0


class ToneEmitter(Emitter):
    """Decodes tone speech: silent frame -> blank, otherwise the character whose tone is loudest."""

    name = "tone-test-emitter"
    labels = LABELS

    def __call__(self, audio_16k):
        hop = int(FRAME_S * ALIGN_SAMPLE_RATE)
        n = len(audio_16k) // hop
        frames = np.asarray(audio_16k[: n * hop]).reshape(n, hop)
        out = torch.full((n, len(LABELS)), float(np.log(0.1 / (len(LABELS) - 2))))
        out[:, -1] = 0.0  # wildcard column, appended as MMS_FA does
        spectrum = np.abs(np.fft.rfft(frames, axis=1))
        freqs = np.fft.rfftfreq(hop, 1 / ALIGN_SAMPLE_RATE)
        for t in range(n):
            if np.sqrt(np.mean(frames[t] ** 2)) < 0.01:
                cls = 0
            else:
                cls = int(np.clip(round(freqs[spectrum[t].argmax()] / TONE_HZ), 1, len(LABELS) - 2))
            out[t, cls] = float(np.log(0.9))
        return out


def tone_speech(words_and_gaps, sr=SPEC.output_sample_rate, lead_s=0.3, tail_s=0.3):
    """[(word, gap_after_s)] -> audio; letters only. Returns (audio, [(word, start_s, end_s)])."""
    pieces, timeline, t = [np.zeros(int(lead_s * sr))], [], lead_s
    for word, gap in words_and_gaps:
        start = t
        for k, ch in enumerate(word.lower().strip(",.;:!?")):
            n = int(round(CHAR_S * sr))
            tt = np.arange(n) / sr
            pieces.append(0.5 * np.sin(2 * np.pi * TONE_HZ * LABELS.index(ch) * tt))
            pieces.append(np.zeros(int(round(SPACE_S * sr))))
            t += CHAR_S + SPACE_S
        timeline.append((word, start, t))
        pieces.append(np.zeros(int(round(gap * sr))))
        t += gap
    pieces.append(np.zeros(int(tail_s * sr)))
    return np.concatenate(pieces).astype(np.float32), timeline


# Three phrases separated by 0.4 s pauses (with punctuation); words inside a phrase 0.04 s apart.
PHRASES = [
    "the quick brown fox jumps over the lazy dog,",
    "while seven bright stars shine above the quiet hills.",
    "every morning brings fresh hope to the small village people.",
]


def phrase_words(phrases=PHRASES, inner_gap=0.04, pause=0.4):
    out = []
    for p in phrases:
        ws = p.split()
        out += [(w, inner_gap) for w in ws[:-1]] + [(ws[-1], pause)]
    return out


@pytest.fixture
def tone_recording(tmp_path):
    audio, timeline = tone_speech(phrase_words())
    path = str(tmp_path / "recording.wav")
    sf.write(path, audio, SPEC.output_sample_rate)
    return path, timeline, " ".join(PHRASES)


def tone_aligner():
    return CtcAligner(emitter=ToneEmitter())


# --------------------------------------------------------------------------- F5: why not Whisper timestamps

def test_whisper_word_timestamps_have_no_gaps():
    """F5: HF Whisper word timestamps are contiguous, so they cannot locate pauses (needs forced alignment)."""
    with open(os.path.join(REPO_ROOT, "tests", "fixtures", "jfk_whisper_alignment.json"), encoding="utf-8") as f:
        chunks = json.load(f)["word_chunks"]
    gaps = [b["timestamp"][0] - a["timestamp"][1] for a, b in zip(chunks[:-1], chunks[1:])]
    assert max(abs(g) for g in gaps) == 0.0


# --------------------------------------------------------------------------- text -> aligner units

def test_romanize_word():
    assert romanize_word("Americans,", LABELS) == "americans"
    assert romanize_word("café", LABELS) == "cafe"
    assert romanize_word("don’t", LABELS) == "don't"
    assert romanize_word("—", LABELS) == ""  # punctuation only: not spoken
    assert romanize_word("1960", LABELS) is None  # spoken, nothing alignable: wildcard


def test_transcript_units_attach_punctuation_and_mark_wildcards():
    units = transcript_units("In 1960 — he said, yes.", LABELS)
    assert units == [("In", "in"), ("1960 —", None), ("he", "he"), ("said,", "said"), ("yes.", "yes")]


def test_non_latin_needs_uroman():
    try:
        import uroman  # noqa: F401
        pytest.skip("uroman installed")
    except ImportError:
        pass
    with pytest.raises(AlignmentError, match="uroman"):
        romanize_word("नमस्ते", LABELS)


# --------------------------------------------------------------------------- alignment

def test_alignment_recovers_word_times(tone_recording):
    path, timeline, text = tone_recording
    audio, sr = sf.read(path, dtype="float32")
    words = tone_aligner().align(audio, sr, text).words
    assert [w.text for w in words] == text.split()
    for w, (_, start, end) in zip(words, timeline):
        assert abs(w.start_s - start) <= 2 * FRAME_S
        assert abs(w.end_s - (end - SPACE_S)) <= 2 * FRAME_S
        assert w.score > 0.8


def test_alignment_rejects_text_longer_than_audio():
    audio, _ = tone_speech([("hi", 0.0)])
    with pytest.raises(AlignmentError, match="too long for audio"):
        tone_aligner().align(audio, SPEC.output_sample_rate, "hi " + "word " * 40)


def test_alignment_window_limit():
    with pytest.raises(AlignmentError, match="split it first"):
        CtcAligner(emitter=ToneEmitter(), max_window_s=1.0).align(np.zeros(48000, np.float32), 24000, "hi")


def test_missing_text_is_detected_as_unexplained_speech(tone_recording):
    """CTC absorbs speech missing from the transcript by stretching the last word over it; the forced path
    labels those frames blank while the model hears characters, so they count as unexplained speech."""
    from xtts_trainer.prepare.segment import Segment, segment_problems

    path, _, _ = tone_recording
    audio, sr = sf.read(path, dtype="float32")
    dur = len(audio) / sr
    full = tone_aligner().align(audio, sr, " ".join(PHRASES))
    partial = tone_aligner().align(audio, sr, " ".join(PHRASES[:2]))
    assert full.unexplained_speech_s(0, dur) == 0.0
    assert partial.words[-1].end_s - partial.words[-1].start_s > 3.0  # "hills." stretched over phrase 3
    assert partial.unexplained_speech_s(0, dur) > 1.0
    assert segment_problems(Segment(0, dur, full.words), full.unexplained_speech_s(0, dur)) == []
    problems = segment_problems(Segment(0, dur, partial.words), partial.unexplained_speech_s(0, dur))
    assert any(p.startswith("unexplained_speech") for p in problems)


# --------------------------------------------------------------------------- cut planning

def _w(text, start, end, star=False):
    return AlignedWord(text, start, end, 0.9, star)


def test_cuts_only_at_gaps_and_never_next_to_wildcards():
    words = [_w("a", 0, 1), _w("b", 1.05, 2), _w("c", 2.5, 3), _w("1960", 3.5, 4, star=True), _w("d", 4.5, 5)]
    assert cut_points(words, 0.15) == [0, 2, 5]


def test_plan_segments_respects_limits_and_text_is_exact_words():
    # 16 words of 0.5 s, pause of 0.2 s after every 4th: four ~2.3 s phrases that only fit in pairs.
    # The 0.2 s pauses are narrower than two edge pads, so padding must stop at half the gap.
    words, t = [], 0.0
    for i in range(16):
        words.append(_w(f"w{i}{',' if i % 4 == 3 else ''}", t, t + 0.5))
        t += 0.5 + (0.2 if i % 4 == 3 else 0.05)
    plan = plan_segments(words, t, SegmentConfig(min_s=3.0, max_s=6.0))
    assert len(plan.segments) == 2 and not plan.dropped
    for seg in plan.segments:
        assert 3.0 <= seg.duration_s <= 6.0
        assert seg.text == " ".join(w.text for w in seg.words)
        assert seg.words[-1].text.endswith(",") or seg.words[-1] is words[-1]
    assert [w for s in plan.segments for w in s.words] == words
    for a, b in zip(plan.segments[:-1], plan.segments[1:]):
        assert a.end_s <= b.start_s  # padding never overlaps the neighbour


def test_stretched_word_is_a_problem():
    from xtts_trainer.prepare.segment import Segment, segment_problems

    ok = Segment(0, 3, [_w("hello", 0.2, 0.6), _w("there.", 0.7, 1.2)])
    stretched = Segment(0, 3, [_w("hello", 0.2, 0.6), _w("there.", 0.7, 2.9)])
    assert segment_problems(ok, 0.0) == []
    assert [p.split(":")[0] for p in segment_problems(stretched, 0.0)] == ["stretched_word"]


def test_cuts_prefer_the_longest_pause_jfk():
    """JFK is just over 11 s, so it must be split. Pauses follow the real take; word lengths are approximate.
    v1 (kept audio) cut at the 0.16 s "do | for" boundary (Colab run 3). v2 (weighted pause bonus) cut after
    "And so," and dropped those words to collect the bonus (Colab run 4). Speech first, pauses as tie-break."""
    text = "And so, my fellow Americans: ask not what your country can do for you, ask what you can do for your country."
    pauses = {"so,": 0.3, "Americans:": 0.35, "not": 1.06, "do": 0.16, "you,": 0.45}
    words, t = [], 0.1
    for w in text.split():
        dur = 0.13 + 0.06 * len(w)
        words.append(_w(w, t, t + dur))
        t += dur + pauses.get(w, 0.05)
    plan = plan_segments(words, words[-1].end_s + 0.3, SegmentConfig(min_s=3.0, max_s=11.0))
    assert words[-1].end_s - words[0].start_s > 11.0  # the whole take doesn't fit in one clip
    assert not plan.dropped and [w for s in plan.segments for w in s.words] == words  # no speech traded away
    assert plan.segments[0].text.startswith("And so, my fellow Americans:")
    ends = [s.words[-1].text for s in plan.segments[:-1]]
    assert "not" in ends and "do" not in ends  # cut at the 1.06 s pause, never at the 0.16 s one


def _phrase(t, n_words, word_s=0.4, gap=0.05, label="w"):
    words = []
    for k in range(n_words):
        words.append(_w(f"{label}{k}", t, t + word_s))
        t += word_s + gap
    return words, t - gap


def test_pause_never_outweighs_speech():
    """A short head phrase followed by a long pause ("And so," in the real take) must stay: dropping it to
    cut at the pause would trade speech for pause, which v2 did."""
    head = [_w("Yes,", 0.1, 0.3)]
    body, end = _phrase(1.2, 8)  # 0.9 s pause after "Yes,", then 3.55 s of speech
    plan = plan_segments(head + body, end + 0.3, SegmentConfig(min_s=3.0, max_s=11.0))
    assert not plan.dropped and plan.segments[0].words[0].text == "Yes,"


@pytest.mark.parametrize("first_gap, second_gap", [(0.2, 0.8), (0.8, 0.2)])
def test_equal_speech_plans_cut_at_the_longer_pause(first_gap, second_gap):
    """Two plans keep all the speech; the cut must land on the longer pause, whichever side it is on."""
    a, t = _phrase(0.1, 8, label="a")             # 3.55 s
    b, t = _phrase(t + first_gap, 2, label="b")    # 0.85 s
    c, t = _phrase(t + second_gap, 8, label="c")   # 3.55 s
    # Either [a b][c] or [a][b c] fits 3-5.6 s with padding; [a b c] doesn't
    plan = plan_segments(a + b + c, t + 0.3, SegmentConfig(min_s=3.0, max_s=5.6))
    assert not plan.dropped and len(plan.segments) == 2
    cut_after = plan.segments[0].words[-1].text
    assert cut_after == ("b1" if second_gap > first_gap else "a7")


def test_one_weak_word_is_tolerated_but_not_two_or_a_missing_one():
    """JFK's reduced "for" scored 0.12 in a correct clip (Colab runs 3-4); the old any-word-under-0.2 rule
    dropped it. One weak word passes; two weak words, or one near zero (no audio behind it), don't."""
    from xtts_trainer.prepare.segment import Segment, segment_problems

    def seg(*scores):
        words = [AlignedWord(f"w{i}", i * 0.5, i * 0.5 + 0.4, sc) for i, sc in enumerate(scores)]
        return Segment(0.0, len(scores) * 0.5, words)

    assert segment_problems(seg(0.9, 0.12, 0.95, 0.9), 0.0) == []
    assert [p.split(":")[0] for p in segment_problems(seg(0.9, 0.12, 0.15, 0.9), 0.0)] == ["weak_alignment"]
    assert [p.split(":")[0] for p in segment_problems(seg(0.9, 0.02, 0.95, 0.9), 0.0)] == ["low_alignment_score"]


def test_plan_segments_drops_with_reason_when_no_pause():
    words = [_w(f"w{i}", i * 0.5, i * 0.5 + 0.48) for i in range(40)]  # 20 s, no gap >= 150 ms
    plan = plan_segments(words, 20.0, SegmentConfig(min_s=3.0, max_s=11.0))
    assert not plan.segments
    assert [d.reason for d in plan.dropped] == ["no_pause_within_max_s"]


# --------------------------------------------------------------------------- punctuation / pauses

def test_pause_issues_and_fix():
    words = [("hello,", 0.0, 0.5), ("there", 0.52, 1.0), ("friend", 1.5, 2.0), ("again.", 2.3, 2.8), ("ok", 3.0, 3.2)]
    issues = pause_issues(words, PauseConfig(min_gap_s=0.15, unpunctuated_gap_s=0.3))
    assert [(i["kind"], i["after"]) for i in issues] == [
        ("punctuation_without_pause", "hello,"), ("unpunctuated_pause", "there")]
    assert insert_pause_commas(words) == ["hello,", "there,", "friend", "again.", "ok"]
    assert insert_pause_commas([('"there"', 0, 1), ("x", 1.5, 2)]) == ['"there,"', "x"]


# --------------------------------------------------------------------------- end to end

def _prepare(tmp_path, recording, **kwargs):
    path, _, text = recording
    out = str(tmp_path / "prepared")
    config = PrepConfig(segment=SegmentConfig(min_s=1.5, max_s=6.0))
    records, report = prepare_recording(path, out, lang="en", speaker="tone", transcript=text,
                                        aligner=tone_aligner(), config=config, **kwargs)
    dataset = write_prepared(out, records, [report])
    vocab = write_test_vocab(str(tmp_path / "vocab.json"))
    return dataset, vocab, records, report


def _validate(dataset, vocab, **kwargs):
    return validate_dataset(dataset, vocab, eval_set_path=None, reference_lock_path=None, **kwargs)


def test_prepared_dataset_passes_validation_with_traceable_text(tmp_path, tone_recording):
    dataset, vocab, records, report = _prepare(tmp_path, tone_recording)
    assert [r.text for r in records] == PHRASES  # cut exactly at the three pauses
    assert report["segments"] == 3 and not report["dropped"]
    assert report["pause_issues_before_fix"] == {}
    for r in records:
        assert r.transcript_source == "human"
        assert r.alignment["aligner"] == "tone-test-emitter"
        assert " ".join(w["w"] for w in r.alignment["words"]) == r.text
    result = _validate(dataset, vocab, realigner=realignment_checker(tone_aligner()))
    assert result.ok, result.errors
    assert result.stats["aligned_clips"] == 3 and result.stats["pause_issues"] == {}
    assert json.loads(Path(os.path.dirname(dataset), "prep_report.json").read_text(encoding="utf-8"))["recordings"][0]["segments"] == 3


@pytest.mark.parametrize("shift_s", [1.0, -1.0])
def test_mutation_shifting_a_cut_by_one_second_fails_validation(tmp_path, tone_recording, shift_s):
    """Phase 6 exit criterion: move a clip's end by 1 s (into the next phrase, or into its own words)."""
    dataset, vocab, records, _ = _prepare(tmp_path, tone_recording)
    source, sr = sf.read(tone_recording[0], dtype="float32")
    r = records[0]
    start, end = r.alignment["start_s"], r.alignment["end_s"] + shift_s
    clip = source[int(start * sr):int(end * sr)]
    sf.write(os.path.join(os.path.dirname(dataset), r.audio), clip, sr, subtype="PCM_16")
    r.duration_s = round(len(clip) / sr, 4)
    r.alignment = {k: v for k, v in r.alignment.items() if k != "words"}  # a re-cut has no stored words
    write_dataset(dataset, records)
    result = _validate(dataset, vocab, realigner=realignment_checker(tone_aligner()))
    assert [(e.line, e.code) for e in result.errors] == [(2, "alignment_mismatch")]
    # without re-alignment the shift goes unnoticed: this is why --realign exists
    assert _validate(dataset, vocab).ok


def _single_clip(tmp_path, name, path, text, fix_pauses):
    """One clip from the whole recording (cuts only at >= 0.5 s gaps), marked single_clip_speaker."""
    out = str(tmp_path / name)
    config = PrepConfig(segment=SegmentConfig(min_s=1.0, max_s=12.0, min_gap_s=0.5), fix_pauses=fix_pauses)
    records, report = prepare_recording(path, out, "en", "tone", transcript=text, aligner=tone_aligner(), config=config)
    assert len(records) == 1, report
    records[0].single_clip_speaker = True
    return write_prepared(out, records, [report]), records[0], report


def test_unpunctuated_pause_is_reported_and_fixable(tmp_path):
    words = phrase_words(["one two three four five six seven eight nine ten eleven twelve."])
    words[2] = (words[2][0], 0.45)  # long pause after "three", no comma
    audio, _ = tone_speech(words)
    path = str(tmp_path / "rec.wav")
    sf.write(path, audio, SPEC.output_sample_rate)
    text = " ".join(w for w, _ in words)
    vocab = write_test_vocab(str(tmp_path / "vocab.json"))

    dataset, record, report = _single_clip(tmp_path, "plain", path, text, fix_pauses=False)
    assert report["pause_issues_before_fix"] == {"unpunctuated_pause": 1}
    result = _validate(dataset, vocab)
    assert result.ok and result.stats["pause_issues"] == {"unpunctuated_pause": 1}
    assert any("punctuation/pause" in w for w in result.warnings)
    strict = _validate(dataset, vocab, config=ValidationConfig(strict_pauses=True))
    assert {e.code for e in strict.errors} == {"unpunctuated_pause"}

    dataset, record, _ = _single_clip(tmp_path, "fixed", path, text, fix_pauses=True)
    assert "three," in record.text.split()
    strict = _validate(dataset, vocab, config=ValidationConfig(strict_pauses=True))
    assert strict.ok, strict.errors


def test_validator_rejects_alignment_that_does_not_spell_the_text(tmp_path, tone_recording):
    dataset, vocab, records, _ = _prepare(tmp_path, tone_recording)
    records[1].text = records[1].text.replace("seven", "eight")
    write_dataset(dataset, records)
    assert {e.code for e in _validate(dataset, vocab).errors} == {"alignment_text_mismatch"}


@pytest.mark.parametrize("alignment, message", [
    ({"start_s": 1.0}, "missing 'end_s'"),
    ({"start_s": 1.0, "end_s": 0.5}, "end_s must be after"),
    ({"start_s": 0, "end_s": 1, "extra": 1}, "unknown keys"),
    ({"start_s": 0, "end_s": 1, "words": [{"w": "a", "start_s": 0.5, "end_s": 0.2, "score": 1}]}, "not ordered"),
    ({"start_s": 0, "end_s": 1, "words": [{"w": "a", "start_s": 0}]}, "exactly the keys"),
])
def test_schema_rejects_malformed_alignment(tmp_path, alignment, message):
    path = tmp_path / "dataset.jsonl"
    record = {"audio": "a.wav", "text": "a", "lang": "en", "speaker": "s", "duration_s": 1.0, "sr": 24000,
              "transcript_source": "human", "alignment": alignment}
    path.write_text(json.dumps({"format": "open-xtts-dataset", "version": 1}) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(DatasetFormatError, match=message):
        load_dataset(str(path))


def test_phrase_transcriber_path_records_asr_source(tmp_path, tone_recording):
    """Without a human transcript, Whisper phrases only window the alignment; cuts still come from gaps."""
    path, timeline, _ = tone_recording
    # Phrase times deliberately off by 0.2 s: windows are padded, cuts don't use them
    phrases = [(p, max(0.0, timeline[i * 9][1] - 0.2) if i else 0.0, None) for i, p in enumerate(PHRASES)]
    phrases = [(p, s, None if i == len(PHRASES) - 1 else phrases[i + 1][1]) for i, (p, s, _) in enumerate(phrases)]
    out = str(tmp_path / "asr")
    records, report = prepare_recording(path, out, "en", "tone", asr_model="openai/test-whisper",
                                        phrase_transcriber=lambda a, sr, lang: phrases, aligner=tone_aligner(),
                                        config=PrepConfig(segment=SegmentConfig(min_s=1.5, max_s=6.0)))
    assert [r.text for r in records] == PHRASES
    assert {r.transcript_source for r in records} == {"asr:openai/test-whisper"}


# --------------------------------------------------------------------------- sources (speech_data)

def test_ljspeech_text_comes_from_normalized_column():
    from xtts_trainer.speech_data import ljspeech_texts

    csv_text = 'LJ001-0007|of about 1455,|of about fourteen fifty-five,\nLJ001-0002|in being "modern."|in being "modern."\n'
    assert ljspeech_texts(csv_text) == {"LJ001-0007": "of about fourteen fifty-five,", "LJ001-0002": 'in being "modern."'}
    with pytest.raises(ValueError, match="expected 3"):
        ljspeech_texts("LJ001-0001|only two\n")


def test_sources_declare_language_and_carry_no_hardcoded_cuts():
    from xtts_trainer.speech_data import SOURCE_SETS, SOURCES, Source

    fields = set(Source.__dataclass_fields__)
    assert not {"time_slice", "text"} & fields
    for name, source in SOURCES.items():
        assert source.lang and source.speaker, name
        assert source.license, f"{name} must say why we may train on it"
        downloads = set(source.urls) | ({source.metadata_url} if source.metadata_url else set())
        assert set(source.sha256) == downloads, f"{name}: every downloaded file pinned"
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in source.sha256.values()), f"{name}: pins are SHA-256 hex digests"
        assert all(re.search(r"/(dev|main|master)/", u) is None for u in downloads), f"{name}: no moving branch URLs"
        assert source.kind in ("corpus_clips", "recording")
        assert source.kind != "corpus_clips" or source.metadata_url
    assert all(n in SOURCES for names in SOURCE_SETS.values() for n in names)


def test_unpinned_download_is_refused_with_a_clear_error():
    from xtts_trainer.speech_data import Source

    source = Source(lang="en", speaker="x", description="unpinned", urls=["https://example.org/a.wav"], license="test")
    with pytest.raises(KeyError, match="no SHA-256 pin"):
        source.pinned(source.urls[0])


def test_download_checks_the_sha256_pin(tmp_path):
    import hashlib

    from xtts_trainer.speech_data import _download

    src = tmp_path / "file.bin"
    src.write_bytes(b"recording")
    good = hashlib.sha256(b"recording").hexdigest()
    assert Path(_download(src.as_uri(), str(tmp_path / "dl"), good)).read_bytes() == b"recording"
    assert Path(_download(src.as_uri(), str(tmp_path / "dl-uppercase"), good.upper())).read_bytes() == b"recording"
    with pytest.raises(RuntimeError, match="SHA-256"):
        _download(src.as_uri(), str(tmp_path / "dl"), "0" * 64)  # the cached copy is checked too
    # No pin, or a SHA-1-length pin, is refused before anything is downloaded
    for bad in (None, "", hashlib.sha1(b"recording").hexdigest()):
        with pytest.raises(ValueError, match="SHA-256 pin"):
            _download(src.as_uri(), str(tmp_path / "unpinned"), bad)
    assert not (tmp_path / "unpinned").exists()


def test_multi_file_source_keeps_every_files_clips(tmp_path, monkeypatch):
    """Clip numbering restarts per recording, so each file needs its own prefix (art_of_war has 7 files)."""
    import xtts_trainer.speech_data as speech_data

    import hashlib

    urls, pins = [], {}
    for k, phrases in enumerate([PHRASES, [p.upper() for p in PHRASES[::-1]]]):
        audio, _ = tone_speech(phrase_words(phrases))
        path = tmp_path / f"part{k}.wav"
        sf.write(str(path), audio, SPEC.output_sample_rate)
        urls.append(path.as_uri())
        pins[path.as_uri()] = hashlib.sha256(path.read_bytes()).hexdigest()  # hash the file, not its URI
    monkeypatch.setitem(speech_data.SOURCES, "two_parts", speech_data.Source(
        lang="en", speaker="tone", description="two files", urls=urls, license="test", transcript=None, sha256=pins))
    monkeypatch.setattr(speech_data, "prepare_recording", _with_transcripts(speech_data.prepare_recording, urls))
    dataset = speech_data.prepare_sources(["two_parts"], str(tmp_path / "out"), aligner=tone_aligner(),
                                          config=PrepConfig(segment=SegmentConfig(min_s=1.5, max_s=6.0)))
    records = load_dataset(dataset)
    assert len(records) == 6 and len({r.audio for r in records}) == 6
    report = json.loads(Path(tmp_path / "out" / "prep_report.json").read_text(encoding="utf-8"))["recordings"]
    assert [r["license"] for r in report] == ["test", "test"]


def _with_transcripts(prepare, urls):
    """prepare_recording with the human transcript of each tone file (the test source has none)."""
    texts = [" ".join(PHRASES), " ".join(p.upper() for p in PHRASES[::-1])]

    def wrapped(path, *args, **kwargs):
        kwargs["transcript"] = texts[[os.path.basename(u) for u in urls].index(os.path.basename(path))]
        return prepare(path, *args, **kwargs)
    return wrapped


@pytest.mark.parametrize("argv, message", [
    (["--lang", "hi"], "apply only to --input_file"),
    (["--input_file", "x.wav", "--lang", "en"], "needs --lang and --speaker_name"),
    (["--source", "jfk", "--input_file", "x.wav"], "either --source or --input_file"),
])
def test_cli_rejects_language_misuse(argv, message, capsys):
    from xtts_trainer.speech_data import main

    with pytest.raises(SystemExit):
        main(argv)
    assert message in capsys.readouterr().err


# --------------------------------------------------------------------------- real MMS_FA (Colab)

def _mms_cached():
    path = os.path.join(torch.hub.get_dir(), "checkpoints", "model.pt")
    return os.path.exists(path) and os.path.getsize(path) > 1_000_000_000




@pytest.mark.skipif(not _mms_cached() or not os.path.exists(os.path.join(REPO_ROOT, "xtts_base", "vocab.json")),
                    reason="needs MMS_FA weights in the torch hub cache and xtts_base/vocab.json (Colab step 7b)")
@pytest.mark.parametrize("shift_s", [0.0, 1.0, -1.0])
def test_mms_jfk_alignment_and_shifted_cut(tmp_path, shift_s):
    """Real aligner on JFK: prepared clips realign cleanly; a cut moved by 1 s is rejected."""
    from xtts_trainer.speech_data import JFK_SHA256, JFK_URL, SOURCES, _download

    jfk = _download(JFK_URL, str(tmp_path / "dl"), JFK_SHA256)
    out = str(tmp_path / "jfk")
    aligner = CtcAligner()
    records, report = prepare_recording(jfk, out, "en", "jfk", transcript=SOURCES["jfk"].transcript, aligner=aligner,
                                        config=PrepConfig(segment=SegmentConfig(min_s=2.5, max_s=11.0)))
    assert len(records) >= 2, report
    assert records[0].text.startswith("And so, my fellow Americans:")
    dataset = write_prepared(out, records, [report])
    if shift_s:
        source, sr = sf.read(jfk, dtype="float32")
        r = records[0]
        src = torchaudio.functional.resample(torch.from_numpy(source), sr, SPEC.output_sample_rate).numpy()
        a, b = r.alignment["start_s"], r.alignment["end_s"] + shift_s
        clip = src[int(a * SPEC.output_sample_rate):int(b * SPEC.output_sample_rate)]
        sf.write(os.path.join(out, r.audio), clip, SPEC.output_sample_rate, subtype="PCM_16")
        r.duration_s = round(len(clip) / SPEC.output_sample_rate, 4)
        r.alignment = {k: v for k, v in r.alignment.items() if k != "words"}
        write_dataset(dataset, records)
    real_vocab = os.path.join(REPO_ROOT, "xtts_base", "vocab.json")
    result = _validate(dataset, real_vocab, realigner=realignment_checker(aligner))
    # Calibration evidence: every clip re-aligned as validation sees it, word by word (scores, unexplained speech)
    from xtts_trainer.prepare.pipeline import load_mono

    realigned = []
    for r in records:
        audio = load_mono(os.path.join(out, r.audio), SPEC.output_sample_rate)
        window = aligner.align(audio, SPEC.output_sample_rate, r.text)
        clip_s = len(audio) / SPEC.output_sample_rate
        realigned.append({"audio": r.audio, "text": r.text, "duration_s": round(clip_s, 3),
                          "unexplained_speech_s": round(window.unexplained_speech_s(0.0, clip_s), 3),
                          "words": [[w.text, round(w.start_s, 3), round(w.end_s, 3), round(w.score, 3)] for w in window.words]})
    evidence_dir = os.path.join(REPO_ROOT, "docs", "evidence", "phase6")
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, f"jfk_realign_shift_{shift_s:+.0f}s.json"), "w", encoding="utf-8") as f:
        json.dump({"shift_s": shift_s, "report": report, "realigned": realigned, "validation": result.to_json()}, f, indent=2)
    if shift_s:
        assert "alignment_mismatch" in {e.code for e in result.errors}
    else:
        assert result.ok, result.errors
