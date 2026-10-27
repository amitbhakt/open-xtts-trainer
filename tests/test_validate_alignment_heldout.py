"""scripts/validate_alignment_heldout.py: mutation labels, provenance checks, and the re-alignment check end to end."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest

from scripts import validate_alignment_heldout as heldout
from test_prepare import phrase_words, tone_aligner, tone_speech

RATE = heldout.RATE


def _clip(seconds=3.0, first=0.2, last=2.8):
    words = [{"w": "a", "start_s": first, "end_s": 1.4, "score": 0.9},
             {"w": "b", "start_s": 1.5, "end_s": last, "score": 0.9}]
    return np.ones(int(seconds * RATE), dtype=np.float32), words


def test_mutation_labels_follow_speech_not_clip_edges():
    """Clips keep edge_pad_s of silence; cuts are measured from the words, and trimming silence must pass."""
    clip, words = _clip()
    muts = {m["name"]: m for m in heldout.build_mutations(clip, words, "a b", "en", next_speech=np.ones(RATE))}
    assert muts["control_trim_tail_silence"]["expected_pass"] is True
    assert len(muts["control_trim_tail_silence"]["audio"]) == int(round(2.85 * RATE))  # last word end + 50 ms
    assert muts["control_trim_head_silence"]["expected_pass"] is True
    assert len(muts["control_trim_head_silence"]["audio"]) == len(clip) - int(round(0.15 * RATE))
    for cut in heldout.TAIL_SPEECH_CUTS_S:
        m = muts[f"tail_speech_cut_{int(cut * 1000)}ms"]
        assert m["expected_pass"] is False and m["speech_removed_s"] == cut
        assert len(m["audio"]) == int(round((2.8 - cut) * RATE))  # cut into speech by exactly `cut`
    for cut in heldout.HEAD_SPEECH_CUTS_S:
        m = muts[f"head_speech_cut_{int(cut * 1000)}ms"]
        assert len(m["audio"]) == len(clip) - int(round((0.2 + cut) * RATE))
    assert muts["insert_head_word"]["text"] == "Moreover a b" and muts["insert_tail_word"]["text"] == "a b extraordinary"
    assert muts["spillover_0.05s"]["expected_pass"] is True and muts["spillover_0.50s"]["expected_pass"] is False
    assert len(muts["spillover_0.50s"]["audio"]) == int(round(2.85 * RATE)) + int(0.5 * RATE)


def test_cuts_that_would_leave_too_little_speech_are_skipped():
    clip, words = _clip(seconds=1.2, first=0.2, last=0.95)  # 0.75 s of speech
    names = {m["name"] for m in heldout.build_mutations(clip, words, "a b", "fr", next_speech=None)}
    assert "tail_speech_cut_150ms" in names and "tail_speech_cut_300ms" not in names
    assert not any(n.startswith("spillover") for n in names)  # no next utterance, no spillover test
    assert "insert_mid_word" in names


def test_pins_are_verified_or_recorded_never_ignored():
    assert heldout.check_pin("u", "abc", {}) == "recorded"
    assert heldout.check_pin("u", "abc", {"u": "abc"}) == "verified"
    with pytest.raises(RuntimeError, match="pin"):
        heldout.check_pin("u", "abc", {"u": "def"})


def test_environment_check_reads_the_lock():
    locked = heldout.locked_versions()
    assert locked["torch"] and locked["torchaudio"]
    assert heldout.environment_problems({"torch": locked["torch"] + "+cu130", "torchaudio": locked["torchaudio"]},
                                        locked) == []
    assert heldout.environment_problems({"torch": "2.1.2", "torchaudio": locked["torchaudio"]}, locked)


def test_cer_normalizes_case_and_punctuation():
    assert heldout.cer("Ceci est un test.", "ceci est un test") == 0.0
    assert 0 < heldout.cer("abcd", "abce") <= 0.25


def test_realign_check_end_to_end_with_a_synthetic_aligner():
    """The plumbing, not detection power (that is what the held-out run measures): controls pass, large cuts,
    edge insertions and long spillover are rejected, short spillover passes, and scores are logged."""
    words_and_gaps = phrase_words(["the quick brown fox jumps over the lazy dog."])
    audio, timeline = tone_speech(words_and_gaps, lead_s=0.2, tail_s=0.2)
    words = [{"w": w, "start_s": s, "end_s": e, "score": 1.0} for w, s, e in timeline]
    text = " ".join(w for w, _ in words_and_gaps)
    aligner = tone_aligner()
    assert heldout.realign(aligner, audio, text, heldout.FROZEN_QUALITY)["problems"] == []
    next_speech = audio[int(0.2 * RATE):]
    outcome = {}
    for m in heldout.build_mutations(audio, words, text, "en", next_speech=next_speech):
        check = heldout.realign(aligner, m["audio"], m["text"], heldout.FROZEN_QUALITY)
        assert check["scores"] and check["unexplained_speech_s"] is not None, m["name"]
        outcome[m["name"]] = not check["problems"]
    assert outcome["control_trim_tail_silence"] and outcome["control_trim_head_silence"]
    for name in ("tail_speech_cut_300ms", "tail_speech_cut_600ms", "head_speech_cut_300ms", "head_speech_cut_600ms",
                 "insert_head_word", "insert_tail_word", "spillover_0.50s"):
        assert outcome[name] is False, name
    assert outcome["spillover_0.05s"] is True


def test_summary_counts_clean_rejections_and_mutation_rates():
    results = [{"lang": "en", "clean": [{"accepted": True}, {"accepted": False}],
                "mutations": [{"name": "insert_head_word", "expected_pass": False, "actual_pass": False}]},
               {"lang": "fr", "clean": [{"accepted": True}], "mutations": []}]
    summary = heldout.summarize(results)
    assert summary["clean_accepted"] == "2/3"
    assert {k: v["accepted"] for k, v in summary["clean_accepted_by_language"].items()} == {"en": "1/2", "fr": "1/1"}
    assert summary["mutations"]["insert_head_word"]["passed"] == "0/1"
    assert summary["mutations"]["insert_head_word"]["rejected_ci95"] == heldout.clopper_pearson(1, 1)
    json.dumps(summary)


def test_clopper_pearson_matches_known_values():
    assert heldout.clopper_pearson(4, 4) == [0.3976, 1.0]  # 4/4 only bounds acceptance above ~40%
    low, high = heldout.clopper_pearson(40, 40)
    assert high == 1.0 and 0.91 < low < 0.92
    assert heldout.clopper_pearson(0, 0) == [None, None]


def test_full_run_end_to_end_on_a_synthetic_source(tmp_path, monkeypatch):
    """run() from download to report, with the tone aligner standing in for MMS_FA (no network, no GPU)."""
    import soundfile as sf

    import xtts_trainer.prepare as prepare_module
    from dataset_helpers import write_test_vocab

    audio, _ = tone_speech(phrase_words(), lead_s=0.3, tail_s=0.3)
    wav = tmp_path / "synthetic.wav"
    sf.write(str(wav), audio, RATE)
    text = ("the quick brown fox jumps over the lazy dog, while seven bright stars shine above the quiet hills. "
            "every morning brings fresh hope to the small village people.")
    source = {"id": "synthetic", "lang": "en", "kind": "recording", "description": "tone speech",
              "url": wav.as_uri(), "license": "test", "max_seconds": None, "transcript": text,
              "transcript_verified": True}
    monkeypatch.setattr(heldout, "HELDOUT_SOURCES", [source])
    monkeypatch.setattr(heldout, "DEFAULT_VOCAB", tmp_path / "vocab.json")
    write_test_vocab(str(tmp_path / "vocab.json"))
    monkeypatch.setattr(heldout, "PINS_PATH", tmp_path / "pins.json")
    monkeypatch.setattr(prepare_module, "CtcAligner", lambda device=None: tone_aligner())
    out = tmp_path / "report.json"
    report = heldout.run("cpu", out, str(tmp_path / "work"), allow_unverified=False, allow_unlocked_env=True)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"] and saved["frozen_quality_config"]["max_weak_words"] == 3
    result = saved["results"][0]
    assert result["provenance"]["pin"] == "recorded"
    assert json.loads((tmp_path / "pins.json").read_text())[wav.as_uri()] == result["provenance"]["sha256"]
    kept = [c for c in result["clean"] if c["prep_kept"]]
    assert kept and all(c["accepted"] and c["validator_ok"] for c in kept), result["clean"]
    names = {m["name"] for m in result["mutations"]}
    assert {"control_trim_tail_silence", "insert_head_word", "spillover_0.50s"} <= names
    assert all(m["scores"] for m in result["mutations"] if not m["problems"])
    # A second run verifies the recorded pin instead of re-recording it
    again = heldout.run("cpu", out, str(tmp_path / "work"), allow_unverified=False, allow_unlocked_env=True)
    assert again["results"][0]["provenance"]["pin"] == "verified"


def test_run_refuses_unverified_transcripts_and_unlocked_environments(tmp_path, monkeypatch):
    source = dict(heldout.HELDOUT_SOURCES[2], transcript_verified=False)
    monkeypatch.setattr(heldout, "HELDOUT_SOURCES", [source])
    with pytest.raises(SystemExit, match="not verified"):
        heldout.run("cpu", tmp_path / "r.json", str(tmp_path), allow_unverified=False, allow_unlocked_env=True)
    monkeypatch.setattr(heldout, "locked_versions", lambda: {"torch": "0.0.0"})
    with pytest.raises(SystemExit, match="locked environment"):
        heldout.run("cpu", tmp_path / "r.json", str(tmp_path), allow_unverified=True, allow_unlocked_env=False)


def test_arctic_source_end_to_end_with_its_own_license(tmp_path, monkeypatch):
    """The ARCTIC branch: utterances and transcripts come from the tarball, the speaker's COPYING is downloaded and
    pinned, and the same ids are used for every utterance's clips."""
    import io
    import tarfile

    import soundfile as sf

    import eval.dataset as dataset_module
    import xtts_trainer.prepare as prepare_module
    from dataset_helpers import write_test_vocab

    phrases = ["the quick brown fox jumps over the lazy dog.", "while seven bright stars shine above the hills.",
               "every morning brings fresh hope to the village."]
    utts = ["arctic_a0003", "arctic_a0004", "arctic_a0005"]
    tar_path = tmp_path / "cmu_us_zzz_arctic.tar.bz2"
    with tarfile.open(tar_path, "w:bz2") as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for utt, phrase in zip(utts, phrases):
            audio, _ = tone_speech(phrase_words([phrase]), lead_s=0.3, tail_s=0.3)
            buf = io.BytesIO()
            sf.write(buf, audio, RATE, format="WAV")
            add(f"cmu_us_zzz_arctic/wav/{utt}.wav", buf.getvalue())
        add("cmu_us_zzz_arctic/etc/txt.done.data",
            "".join(f'( {u} "{p}" )\n' for u, p in zip(utts, phrases)).encode("utf-8"))
    (tmp_path / "COPYING").write_text("test license\n", encoding="utf-8")
    source = {"id": "arctic_zzz", "lang": "en", "kind": "arctic", "description": "fake ARCTIC speaker",
              "url": tar_path.as_uri(), "corpus_speaker": "zzz", "license": "test"}
    monkeypatch.setattr(heldout, "HELDOUT_SOURCES", [source])
    monkeypatch.setattr(heldout, "ARCTIC_UTTERANCES", utts)
    monkeypatch.setattr(heldout, "EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(heldout, "PINS_PATH", tmp_path / "pins.json")
    monkeypatch.setattr(heldout, "DEFAULT_VOCAB", tmp_path / "vocab.json")
    write_test_vocab(str(tmp_path / "vocab.json"))
    monkeypatch.setattr(dataset_module, "CMU_ARCTIC_LICENSE_URL", (tmp_path / "COPYING").as_uri())
    monkeypatch.setattr(prepare_module, "CtcAligner", lambda device=None: tone_aligner())

    report = heldout.run("cpu", tmp_path / "report.json", str(tmp_path / "work"), False, True)
    result = report["results"][0]
    assert result["provenance"]["transcripts"] == "corpus prompt file etc/txt.done.data"
    assert (tmp_path / "evidence" / "licenses" / "zzz_COPYING").read_text() == "test license\n"
    pins = json.loads((tmp_path / "pins.json").read_text())
    assert pins[(tmp_path / "COPYING").as_uri()] == result["provenance"]["license"]["sha256"]
    assert pins[tar_path.as_uri()] == result["provenance"]["sha256"]
    assert {c["recording"] for c in result["clean"]} == set(utts)
    assert all(c["accepted"] for c in result["clean"]), result["clean"]
    # Each utterance's next one supplies the spillover speech
    assert any(m["name"] == "spillover_0.50s" for m in result["mutations"])
