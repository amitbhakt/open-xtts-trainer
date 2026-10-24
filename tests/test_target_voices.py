"""Target-speaker evaluation: held-out reference clips, the leakage guard, and eval with --voices target."""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest
import soundfile as sf
import torch

from dataset_helpers import write_test_vocab
from eval.reference_voices import MAX_CLIP_S, MIN_CLIP_S
from eval.target_voices import LOCK_FILENAME, build_target_voices as _build, holdout_hashes, verify_target_voices
from xtts_trainer.data import DatasetRecord, load_dataset, validate_dataset, write_dataset
from xtts_trainer.spec import SPEC

SR = SPEC.output_sample_rate


def build_target_voices(dataset, out_dir, **kw):
    """Tests build in tmp dirs; the real CLI may only use the canonical location (tested below)."""
    return _build(dataset, out_dir, allow_unchecked_location=True, **kw)


def _dataset(root, speakers=("me", "you"), n=8, seconds=2.0):
    records = []
    for s, speaker in enumerate(speakers):
        for i in range(n):
            rng = np.random.RandomState(100 * s + i)
            t = np.arange(int(SR * seconds)) / SR
            audio = (0.2 * np.sin(2 * np.pi * rng.uniform(100, 800) * t) + 0.01 * rng.randn(len(t))).astype("float32")
            name = f"{speaker}_{i}.wav"
            sf.write(os.path.join(root, name), audio, SR)
            records.append(DatasetRecord(audio=name, text=f"sentence {i} by {speaker}", lang="en", speaker=speaker,
                                         duration_s=seconds, sr=SR, transcript_source="human"))
    return write_dataset(os.path.join(root, "dataset.jsonl"), records)


def test_build_holds_out_whole_clips_and_writes_the_train_split(tmp_path):
    dataset = _dataset(str(tmp_path))
    out = str(tmp_path / "targets")
    lock = build_target_voices(dataset, out, seed=3)
    assert sorted(lock["voices"]) == ["target_me", "target_you"]
    train = load_dataset(lock["train_dataset"])
    for vid, entry in lock["voices"].items():
        assert MIN_CLIP_S <= entry["duration_s"] <= MAX_CLIP_S
        held = {c["audio"] for c in entry["source_clips"]}
        assert held and not held & {r.audio for r in train}  # held-out clips never reach training
    assert len(train) == 16 - sum(len(e["source_clips"]) for e in lock["voices"].values())
    assert verify_target_voices(out).keys() == lock["voices"].keys()
    again = build_target_voices(dataset, str(tmp_path / "again"), seed=3)
    assert [e["source_clips"] for e in again["voices"].values()] == [e["source_clips"] for e in lock["voices"].values()]


def test_training_on_held_out_clips_is_refused(tmp_path):
    dataset = _dataset(str(tmp_path))
    out = str(tmp_path / "targets")
    lock = build_target_voices(dataset, out)
    vocab = write_test_vocab(str(tmp_path / "vocab.json"))
    target_lock = os.path.join(out, LOCK_FILENAME)
    kw = dict(eval_set_path=None, reference_lock_path=None, target_lock_path=target_lock)
    leaked = validate_dataset(dataset, vocab, **kw)
    n_held = sum(len(e["source_clips"]) for e in lock["voices"].values())
    assert [e.code for e in leaked.errors] == ["target_holdout_overlap"] * n_held
    assert validate_dataset(lock["train_dataset"], vocab, **kw).ok
    assert len(holdout_hashes(target_lock)) == n_held + 2  # each source clip plus each assembled clip


def test_tampered_target_clip_is_refused(tmp_path):
    out = str(tmp_path / "targets")
    build_target_voices(_dataset(str(tmp_path)), out)
    sf.write(os.path.join(out, "target_me.wav"), np.zeros(SR * 7, dtype="float32"), SR)
    with pytest.raises(RuntimeError, match="doesn't match"):
        verify_target_voices(out)


def test_too_little_audio_or_too_few_training_clips(tmp_path):
    short = tmp_path / "short"
    short.mkdir()
    with pytest.raises(ValueError, match="not enough audio"):
        build_target_voices(_dataset(str(short), speakers=("me",), n=2, seconds=1.0), str(tmp_path / "t1"))
    few = tmp_path / "few"
    few.mkdir()
    with pytest.raises(ValueError, match="training clips"):
        build_target_voices(_dataset(str(few), speakers=("me",), n=4, seconds=2.0), str(tmp_path / "t2"))


def test_eval_with_target_voices(monkeypatch, tmp_path):
    """run_eval_baseline --voices target evaluates the held-out clips under their target_ ids."""
    import TTS.tts.models.xtts as xtts_module
    import eval.harness as harness_module
    import scripts.run_eval_baseline as baseline_module
    from eval.metrics import SpeakerSimilarity, UTMOSPredictor
    from test_eval_phase1 import FakeEncoder, FakeMOS
    from xtts_trainer.spec.xtts_spec import VENDORED_CONFIG_PATH

    out = str(tmp_path / "targets")
    build_target_voices(_dataset(str(tmp_path)), out)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    shutil.copy(VENDORED_CONFIG_PATH, ckpt / "config.json")
    (ckpt / "model.pth").write_bytes(b"fake")
    (ckpt / "vocab.json").write_bytes(b"{}")
    wav = (np.sin(np.linspace(0, 3000, SR * 2)) * 0.3).astype(np.float32)
    conditioned = []

    class FakeXtts:
        @staticmethod
        def init_from_config(config):
            return FakeXtts()

        def load_checkpoint(self, config, checkpoint_path, vocab_path, use_deepspeed):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def get_conditioning_latents(self, audio_path):
            conditioned.append(audio_path[0])
            return torch.zeros(1, 32, 1024), torch.zeros(1, 512, 1)

        def inference(self, **kw):
            return {"wav": wav}

    monkeypatch.setattr(xtts_module, "Xtts", FakeXtts)
    monkeypatch.setattr(harness_module, "SpeakerSimilarity", lambda device: SpeakerSimilarity(model=FakeEncoder()))
    monkeypatch.setattr(harness_module, "UTMOSPredictor", lambda device: UTMOSPredictor(model=FakeMOS()))
    monkeypatch.setattr(harness_module, "asr_transcribe", lambda audio, sr, lang, model_name, device: "text")
    manifest = baseline_module.generate_baseline(
        checkpoint_dir=str(ckpt), output_path=str(tmp_path / "m.json"), seeds=[1], languages=["en"], subset_size=2,
        voices="target", target_dir=out, label="base",
    )
    assert manifest["voices"] == "target"
    assert [v["speaker_id"] for v in manifest["reference_voices"]] == ["target_me", "target_you"]
    # each voice id was synthesized from its own held-out reference clip, in order
    assert conditioned == [os.path.join(out, "target_me.wav"), os.path.join(out, "target_you.wav")]
    assert [run["speaker_id"] for run in manifest["runs"]] == ["target_me", "target_you"]
    with pytest.raises(ValueError, match="voices must be"):
        baseline_module.generate_baseline(checkpoint_dir=str(ckpt), voices="everyone")
    # Without --output, each label and voice set gets its own file: a target run can't overwrite another run
    monkeypatch.chdir(tmp_path)
    for label in ("base", "tuned"):
        baseline_module.generate_baseline(checkpoint_dir=str(ckpt), seeds=[1], languages=["en"], subset_size=2,
                                          voices="target", target_dir=out, label=label)
    written = sorted(os.listdir(tmp_path / "eval" / "baselines"))
    commit = baseline_module.PINNED_UPSTREAM_COMMIT[:12]
    assert written == [f"base_xtts_{commit}_target.json", f"tuned_xtts_{commit}_target.json"]
    # ...and, run from another directory, still measured against the frozen eval set (absolute default path)
    from eval.dataset import DEFAULT_EVAL_SET_PATH

    frozen = json.load(open(DEFAULT_EVAL_SET_PATH))["sha256"]
    on_disk = json.load(open(tmp_path / "eval" / "baselines" / f"base_xtts_{commit}_target.json"))
    assert on_disk["eval_set_sha256"] == frozen and not (tmp_path / "eval" / "dataset").exists()
    for bad in ("../escape", "a/b", ".hidden", ""):
        with pytest.raises(ValueError, match="plain name"):
            baseline_module.generate_baseline(checkpoint_dir=str(ckpt), voices="target", target_dir=out, label=bad)


def test_builder_only_writes_where_training_checks(tmp_path):
    from eval.target_voices import DEFAULT_TARGET_DIR
    from xtts_trainer.data.validate import DEFAULT_TARGET_LOCK

    assert os.path.join(DEFAULT_TARGET_DIR, LOCK_FILENAME) == DEFAULT_TARGET_LOCK
    with pytest.raises(ValueError, match="only lock location"):
        _build(_dataset(str(tmp_path)), str(tmp_path / "elsewhere"))


def test_rebuilding_replaces_the_previous_holdout(tmp_path, monkeypatch):
    """The source dataset contains the previous holdout by definition; rebuilding must not trip the guard."""
    import xtts_trainer.data.validate as validate_module

    dataset = _dataset(str(tmp_path))
    out = str(tmp_path / "targets")
    monkeypatch.setattr(validate_module, "DEFAULT_TARGET_LOCK", os.path.join(out, LOCK_FILENAME))
    first = build_target_voices(dataset, out, seed=1)
    second = build_target_voices(dataset, out, seed=2)  # the lock from the first run now exists
    assert first["voices"].keys() == second["voices"].keys()


def test_speaker_ids_must_be_safe_file_names(tmp_path):
    records = [DatasetRecord(audio=f"c{i}.wav", text="hello", lang="en", speaker="../escape", duration_s=2.0, sr=SR,
                             transcript_source="human") for i in range(6)]
    for i in range(6):
        sf.write(str(tmp_path / f"c{i}.wav"), (0.1 * np.random.RandomState(i).randn(SR * 2)).astype("float32"), SR)
    dataset = write_dataset(str(tmp_path / "dataset.jsonl"), records)
    with pytest.raises(ValueError, match="can't be file names"):
        build_target_voices(dataset, str(tmp_path / "targets"))
    assert not (tmp_path / "escape.wav").exists()


def test_holdout_search_continues_past_clips_that_trim_short(tmp_path):
    """Mostly-silent clips trim to almost nothing; the search must keep loading clips, not stop at a fixed prefix."""
    records = []
    for i in range(14):
        rng = np.random.RandomState(i)
        audio = np.zeros(int(SR * 2.0), dtype="float32")
        if i >= 10:  # only the last four clips have 2 s of signal
            t = np.arange(len(audio)) / SR
            audio = (0.2 * np.sin(2 * np.pi * rng.uniform(100, 800) * t)).astype("float32")
        else:
            audio[: SR // 10] = 0.2 * rng.randn(SR // 10)  # 0.1 s of signal, then silence
        sf.write(str(tmp_path / f"c{i}.wav"), audio, SR)
        records.append(DatasetRecord(audio=f"c{i}.wav", text="hello", lang="en", speaker="me", duration_s=2.0,
                                     sr=SR, transcript_source="human"))
    dataset = write_dataset(str(tmp_path / "dataset.jsonl"), records)
    lock = build_target_voices(dataset, str(tmp_path / "targets"), seed=0)
    assert MIN_CLIP_S <= lock["voices"]["target_me"]["duration_s"] <= MAX_CLIP_S


def test_holdout_keeps_only_trimmed_audio(tmp_path, monkeypatch):
    """Every loaded clip is edge-trimmed as it is loaded, so silent clips don't pile up full-length in memory."""
    import eval.target_voices as target_module

    trimmed = []
    real_trim = target_module.trim_edges
    monkeypatch.setattr(target_module, "trim_edges", lambda wav, rate: trimmed.append(len(wav)) or real_trim(wav, rate))
    lock = build_target_voices(_dataset(str(tmp_path), speakers=("me",), n=8), str(tmp_path / "targets"))
    assert len(trimmed) >= len(lock["voices"]["target_me"]["source_clips"])
