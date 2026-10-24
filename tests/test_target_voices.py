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
from eval.target_voices import LOCK_FILENAME, build_target_voices, holdout_hashes, verify_target_voices
from xtts_trainer.data import DatasetRecord, load_dataset, validate_dataset, write_dataset
from xtts_trainer.spec import SPEC

SR = SPEC.output_sample_rate


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
            assert os.path.basename(audio_path[0]).startswith("target_")
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
    with pytest.raises(ValueError, match="voices must be"):
        baseline_module.generate_baseline(checkpoint_dir=str(ckpt), voices="everyone")
