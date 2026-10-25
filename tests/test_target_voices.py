"""Target-speaker evaluation: held-out reference clips, the leakage guard, and eval with --voices target."""

import json
import os
import shutil
import sys
from pathlib import Path

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


def _dataset(root, speakers=("me", "you"), n=8, seconds=2.0, seed=0):
    records = []
    for s, speaker in enumerate(speakers):
        for i in range(n):
            rng = np.random.RandomState(seed + 100 * s + i)
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


def test_training_on_held_out_clips_is_refused(tmp_path, monkeypatch):
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
    assert not any("lock is missing" in w for w in leaked.warnings)
    # The lock gone but its clips still there: fail closed rather than train with the guard disabled.
    os.remove(target_lock)
    unguarded = validate_dataset(dataset, vocab, **kw)
    assert not unguarded.ok and not unguarded.stats["target_holdout_checked"]
    assert [e.code for e in unguarded.errors] == ["target_holdout_lock_missing"]
    # A bare relative lock name must check its current directory too.
    monkeypatch.chdir(tmp_path)
    sf.write("target_bare.wav", np.zeros(SR, dtype="float32"), SR)
    bare = validate_dataset(dataset, vocab, eval_set_path=None, reference_lock_path=None,
                            target_lock_path=LOCK_FILENAME)
    assert [e.code for e in bare.errors] == ["target_holdout_lock_missing"]
    # No holdout ever built: nothing to warn about
    clean = validate_dataset(dataset, vocab, **{**kw, "target_lock_path": str(tmp_path / "none" / LOCK_FILENAME)})
    assert not any("lock is missing" in w for w in clean.warnings)


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

    # Every call runs from tmp_path: a call that should be refused can't write a manifest into the repository
    monkeypatch.chdir(tmp_path)
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
        voices="target", target_dir=out, label="base", allow_unchecked_location=True,
    )
    assert manifest["voices"] == "target"
    assert [v["speaker_id"] for v in manifest["reference_voices"]] == ["target_me", "target_you"]
    # each voice id was synthesized from its own held-out reference clip, in order
    assert conditioned == [os.path.join(out, "target_me.wav"), os.path.join(out, "target_you.wav")]
    assert [run["speaker_id"] for run in manifest["runs"]] == ["target_me", "target_you"]
    for pooled in ("everyone", "all"):  # target and eval voices are never pooled into one manifest
        with pytest.raises(ValueError, match="voices must be"):
            baseline_module.generate_baseline(checkpoint_dir=str(ckpt), voices=pooled)
    # Clips from anywhere but the canonical directory would be clips training never refuses
    with pytest.raises(ValueError, match="read only from"):
        baseline_module.generate_baseline(checkpoint_dir=str(ckpt), voices="target", target_dir=out)
    # Without --output, each label and voice set gets its own file: a target run can't overwrite another run
    for label in ("base", "tuned"):
        baseline_module.generate_baseline(checkpoint_dir=str(ckpt), seeds=[1], languages=["en"], subset_size=2,
                                          voices="target", target_dir=out, label=label,
                                          allow_unchecked_location=True)
    written = sorted(os.listdir(tmp_path / "eval" / "baselines"))
    commit = baseline_module.PINNED_UPSTREAM_COMMIT[:12]
    assert written == [f"base_xtts_{commit}_target.json", f"tuned_xtts_{commit}_target.json"]
    # ...and, run from another directory, still measured against the frozen eval set (absolute default path)
    from eval.dataset import DEFAULT_EVAL_SET_PATH

    frozen = json.loads(Path(DEFAULT_EVAL_SET_PATH).read_text(encoding="utf-8"))["sha256"]
    on_disk = json.loads(Path(tmp_path / "eval" / "baselines" / f"base_xtts_{commit}_target.json").read_text(encoding="utf-8"))
    assert on_disk["eval_set_sha256"] == frozen and not (tmp_path / "eval" / "dataset").exists()
    for bad in ("../escape", "a/b", ".hidden", ""):
        with pytest.raises(ValueError, match="plain name"):
            baseline_module.generate_baseline(checkpoint_dir=str(ckpt), voices="target", target_dir=out, label=bad,
                                              allow_unchecked_location=True)
    for bad_batch in (0, -1, -16):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            baseline_module.generate_baseline(checkpoint_dir=str(ckpt), batch_size=bad_batch)


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
    """Clips are edge-trimmed as they load, so silence-padded clips don't pile up full-length in memory."""
    import eval.target_voices as target_module

    raw_len, records = int(SR * 3.0), []
    for i in range(8):  # 1 s silence, 1 s tone, 1 s silence
        audio = np.zeros(raw_len, dtype="float32")
        t = np.arange(SR) / SR
        audio[SR:2 * SR] = 0.2 * np.sin(2 * np.pi * (200 + 50 * i) * t)
        sf.write(str(tmp_path / f"c{i}.wav"), audio, SR)
        records.append(DatasetRecord(audio=f"c{i}.wav", text="hello", lang="en", speaker="me", duration_s=3.0,
                                     sr=SR, transcript_source="human"))
    dataset = write_dataset(str(tmp_path / "dataset.jsonl"), records)
    retained = []
    real_assemble = target_module.assemble_clip
    monkeypatch.setattr(target_module, "assemble_clip",
                        lambda loaded, rate: retained.append([len(a) for a in loaded]) or real_assemble(loaded, rate))
    build_target_voices(dataset, str(tmp_path / "targets"))
    # Every array the search holds is the trimmed one: well under the 3 s file, and about the 1 s of tone
    held = [n for call in retained for n in call]
    assert held and max(held) < 1.5 * SR < raw_len


def test_failed_build_leaves_the_previous_holdout_untouched(tmp_path):
    """A later speaker failing its check must not overwrite earlier speakers' clips under the old lock."""
    root = tmp_path / "data"
    root.mkdir()
    _dataset(str(root), speakers=("me",), n=10)
    records = load_dataset(str(root / "dataset.jsonl"))
    _dataset(str(root), speakers=("you",), n=6, seed=1000)  # 'you' keeps fewer training clips than 'me'
    dataset = write_dataset(str(root / "dataset.jsonl"), records + load_dataset(str(root / "dataset.jsonl")))
    out = str(tmp_path / "targets")
    first = build_target_voices(dataset, out, seed=1)
    kept = {vid: (10 if vid == "target_me" else 6) - len(e["source_clips"]) for vid, e in first["voices"].items()}
    assert kept["target_me"] > kept["target_you"]
    before = {name: (tmp_path / "targets" / name).read_bytes() for name in sorted(os.listdir(out))}
    # 'me' (processed first) passes this minimum and 'you' fails it, so only an atomic build leaves 'me' alone
    with pytest.raises(ValueError, match="speaker 'you'"):
        build_target_voices(dataset, out, seed=2, min_train_clips=kept["target_you"] + 1)
    assert {name: (tmp_path / "targets" / name).read_bytes() for name in sorted(os.listdir(out))} == before
    verify_target_voices(out)


def test_write_failure_does_not_publish_a_partial_target_set(tmp_path, monkeypatch):
    """Audio and lock writes happen in staging, so a later write failure preserves the prior complete build."""
    import eval.target_voices as target_module

    dataset = _dataset(str(tmp_path))
    out = str(tmp_path / "targets")
    first = build_target_voices(dataset, out, seed=1)
    before = {name: (tmp_path / "targets" / name).read_bytes() for name in os.listdir(out)}
    train_before = Path(first["train_dataset"]).read_bytes()
    real_write = target_module.sf.write
    calls = 0

    def fail_second_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(target_module.sf, "write", fail_second_write)
    with pytest.raises(OSError, match="disk full"):
        build_target_voices(dataset, out, seed=2)
    assert {name: (tmp_path / "targets" / name).read_bytes() for name in os.listdir(out)} == before
    assert Path(first["train_dataset"]).read_bytes() == train_before
    verify_target_voices(out)


def test_building_from_the_training_split_is_refused(tmp_path):
    """dataset_train.jsonl lacks the previous holdout; rebuilding from it would drop those clips for good."""
    lock = build_target_voices(_dataset(str(tmp_path)), str(tmp_path / "targets"))
    with pytest.raises(ValueError, match="training split"):
        build_target_voices(lock["train_dataset"], str(tmp_path / "targets"))


def test_should_reuse_cached_base_requires_explicit_batch_size(tmp_path):
    """Cached baseline reuse requires matching explicit batch_size and rejects legacy sequential manifests."""
    import hashlib
    from scripts.run_target_finetune_ab import should_reuse_cached_base, EVAL_BATCH_SIZE

    # Create dummy checkpoint files
    base_dir = tmp_path / "xtts_base"
    base_dir.mkdir()
    hashes = {}
    for fname in ("model.pth", "config.json", "vocab.json"):
        content = f"dummy {fname}".encode()
        (base_dir / fname).write_bytes(content)
        hashes[fname] = hashlib.sha256(content).hexdigest()

    valid_manifest = {
        "seeds": [42, 123, 999],
        "subset_size": 12,
        "languages": ["en"],
        "asr_model": "openai/whisper-large-v3",
        "checkpoint_sha256": hashes,
        "batch_size": EVAL_BATCH_SIZE,
    }

    # 1. Valid manifest with explicit batch_size matching EVAL_BATCH_SIZE is accepted
    assert should_reuse_cached_base(
        valid_manifest,
        expected_seeds=[42, 123, 999],
        expected_subset=12,
        xtts_base_dir=str(base_dir),
        eval_batch_size=EVAL_BATCH_SIZE,
    ) is True

    # 2. Legacy manifest without explicit batch_size is rejected
    legacy_manifest = dict(valid_manifest)
    del legacy_manifest["batch_size"]
    assert should_reuse_cached_base(
        legacy_manifest,
        expected_seeds=[42, 123, 999],
        expected_subset=12,
        xtts_base_dir=str(base_dir),
        eval_batch_size=EVAL_BATCH_SIZE,
    ) is False

    # 3. Manifest with mismatched batch_size is rejected
    mismatched_batch = dict(valid_manifest, batch_size=8)
    assert should_reuse_cached_base(
        mismatched_batch,
        expected_seeds=[42, 123, 999],
        expected_subset=12,
        xtts_base_dir=str(base_dir),
        eval_batch_size=EVAL_BATCH_SIZE,
    ) is False
