"""Phase 3: the dataset contract. Each required bad fixture must be rejected with a specific error code."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import soundfile as sf

from xtts_trainer.data import (
    DatasetFormatError,
    DatasetRecord,
    DatasetValidationError,
    load_dataset,
    validate_dataset,
    write_dataset,
)
from xtts_trainer.data.validate import ValidationConfig
from dataset_helpers import write_test_vocab

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 22050


def _write_vocab(path):
    """Minimal XTTS-style tokenizer without q, x, z (they tokenize to [UNK])."""
    return write_test_vocab(path, letters="abcdefghijklmnoprstuvwy")


class Builder:
    """Writes clips and a dataset.jsonl into a temp dir; each test starts from a valid baseline."""

    def __init__(self, root):
        self.root = str(root)
        os.makedirs(os.path.join(self.root, "clips"), exist_ok=True)
        self.records = []
        self.vocab = _write_vocab(os.path.join(self.root, "vocab.json"))

    def clip(self, name, seconds, seed=0, sr=SR):
        rng = np.random.RandomState(seed)
        path = os.path.join(self.root, "clips", name)
        sf.write(path, (rng.randn(int(seconds * sr)) * 0.1).astype(np.float32), sr)
        return f"clips/{name}"

    def add(self, text, seconds=1.2, speaker="spk1", lang="en", seed=None, **overrides):
        audio = self.clip(f"c{len(self.records):03d}.wav", seconds, seed=len(self.records) if seed is None else seed)
        fields = dict(audio=audio, text=text, lang=lang, speaker=speaker, duration_s=seconds, sr=SR,
                      transcript_source="human")
        fields.update(overrides)
        self.records.append(DatasetRecord(**fields))
        return self.records[-1]

    def valid_base(self):
        self.add("hello world.")
        self.add("good morning to you.", seconds=1.6)
        return self

    def validate(self, **kwargs):
        path = write_dataset(os.path.join(self.root, "dataset.jsonl"), self.records)
        kwargs.setdefault("eval_set_path", os.path.join(REPO_ROOT, "eval", "dataset", "eval_set.json"))
        kwargs.setdefault("reference_lock_path", None)
        return validate_dataset(path, self.vocab, **kwargs)


def codes(report):
    return {e.code for e in report.errors}


def test_valid_dataset_passes(tmp_path):
    report = Builder(tmp_path).valid_base().validate()
    assert report.ok, report.errors
    assert report.stats["clips"] == 2 and report.stats["eval_overlap_checked"]


# --- The rejection fixtures required by docs/recipe_design.md §6, phase 3 exit criteria -------------------------------

def test_rejects_short_clip_with_full_sentence(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("this is a very long sentence that could never be spoken in such a tiny clip of audio at all", seconds=1.0)
    assert "chars_per_second" in codes(b.validate())


def test_rejects_empty_transcript(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("   ")
    assert "empty_text" in codes(b.validate())


def test_rejects_unknown_language(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("hello there.", lang="xx")
    assert "unknown_language" in codes(b.validate())


def test_rejects_registered_language_missing_from_vocab(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("hola amigos.", lang="es")
    assert "language_not_in_vocab" in codes(b.validate())


def test_rejects_unk_characters(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("a quiz.")
    assert "unk_tokens" in codes(b.validate())


def test_rejects_single_clip_speaker_unless_marked(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("only one clip here.", seconds=1.5, speaker="loner")
    assert "single_clip_speaker" in codes(b.validate())
    b.records[-1].single_clip_speaker = True
    assert b.validate().ok


def test_rejects_eval_set_text_overlap(tmp_path):
    with open(os.path.join(REPO_ROOT, "eval", "dataset", "eval_set.json"), "r", encoding="utf-8") as f:
        eval_text = json.load(f)["english_sentences"][0]["text"]
    b = Builder(tmp_path).valid_base()
    b.add(eval_text.replace("z", "s").replace("q", "k").replace("x", "ks"), seconds=2.5)
    b.records[-1].text = eval_text  # the exact eval sentence
    assert "eval_overlap_text" in codes(b.validate())


def test_rejects_reference_voice_audio_overlap(tmp_path):
    b = Builder(tmp_path).valid_base()
    leaked = b.add("good evening.", seconds=1.2)
    from eval.reference_voices import sha256_file

    lock = os.path.join(b.root, "lock.json")
    with open(lock, "w", encoding="utf-8") as f:
        json.dump({"voices": {"arctic_slt": {"clip_sha256": sha256_file(leaked.audio_path(b.root))}}}, f)
    assert "eval_overlap_audio" in codes(b.validate(reference_lock_path=lock))


# --- Other content checks -------------------------------------------------------------------------------

def test_rejects_audio_problems(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("hello again.", duration_s=3.0)                 # declared duration wrong
    b.add("hello there.", sr=16000)                       # declared rate wrong
    b.add("hi.", seconds=0.3)                             # under upstream's 0.5 s minimum
    b.add("a long one.", seconds=12.0)                    # over upstream max_wav_length (~11.6 s)
    b.add("hello world.", seed=0)                         # same audio as line 2
    b.records.append(DatasetRecord(audio="clips/missing.wav", text="hello.", lang="en", speaker="spk1",
                                   duration_s=1.0, sr=SR, transcript_source="human"))
    found = codes(b.validate())
    assert {"duration_mismatch", "sr_mismatch", "too_short", "too_long", "duplicate_audio", "audio_missing"} <= found


def test_asr_cross_check(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.records[0].transcript_source = "asr:openai/whisper-large-v3"
    config = ValidationConfig(asr_cer_max=0.1)
    ok = b.validate(config=config, transcriber=lambda path, r: r.text)
    assert ok.ok and ok.stats["asr_checked"]
    bad = b.validate(config=config, transcriber=lambda path, r: "something else entirely")
    assert "asr_mismatch" in codes(bad)


def test_raise_for_errors_lists_lines(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.add("", seconds=1.0)
    with pytest.raises(DatasetValidationError, match=r"line 4 \[empty_text\]"):
        b.validate().raise_for_errors()


def test_short_clip_coverage_warning(tmp_path):
    b = Builder(tmp_path)
    b.add("hello world, how are you today.", seconds=2.5)
    b.add("good morning to you all, my friends.", seconds=2.8)
    report = b.validate()
    assert report.ok and any("short-utterance coverage" in w for w in report.warnings)


# --- Structural (format) errors -------------------------------------------------------------------------

def _write_raw(tmp_path, lines):
    path = os.path.join(str(tmp_path), "dataset.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


GOOD = '{"audio": "a.wav", "text": "hi", "lang": "en", "speaker": "s", "duration_s": 1.0, "sr": 22050, "transcript_source": "human"}'


@pytest.mark.parametrize("lines, message", [
    ([GOOD], "header"),
    (['{"format": "open-xtts-dataset", "version": 2}', GOOD], "header"),
    (['{"format": "open-xtts-dataset", "version": 1}'], "no clips"),
    (['{"format": "open-xtts-dataset", "version": 1}', GOOD.replace('"sr"', '"rate"')], "unknown keys"),
    (['{"format": "open-xtts-dataset", "version": 1}', GOOD.replace('"human"', '"guess"')], "transcript_source"),
    (['{"format": "open-xtts-dataset", "version": 1}', GOOD.replace("22050", '"22050"')], "type"),
    (['{"format": "open-xtts-dataset", "version": 1}', "a.wav|hi|en"], "invalid JSON"),
])
def test_format_errors(tmp_path, lines, message):
    with pytest.raises(DatasetFormatError, match=message):
        load_dataset(_write_raw(tmp_path, lines))


def test_write_then_load_round_trip(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.records[0].alignment = {"start_s": 0.0, "end_s": 1.2}
    path = write_dataset(os.path.join(b.root, "dataset.jsonl"), b.records)
    assert [r.to_json() for r in load_dataset(path)] == [r.to_json() for r in b.records]


REAL_VOCAB = os.path.join(REPO_ROOT, "xtts_base", "vocab.json")


@pytest.mark.skipif(not os.path.exists(REAL_VOCAB), reason="needs xtts_base/vocab.json (scripts/fetch_base_artifacts.py)")
def test_real_vocab_unk_detection(tmp_path):
    b = Builder(tmp_path).valid_base()
    b.vocab = REAL_VOCAB
    b.add("hello snowman ☃.", seconds=1.6)
    assert "unk_tokens" in codes(b.validate())


# --- Stale latent files (Phase 3 exit criterion) --------------------------------------------------------------

def _latent_setup(tmp_path, n=2):
    import torch
    from dataset_helpers import write_test_dataset
    from xtts_trainer.data.latents import make_latent_payload

    root = str(tmp_path)
    latents = os.path.join(root, "latents")
    os.makedirs(latents)
    wavs = []
    for i in range(n):
        path = os.path.join(root, f"clip{i}.wav")
        sf.write(path, (np.random.RandomState(i).randn(SR * 2) * 0.1).astype(np.float32), SR)
        wavs.append(path)
    records = load_dataset(write_test_dataset(root, wavs))

    def write(i, checkpoint="ckpt-A", **overrides):
        payload = make_latent_payload(torch.randn(1, 40, 1024), torch.randn(1, 512, 1), wavs[i], "hi", "en", checkpoint)
        payload.update(overrides)
        torch.save(payload, os.path.join(latents, f"clip{i}.pt"))

    return root, latents, wavs, records, write


def test_latents_accepted_when_current(tmp_path):
    from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset

    root, latents, wavs, records, write = _latent_setup(tmp_path)
    write(0)
    write(1)
    dataset = LatentHiFiGANDataset(records, root, latents)
    assert len(dataset) == 2 and dataset.checkpoint_sha256 == "ckpt-A"


def test_rejects_stale_latent_after_audio_change(tmp_path):
    from xtts_trainer.data.latents import StaleLatentError
    from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset

    root, latents, wavs, records, write = _latent_setup(tmp_path)
    write(0)
    write(1)
    sf.write(wavs[1], (np.random.RandomState(99).randn(SR * 2) * 0.1).astype(np.float32), SR)  # re-recorded
    with pytest.raises(StaleLatentError, match="changed since extraction"):
        LatentHiFiGANDataset(records, root, latents)


def test_rejects_latents_from_mixed_checkpoints(tmp_path):
    from xtts_trainer.data.latents import StaleLatentError
    from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset

    root, latents, wavs, records, write = _latent_setup(tmp_path)
    write(0, checkpoint="ckpt-A")
    write(1, checkpoint="ckpt-B")
    with pytest.raises(StaleLatentError, match="different GPT checkpoints"):
        LatentHiFiGANDataset(records, root, latents)


def test_rejects_synthetic_and_old_format_latents(tmp_path):
    import torch
    from xtts_trainer.data.latents import SYNTHETIC_CHECKPOINT, StaleLatentError
    from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset

    root, latents, wavs, records, write = _latent_setup(tmp_path)
    write(0, checkpoint=SYNTHETIC_CHECKPOINT)
    write(1, checkpoint=SYNTHETIC_CHECKPOINT)
    with pytest.raises(StaleLatentError, match="synthetic"):
        LatentHiFiGANDataset(records, root, latents)

    torch.save({"gpt_latents": torch.randn(1, 40, 1024), "speaker_embedding": torch.randn(1, 512, 1)},
               os.path.join(latents, "clip0.pt"))  # pre-schema file from the old extractor
    with pytest.raises(StaleLatentError, match="older extractor"):
        LatentHiFiGANDataset(records, root, latents, allow_synthetic_latents=True)


def test_rejects_latent_with_other_spec(tmp_path):
    from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset

    root, latents, wavs, records, write = _latent_setup(tmp_path)
    write(0, spec_hash="0000000000000000")
    write(1)
    with pytest.raises(RuntimeError, match="spec_hash"):
        LatentHiFiGANDataset(records, root, latents)


def test_positional_arguments_keep_their_meaning(tmp_path):
    """target_lock_path was added last: a caller passing config positionally (6th) must still pass a config."""
    b = Builder(tmp_path).valid_base()
    path = write_dataset(os.path.join(b.root, "dataset.jsonl"), b.records)
    strict = ValidationConfig(min_duration_s=5.0)  # makes every clip too short
    report = validate_dataset(path, b.vocab, (), None, None, strict)
    assert "too_short" in codes(report)
