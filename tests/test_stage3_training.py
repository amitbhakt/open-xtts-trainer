"""Stage 3 (HiFi-GAN decoder): segments, validation, warm-up rule, optional losses, best checkpoint and resume."""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest
import soundfile as sf
import torch

from dataset_helpers import write_synthetic_latent, write_test_dataset
from xtts_trainer.config import HiFiGANConfig
from xtts_trainer.data import load_dataset
from xtts_trainer.models import build_hifigan, hifigan_decoder_input, safe_torch_load, stamp
from xtts_trainer.spec import SPEC
from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset, collate_segments
from xtts_trainer.stage3_hifigan.extract_latents import extract_latents
from xtts_trainer.stage3_hifigan.train import (
    full_band_loss_mel, speaker_consistency_loss, train_hifigan,
)

SR = SPEC.output_sample_rate
HOP = SPEC.output_hop_length


@pytest.fixture(scope="module")
def decoder():
    torch.manual_seed(0)
    return build_hifigan().eval()


def _data(tmp_path, n=5):
    rng = np.random.RandomState(0)
    paths = []
    for i in range(n):
        t = np.arange(int(SR * (1.2 + 0.2 * i))) / SR
        audio = sum(0.1 * np.sin(2 * np.pi * f * t) for f in rng.uniform(100, 4000, 4))
        path = str(tmp_path / f"c{i}.wav")
        sf.write(path, audio.astype("float32"), SR)
        paths.append(path)
    dataset = write_test_dataset(str(tmp_path), paths)
    extract_latents(dataset, str(tmp_path / "latents"), device="cpu", allow_synthetic_fallback=True)
    return dataset


def _config(**kw):
    defaults = dict(batch_size=2, epochs=2, segment_frames=16, val_fraction=0.2, val_every_steps=2,
                    sample_every_steps=2, lr_warmup_steps=1, disc_warmup_steps=1, mixed_precision=False)
    defaults.update(kw)
    return HiFiGANConfig(**defaults)


def test_decoder_input_reproduces_hifidecoder_forward(decoder):
    """hifigan_decoder_input + waveform_decoder == HifiDecoder.forward, bit for bit (batch 1 and 2)."""
    for batch in (1, 2):
        latents, speaker = torch.randn(batch, 10, SPEC.gpt.model_dim), torch.randn(batch, SPEC.d_vector_dim, 1)
        with torch.no_grad():
            whole = decoder(latents, g=speaker).reshape(batch, -1)
            frames = hifigan_decoder_input(decoder, latents)
            split = decoder.waveform_decoder(frames, g=speaker).reshape(batch, -1)
        assert frames.shape[-1] * HOP == whole.shape[-1] == SPEC.vocoder_num_samples(10)
        assert torch.equal(whole, split)


def test_dataset_default_interpolation_matches_the_decoder(tmp_path, decoder):
    dataset = _data(tmp_path, n=2)
    records = load_dataset(dataset)
    default = LatentHiFiGANDataset(records, str(tmp_path), str(tmp_path / "latents"), allow_synthetic_latents=True)
    assert vars(default.decoder) == {k: getattr(decoder, k) for k in vars(default.decoder)}


def test_segments_skip_silence_and_collate_without_padding(tmp_path, decoder, monkeypatch):
    silence, tone = torch.zeros(1, 3 * SR), torch.sin(torch.linspace(0, 4000, SR)).unsqueeze(0) * 0.3
    wav = str(tmp_path / "late.wav")
    sf.write(wav, torch.cat([silence, tone], -1).reshape(-1).numpy(), SR)
    (tmp_path / "latents").mkdir()
    write_synthetic_latent(str(tmp_path / "latents" / "late.pt"), wav, 80)
    records = load_dataset(write_test_dataset(str(tmp_path), [wav]))
    monkeypatch.setattr(random, "randint", lambda a, b: 0)  # random draws all land in the silence
    dataset = LatentHiFiGANDataset(records, str(tmp_path), str(tmp_path / "latents"), decoder, 16, True)
    audio, frames, speaker = dataset[0]
    assert audio.shape[-1] == frames.shape[-1] * HOP == 16 * HOP
    assert audio.pow(2).mean().sqrt() >= 0.015  # the deterministic scan found the tone
    short = (audio[:, :8 * HOP], frames[:, :8], speaker)
    a, f, s = collate_segments([(audio, frames, speaker), short])
    assert a.shape == (2, 1, 8 * HOP) and f.shape == (2, SPEC.decoder_input_dim, 8) and s.shape == (2, SPEC.d_vector_dim, 1)


def test_training_validates_keeps_best_and_writes_samples(tmp_path):
    dataset = _data(tmp_path)
    best = train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / "out"), _config(),
                         device="cpu", allow_synthetic_latents=True)
    ckpt = safe_torch_load(best)
    metrics = [json.loads(l) for l in Path(tmp_path / "out" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ckpt["val_mel_l1"] == min(m["val_mel_l1"] for m in metrics)
    assert ckpt["reference_val_mel_l1"] == metrics[0]["reference"] and "beats_start" in ckpt
    assert sorted(os.listdir(tmp_path / "out" / "samples")) == ["step_0", "step_2", "step_4"]
    build_hifigan().load_state_dict(ckpt["hifigan_decoder"], strict=True)


def test_resume_replays_the_same_data_stream(tmp_path, monkeypatch):
    """Resume restores the loader order and the window draws: the batches after resuming are bit-identical to
    an uninterrupted run's. Weights are not compared: CPU FFT/conv results differ in the last bits between
    identical runs and GAN training amplifies that (0.4% after 6 steps under the full suite's threading).
    Bit-exact model/optimizer/RNG restore is covered by tests/test_training_hygiene.py and Stage 1."""
    import xtts_trainer.stage3_hifigan.train as train_module

    seen = []
    original = train_module.collate_segments

    def spy(batch):
        out = original(batch)
        seen.append([t.clone() for t in out])
        return out

    monkeypatch.setattr(train_module, "collate_segments", spy)
    dataset = _data(tmp_path)
    kw = dict(device="cpu", allow_synthetic_latents=True)
    straight = safe_torch_load(train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / "a"), _config(epochs=3), **kw))
    straight_batches, seen[:] = list(seen), []
    train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / "b"), _config(epochs=2), **kw)
    resumed = safe_torch_load(train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / "b"), _config(epochs=3),
                                            resume=True, **kw))
    assert len(seen) == len(straight_batches)
    for ours, theirs in zip(seen, straight_batches):
        assert all(torch.equal(x, y) for x, y in zip(ours, theirs))
    assert straight["step"] == resumed["step"]


def test_two_identical_runs_draw_identical_segments(tmp_path, monkeypatch):
    """Regression: windows were drawn with python's global random, whose stream other code consumes, so two
    identical runs trained on different segments (val error 6.45 vs 6.57 after 4 steps)."""
    import xtts_trainer.stage3_hifigan.train as train_module

    runs = []
    original = train_module.collate_segments
    monkeypatch.setattr(train_module, "collate_segments", lambda b: runs[-1].append(original(b)) or runs[-1][-1])
    dataset = _data(tmp_path)
    for name in ("x", "y"):
        runs.append([])
        random.random()  # unrelated use of python's RNG between runs must not matter
        train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / name), _config(epochs=1),
                      device="cpu", allow_synthetic_latents=True)
    assert all(torch.equal(p, q) for a, b in zip(*runs) for p, q in zip(a, b))


def test_finetune_that_never_leaves_disc_warmup_is_refused(tmp_path, decoder):
    dataset = _data(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    torch.save(stamp({"model": {f"hifigan_decoder.{k}": v for k, v in decoder.state_dict().items()}}), base / "model.pth")
    with pytest.raises(ValueError, match="never leave"):
        train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / "out"), _config(disc_warmup_steps=1000),
                      checkpoint_dir=str(base), device="cpu", allow_synthetic_latents=True)
    best = train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / "out2"),
                         _config(disc_warmup_steps=1000, mel_only=True, epochs=1),
                         checkpoint_dir=str(base), device="cpu", allow_synthetic_latents=True)
    metrics = [json.loads(l) for l in Path(tmp_path / "out2" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metrics[0]["reference"] == safe_torch_load(best)["reference_val_mel_l1"]


def test_speaker_consistency_loss(decoder):
    audio = (torch.sin(torch.linspace(0, 3000, SR)) * 0.3).reshape(1, 1, -1)
    other = (torch.randn(1, 1, SR) * 0.3).requires_grad_(True)
    encoder = decoder.speaker_encoder.eval()
    assert speaker_consistency_loss(encoder, audio, audio).item() == pytest.approx(0.0, abs=1e-5)
    loss = speaker_consistency_loss(encoder, other, audio)
    loss.backward()
    assert loss.item() > 0.01 and other.grad is not None and other.grad.abs().sum() > 0
    assert other.shape == (1, 1, SR)  # the encoder squeezes its input in place; ours must be untouched


def test_full_band_mel_reaches_nyquist():
    mel = full_band_loss_mel()
    assert mel.mel.f_max == SR / 2 and SPEC.vocoder_loss_mel.fmax == 8000.0


def test_extract_latents_standalone(tmp_path):
    wav = str(tmp_path / "test.wav")
    sf.write(wav, np.sin(np.linspace(0, 440, SR)).astype("float32"), SR)
    extract_latents(write_test_dataset(str(tmp_path), [wav]), str(tmp_path / "latents"), checkpoint_dir="",
                    device="cpu", allow_synthetic_fallback=True)
    data = safe_torch_load(str(tmp_path / "latents" / "test.pt"))
    assert data["checkpoint_sha256"] == "synthetic" and data["schema_version"] == 1
    assert data["gpt_latents"].shape[-1] == SPEC.gpt.model_dim and data["speaker_embedding"].shape == (1, 512, 1)


def test_stage3_inference_synthesize_audio(tmp_path, decoder):
    from xtts_trainer.stage3_hifigan.inference import synthesize_audio

    ckpt, latent, out = str(tmp_path / "hifigan.pth"), str(tmp_path / "latent.pt"), str(tmp_path / "out.wav")
    torch.save({"hifigan_decoder": decoder.state_dict()}, ckpt)
    torch.save({"gpt_latents": torch.randn(1, 16, 1024), "speaker_embedding": torch.randn(1, 512, 1)}, latent)
    synthesize_audio(checkpoint_path=ckpt, latent_path=latent, output_path=out, device="cpu")
    data, sr = sf.read(out)
    assert sr == SR and len(data) == SPEC.vocoder_num_samples(16)


def test_best_decoder_is_the_best_validation_not_the_last(tmp_path, monkeypatch):
    import xtts_trainer.stage3_hifigan.train as train_module

    scripted = iter([2.0, 1.0, 0.5, 0.9])  # starting decoder, then validations: improves, then gets worse
    monkeypatch.setattr(train_module, "evaluate_decoder", lambda *a, **k: next(scripted))
    dataset = _data(tmp_path)
    best = safe_torch_load(train_hifigan(dataset, str(tmp_path / "latents"), str(tmp_path / "out"), _config(epochs=3),
                                         device="cpu", allow_synthetic_latents=True))
    assert best["val_mel_l1"] == 0.5 and best["step"] == 4 and best["beats_start"]
