"""Tests for Stage 3: XTTS v2 Latent HiFi-GAN Vocoder Training."""

import os
import shutil
import tempfile
import torch
import torchaudio
import pytest

from xtts_trainer.config import HiFiGANConfig, safe_torch_load
from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset
from xtts_trainer.stage3_hifigan.models import (
    build_hifigan_decoder,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from xtts_trainer.stage3_hifigan.train import train_hifigan
from xtts_trainer.stage3_hifigan.extract_latents import extract_latents


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_hifigan_forward_backward_gradient_updates():
    """Verify that HifiDecoder generates matching audio waveform and computes valid gradients."""
    decoder = build_hifigan_decoder()
    decoder.train()

    latents = torch.randn(2, 16, 1024)
    speaker_emb = torch.randn(2, 512, 1)

    wav_gen = decoder(latents, g=speaker_emb)
    # 16 frames produces exactly 17,664 audio samples at 24kHz
    assert wav_gen.shape == (2, 1, 17664)

    target_wav = torch.randn(2, 1, 17664)
    loss = torch.nn.functional.l1_loss(wav_gen, target_wav)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    # Verify gradients exist on waveform_decoder parameters
    has_grad = any(p.grad is not None and p.grad.norm().item() > 0.0 for p in decoder.parameters() if p.requires_grad)
    assert has_grad

    optimizer.step()


def test_stage3_dataset_and_training_loop(temp_workspace):
    """End-to-end integration test: temporal latent dataset, training loop, and checkpoint persistence."""
    audio_dir = os.path.join(temp_workspace, "audio")
    latents_dir = os.path.join(temp_workspace, "latents")
    output_dir = os.path.join(temp_workspace, "checkpoints")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    # Create 3 synthetic 24kHz audio clips (~1.5s = 36000 samples)
    for i in range(3):
        wav_path = os.path.join(audio_dir, f"sample_{i}.wav")
        wav = torch.sin(torch.linspace(0, 400 * (i + 1), 36000)).unsqueeze(0)
        torchaudio.save(wav_path, wav, 24000)

        # Create proportional temporal latents: ~32 frames
        lat_path = os.path.join(latents_dir, f"sample_{i}.pt")
        torch.save(
            {
                "gpt_latents": torch.randn(1, 32, 1024),
                "speaker_embedding": torch.randn(1, 512, 1),
            },
            lat_path,
        )

    config = HiFiGANConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
        discriminator_learning_rate=1e-4,
    )

    ckpt_path = train_hifigan(
        audio_dir=audio_dir,
        latents_dir=latents_dir,
        output_dir=output_dir,
        config=config,
        epochs=2,
        batch_size=2,
        latent_frames_slice=16,
        device="cpu",
    )

    assert os.path.exists(ckpt_path)
    assert os.path.exists(os.path.join(output_dir, "hifigan.pth"))

    # Reload checkpoint
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    assert "hifigan_decoder" in ckpt
    assert "generator" in ckpt
    assert "optim_g" in ckpt
    assert "optim_d" in ckpt
    assert ckpt["epoch"] == 2

    # Verify reloading into fresh HifiDecoder
    fresh_decoder = build_hifigan_decoder()
    missing, unexpected = fresh_decoder.load_state_dict(ckpt["hifigan_decoder"], strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0


def test_extract_latents_standalone(temp_workspace):
    """Verify that extract_latents processes audio directory and exports valid .pt files."""
    audio_dir = os.path.join(temp_workspace, "audio")
    latents_dir = os.path.join(temp_workspace, "latents")
    os.makedirs(audio_dir, exist_ok=True)

    # Create dummy audio file
    wav_path = os.path.join(audio_dir, "test.wav")
    wav = torch.sin(torch.linspace(0, 440, 24000)).unsqueeze(0)
    torchaudio.save(wav_path, wav, 24000)

    extract_latents(audio_dir=audio_dir, output_dir=latents_dir, checkpoint_dir="", device="cpu")

    out_file = os.path.join(latents_dir, "test.pt")
    assert os.path.exists(out_file)
    data = safe_torch_load(out_file, map_location="cpu")
    assert "gpt_latents" in data
    assert "speaker_embedding" in data
    assert data["gpt_latents"].shape[-1] == 1024
    assert data["speaker_embedding"].shape == (1, 512, 1)


def test_stage3_silence_rejection_and_amplitude_stability(temp_workspace):
    """Verify silence rejection in dataset and that output amplitude remains stable after training."""
    audio_dir = os.path.join(temp_workspace, "audio")
    latents_dir = os.path.join(temp_workspace, "latents")
    output_dir = os.path.join(temp_workspace, "checkpoints")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    # Audio with 1s of silence followed by 1s of speech
    silence = torch.zeros(1, 24000)
    speech = torch.sin(torch.linspace(0, 440, 24000)).unsqueeze(0) * 0.2
    mixed_audio = torch.cat([silence, speech], dim=-1)

    for i in range(2):
        wav_path = os.path.join(audio_dir, f"sample_{i}.wav")
        torchaudio.save(wav_path, mixed_audio, 24000)

        lat_path = os.path.join(latents_dir, f"sample_{i}.pt")
        torch.save(
            {
                "gpt_latents": torch.randn(1, 32, 1024),
                "speaker_embedding": torch.randn(1, 512, 1),
            },
            lat_path,
        )

    # Test dataset silence rejection
    dataset = LatentHiFiGANDataset(audio_dir, latents_dir, latent_frames_slice=16)
    waveform_slice, latents_slice, _ = dataset[0]
    slice_rms = torch.sqrt(torch.mean(waveform_slice ** 2)).item()
    # Verified: selected slice is active speech, not leading silence
    assert slice_rms >= 0.015

    # Run 1 epoch of training with gradient clipping and fine-tuning LR
    ckpt_path = train_hifigan(
        audio_dir=audio_dir,
        latents_dir=latents_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        latent_frames_slice=16,
        learning_rate=2e-5,
        device="cpu",
    )

    # Check that trained decoder output amplitude is healthy and hasn't collapsed
    trained_ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    decoder = build_hifigan_decoder()
    decoder.load_state_dict(trained_ckpt["hifigan_decoder"])
    decoder.eval()

    test_latents = torch.randn(1, 16, 1024)
    test_spk = torch.randn(1, 512, 1)
    with torch.no_grad():
        out_wav = decoder(test_latents, g=test_spk)
        out_rms = torch.sqrt(torch.mean(out_wav ** 2)).item()

    # RMS must remain robust (e.g. > 0.01), not collapse to 0.000008
    assert out_rms > 0.01


def test_stage3_inference_synthesize_audio(temp_workspace):
    """Verify that synthesize_audio runs end-to-end and outputs valid WAV file."""
    import soundfile as sf
    from xtts_trainer.stage3_hifigan.inference import synthesize_audio

    ckpt_path = os.path.join(temp_workspace, "hifigan.pth")
    latent_path = os.path.join(temp_workspace, "latent.pt")
    out_wav = os.path.join(temp_workspace, "out.wav")

    decoder = build_hifigan_decoder()
    torch.save({"hifigan_decoder": decoder.state_dict()}, ckpt_path)
    torch.save(
        {
            "gpt_latents": torch.randn(1, 16, 1024),
            "speaker_embedding": torch.randn(1, 512, 1),
        },
        latent_path,
    )

    synthesize_audio(
        checkpoint_path=ckpt_path,
        latent_path=latent_path,
        output_path=out_wav,
        device="cpu",
    )
    assert os.path.exists(out_wav)
    data, sr = sf.read(out_wav)
    assert sr == 24000
    assert len(data) > 0


