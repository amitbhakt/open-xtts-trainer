"""Tests for Stage 3: XTTS v2 Latent HiFi-GAN Vocoder Training."""

import os
import shutil
import tempfile
import torch
import torchaudio
import pytest

from xtts_trainer.config import HiFiGANConfig
from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset
from xtts_trainer.stage3_hifigan.models import (
    build_hifigan_decoder,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from xtts_trainer.stage3_hifigan.train import train_hifigan


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
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
