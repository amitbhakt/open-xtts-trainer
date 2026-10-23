"""Tests for Stage 1: XTTS v2 Discrete VAE (DVAE) Training."""

import os
import shutil
import tempfile
import torch
import torchaudio
import pytest

from xtts_trainer.config import DVAEConfig
from xtts_trainer.stage1_dvae.dataset import MelDataset
from xtts_trainer.stage1_dvae.train import build_dvae_model, train_dvae
from TTS.tts.layers.xtts.dvae import DiscreteVAE


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_dvae_architecture_and_forward_backward():
    """Verify that native DiscreteVAE forward pass computes valid losses and non-zero gradients."""
    dvae = build_dvae_model(
        channels=80,
        num_tokens=512,
        codebook_dim=64,
        hidden_dim=64,
    )
    dvae.train()

    mels = torch.randn(2, 80, 32)
    recon_loss, commitment_loss, recon_mels = dvae(mels)

    assert recon_loss.mean().item() > 0.0
    assert commitment_loss.mean().item() > 0.0
    total_loss = recon_loss.mean() + commitment_loss.mean()

    orig_param = dvae.codebook.embed.data.clone()

    optimizer = torch.optim.Adam(dvae.parameters(), lr=1e-3)
    optimizer.zero_grad()
    total_loss.backward()

    # Verify gradients
    has_grad = any(p.grad is not None and p.grad.norm().item() > 0.0 for p in dvae.parameters() if p.requires_grad)
    assert has_grad

    optimizer.step()


def test_stage1_dataset_and_training_loop(temp_workspace):
    """End-to-end integration test: 22050 Hz mel extraction, training loop, and checkpoint persistence."""
    audio_dir = os.path.join(temp_workspace, "audio")
    output_dir = os.path.join(temp_workspace, "checkpoints")
    os.makedirs(audio_dir, exist_ok=True)

    # Create 3 synthetic audio files (1s @ 22050 Hz)
    for i in range(3):
        wav_path = os.path.join(audio_dir, f"sample_{i}.wav")
        wav = torch.sin(torch.linspace(0, 300 * (i + 1), 22050)).unsqueeze(0)
        torchaudio.save(wav_path, wav, 22050)

    config = DVAEConfig(
        sample_rate=22050,
        num_tokens=256,
        codebook_dim=64,
        hidden_dim=64,
        learning_rate=1e-3,
        batch_size=2,
        epochs=2,
    )

    ckpt_path = train_dvae(
        audio_dir=audio_dir,
        output_dir=output_dir,
        config=config,
        epochs=2,
        batch_size=2,
        device="cpu",
    )

    assert os.path.exists(ckpt_path)
    assert os.path.exists(os.path.join(output_dir, "dvae.pth"))

    # Reload checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "dvae" in ckpt
    assert "optimizer" in ckpt
    assert ckpt["epoch"] == 2

    # Load back into fresh DVAE model
    fresh_dvae = build_dvae_model(channels=80, num_tokens=256, codebook_dim=64, hidden_dim=64)
    missing, unexpected = fresh_dvae.load_state_dict(ckpt["dvae"], strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0
