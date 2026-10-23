"""Tests for verify_pipeline_proof.py integration proof tool."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchaudio
from scripts.verify_pipeline_proof import verify_stage1_dvae, build_dvae


def test_verify_stage1_dvae(tmp_path):
    """Verify Stage 1 DVAE proof logic with dummy audio and weights."""
    ckpt_path = os.path.join(str(tmp_path), "dvae.pth")
    audio_path = os.path.join(str(tmp_path), "audio.wav")

    dvae = build_dvae()
    torch.save({"model": dvae.state_dict()}, ckpt_path)

    # 1 second of 22050 Hz sine wave
    sine = torch.sin(torch.linspace(0, 440, 22050)).unsqueeze(0)
    torchaudio.save(audio_path, sine, 22050)

    result = verify_stage1_dvae(ckpt_path, audio_path, device="cpu")
    assert result["status"] == "PASS"
    assert result["num_codes"] > 0
    assert result["unique_codes"] > 0
    assert result["recon_l1_error"] >= 0.0
