"""Mini self-contained verification test for open-xtts-trainer.

Generates synthetic audio tones and tests Stage 1 (DVAE), Stage 2 (Vocab Extender),
and Stage 3 (HiFi-GAN) forward-backward loops on CPU in under 15 seconds.
"""

import os
import shutil
import tempfile
import torch
import torchaudio

from xtts_trainer.config import AudioConfig, HiFiGANConfig, DVAEConfig
from xtts_trainer.stage1_dvae.train import train_dvae
from xtts_trainer.stage2_gpt.vocab_extender import extend_vocab
from xtts_trainer.stage3_hifigan.extract_latents import extract_latents
from xtts_trainer.stage3_hifigan.train import train_hifigan


def generate_synthetic_audio(dir_path: str, num_files: int = 4, duration_sec: float = 1.0, sr: int = 24000):
    """Generates short synthetic sine waves for testing."""
    import numpy as np
    from scipy.io import wavfile

    os.makedirs(dir_path, exist_ok=True)
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    for i in range(num_files):
        freq = 220.0 * (i + 1)
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        file_path = os.path.join(dir_path, f"sample_{i+1:03d}.wav")
        wavfile.write(file_path, sr, audio)
    print(f"Generated {num_files} synthetic audio files in: {dir_path}")


def test_stage2_vocab_extender(tmp_dir: str):
    print("\n--- Testing Stage 2: Vocab Extender ---")
    base_vocab_path = os.path.join(tmp_dir, "vocab.json")
    target_corpus_path = os.path.join(tmp_dir, "target.txt")
    out_vocab_path = os.path.join(tmp_dir, "vocab_extended.json")

    # Base vocab with 5 tokens
    import json
    base_vocab = {"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3, "c": 4}
    with open(base_vocab_path, "w") as f:
        json.dump(base_vocab, f)

    # Target text with Arabic / Hindi characters
    with open(target_corpus_path, "w", encoding="utf-8") as f:
        f.write("مرحبا नमस्ते")

    new_vocab = extend_vocab(base_vocab_path, target_corpus_path, out_vocab_path)
    assert len(new_vocab) > len(base_vocab), "Vocab was not extended!"
    print("Stage 2 Test Passed!")


def test_stage1_dvae(tmp_dir: str, audio_dir: str):
    print("\n--- Testing Stage 1: DVAE Training ---")
    dvae_out = os.path.join(tmp_dir, "dvae_ckpt")
    cfg = DVAEConfig()
    cfg.batch_size = 2
    cfg.num_tokens = 512
    cfg.hidden_dim = 64
    cfg.codebook_dim = 64

    # Run for 2 epochs on CPU
    train_dvae(
        audio_dir=audio_dir,
        output_dir=dvae_out,
        config=cfg,
        epochs=2,
        batch_size=2,
        device="cpu",
    )
    assert os.path.exists(os.path.join(dvae_out, "dvae_epoch_2.pth")), "DVAE checkpoint not saved!"
    print("Stage 1 Test Passed!")


def test_stage3_hifigan(tmp_dir: str, audio_dir: str):
    print("\n--- Testing Stage 3: Latent HiFi-GAN Vocoder ---")
    latents_dir = os.path.join(tmp_dir, "latents")
    hifigan_out = os.path.join(tmp_dir, "hifigan_ckpt")

    # 1. Extract latents
    extract_latents(audio_dir, latents_dir, device="cpu")

    # 2. Train HiFi-GAN
    cfg = HiFiGANConfig()
    cfg.batch_size = 2
    cfg.segment_size = 4096  # small segment for fast CPU test

    train_hifigan(
        audio_dir=audio_dir,
        latents_dir=latents_dir,
        output_dir=hifigan_out,
        config=cfg,
        epochs=2,
        batch_size=2,
        device="cpu",
    )
    assert os.path.exists(os.path.join(hifigan_out, "hifigan_epoch_2.pt")), "HiFi-GAN checkpoint not saved!"
    print("Stage 3 Test Passed!")


def main():
    print("==================================================")
    print("Running open-xtts-trainer Sanity Test Suite (CPU)")
    print("==================================================")
    tmp_dir = tempfile.mkdtemp(prefix="open_xtts_test_")
    audio_dir = os.path.join(tmp_dir, "audio")

    try:
        generate_synthetic_audio(audio_dir, num_files=4, duration_sec=0.5)
        test_stage2_vocab_extender(tmp_dir)
        test_stage1_dvae(tmp_dir, audio_dir)
        test_stage3_hifigan(tmp_dir, audio_dir)
        print("\n ALL 3 STAGES PASSED THE SANITY TEST!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
