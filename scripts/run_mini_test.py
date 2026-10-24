"""Mini self-contained verification test for open-xtts-trainer.

Generates synthetic audio tones and tests Stage 1 (DVAE), Stage 2 (Vocab Extender),
and Stage 3 (HiFi-GAN) forward-backward loops on CPU in under 15 seconds.
"""

import os
import shutil
import tempfile
import torch
import torchaudio

from xtts_trainer.config import HiFiGANConfig, DVAEConfig
from xtts_trainer.spec import SPEC
from xtts_trainer.stage1_dvae.train import train_dvae
from xtts_trainer.stage2_gpt.vocab_extender import extend_vocab
from xtts_trainer.stage3_hifigan.extract_latents import extract_latents
from xtts_trainer.stage3_hifigan.train import train_hifigan


def generate_synthetic_audio(dir_path: str, num_files: int = 4, duration_sec: float = 1.0, sr: int = SPEC.output_sample_rate):
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

    import soundfile as sf
    from xtts_trainer.data import DATASET_FILENAME, DatasetRecord, write_dataset

    records = []
    for i in range(num_files):
        name = f"sample_{i+1:03d}.wav"
        info = sf.info(os.path.join(dir_path, name))
        records.append(DatasetRecord(audio=name, text="hello world", lang="en", speaker="tone",
                                     duration_s=round(info.frames / info.samplerate, 4), sr=info.samplerate,
                                     transcript_source="human"))
    return write_dataset(os.path.join(dir_path, DATASET_FILENAME), records)


def test_stage2_vocab_extender(tmp_dir: str):
    print("\n--- Testing Stage 2: Vocab Extender ---")
    base_vocab_path = os.path.join(tmp_dir, "vocab.json")
    target_corpus_path = os.path.join(tmp_dir, "target.txt")
    out_vocab_path = os.path.join(tmp_dir, "vocab_extended.json")

    # Base vocab with 5 tokens, in the Hugging Face tokenizer format XTTS uses
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    base_vocab = {"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3, "c": 4}
    Tokenizer(BPE(vocab=base_vocab, merges=[], unk_token="[UNK]")).save(base_vocab_path)

    # Target text with Arabic / Hindi characters
    with open(target_corpus_path, "w", encoding="utf-8") as f:
        f.write("مرحبا नमस्ते")

    new_vocab = extend_vocab(base_vocab_path, target_corpus_path, out_vocab_path)
    assert len(new_vocab) > len(base_vocab), "Vocab was not extended!"
    print("Stage 2 Test Passed!")


def test_stage1_dvae(tmp_dir: str, dataset_path: str):
    print("\n--- Testing Stage 1: DVAE Training ---")
    dvae_out = os.path.join(tmp_dir, "dvae_ckpt")
    cfg = DVAEConfig(batch_size=2, epochs=2, num_tokens=512, hidden_dim=64, codebook_dim=64, val_fraction=0.25,
                     min_active_codes=1, frozen_codebook=True)
    train_dvae(dataset_path=dataset_path, output_dir=dvae_out, config=cfg, device="cpu")
    assert os.path.exists(os.path.join(dvae_out, "dvae.pth")), "DVAE checkpoint not saved!"
    print("Stage 1 Test Passed!")


def test_stage3_hifigan(tmp_dir: str, dataset_path: str):
    print("\n--- Testing Stage 3: Latent HiFi-GAN Vocoder ---")
    latents_dir = os.path.join(tmp_dir, "latents")
    hifigan_out = os.path.join(tmp_dir, "hifigan_ckpt")

    # 1. Extract latents
    extract_latents(dataset_path, latents_dir, device="cpu", allow_synthetic_fallback=True)

    # 2. Train HiFi-GAN
    cfg = HiFiGANConfig(batch_size=2, epochs=2, segment_frames=16, val_fraction=0.25, disc_warmup_steps=1,
                        mixed_precision=False)
    train_hifigan(dataset_path=dataset_path, latents_dir=latents_dir, output_dir=hifigan_out, config=cfg,
                  device="cpu", allow_synthetic_latents=True)
    assert os.path.exists(os.path.join(hifigan_out, "hifigan.pth")), "HiFi-GAN checkpoint not saved!"
    print("Stage 3 Test Passed!")


def main():
    print("==================================================")
    print("Running open-xtts-trainer Sanity Test Suite (CPU)")
    print("==================================================")
    tmp_dir = tempfile.mkdtemp(prefix="open_xtts_test_")
    audio_dir = os.path.join(tmp_dir, "audio")

    try:
        dataset_path = generate_synthetic_audio(audio_dir, num_files=4, duration_sec=0.6)
        test_stage2_vocab_extender(tmp_dir)
        test_stage1_dvae(tmp_dir, dataset_path)
        test_stage3_hifigan(tmp_dir, dataset_path)
        print("\n ALL 3 STAGES PASSED THE SANITY TEST!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
