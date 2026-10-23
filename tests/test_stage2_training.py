"""Tests for Stage 2: XTTS v2 GPT Backbone Training."""

import os
import shutil
import tempfile
import json
import torch
import torchaudio
import pytest

from xtts_trainer.stage2_gpt.vocab_extender import extend_vocab, resize_gpt_embeddings
from xtts_trainer.stage2_gpt.train import (
    GPTTextAudioDataset,
    collate_gpt_batch,
    build_gpt_model,
    train_gpt,
)
from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def create_dummy_vocab_json(path: str) -> dict:
    vocab = {
        "[UNK]": 0,
        "[STOP]": 1,
        "[START]": 2,
        "[SPACE]": 3,
        "a": 4,
        "b": 5,
        "c": 6,
        "d": 7,
        "e": 8,
        "f": 9,
        "g": 10,
        "h": 11,
        "i": 12,
        "j": 13,
        "k": 14,
        "l": 15,
        "m": 16,
        "n": 17,
        "o": 18,
        "p": 19,
        "r": 20,
        "s": 21,
        "t": 22,
        "u": 23,
        "v": 24,
        "w": 25,
        "y": 26,
    }
    data = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        },
        "added_tokens": [
            {"id": 0, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "[STOP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "[START]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 3, "content": "[SPACE]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 4, "content": "[en]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return vocab


def test_vocab_extender_resizes_embedding_and_head():
    """Verify that extending vocab correctly updates both text_embedding and text_head."""
    gpt = GPT(
        layers=2,
        model_dim=32,
        start_text_token=2,
        stop_text_token=1,
        heads=2,
        max_text_tokens=50,
        max_mel_tokens=50,
        max_prompt_tokens=20,
        number_text_tokens=27,
        num_audio_tokens=8194,
        start_audio_token=8192,
        stop_audio_token=8193,
        use_perceiver_resampler=False,
        code_stride_len=1024,
    )

    old_vocab_size = gpt.text_embedding.weight.shape[0]
    assert old_vocab_size == 27
    assert gpt.text_head.weight.shape[0] == 27

    # Save original start token row
    orig_start_emb = gpt.text_embedding.weight[-1, :].clone()
    orig_start_head = gpt.text_head.weight[-1, :].clone()

    new_vocab_size = 35
    resize_gpt_embeddings(gpt, new_vocab_size)

    assert gpt.text_embedding.weight.shape[0] == new_vocab_size
    assert gpt.text_head.weight.shape[0] == new_vocab_size
    assert gpt.text_head.bias.shape[0] == new_vocab_size
    assert gpt.number_text_tokens == new_vocab_size

    # Verify start token row was preserved at index -1
    assert torch.allclose(gpt.text_embedding.weight[-1, :], orig_start_emb)
    assert torch.allclose(gpt.text_head.weight[-1, :], orig_start_head)


def test_gpt_forward_backward_gradient_updates(temp_workspace):
    """Verify that Stage 2 computes valid nonzero CE loss and updates parameters."""
    gpt = GPT(
        layers=2,
        model_dim=64,
        start_text_token=2,
        stop_text_token=1,
        heads=2,
        max_text_tokens=50,
        max_mel_tokens=50,
        max_prompt_tokens=20,
        number_text_tokens=30,
        num_audio_tokens=8194,
        start_audio_token=8192,
        stop_audio_token=8193,
        use_perceiver_resampler=False,
        code_stride_len=1024,
    )

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    orig_param = gpt.mel_embedding.weight.data.clone()

    # Create dummy batch
    batch = {
        "text_inputs": torch.randint(3, 25, (2, 6)),
        "text_lengths": torch.tensor([6, 4]),
        "audio_codes": torch.randint(0, 8000, (2, 16)),
        "wav_lengths": torch.tensor([16 * 1024, 12 * 1024]),
        "cond_mels": torch.randn(2, 1, 80, 20),
    }

    optimizer.zero_grad()
    loss_text, loss_mel, _ = gpt(
        text_inputs=batch["text_inputs"],
        text_lengths=batch["text_lengths"],
        audio_codes=batch["audio_codes"],
        wav_lengths=batch["wav_lengths"],
        cond_mels=batch["cond_mels"],
    )

    assert loss_text.item() > 0.0
    assert loss_mel.item() > 0.0
    total_loss = loss_text * 0.01 + loss_mel * 1.0

    # Ensure loss is not a constant 0.5
    assert abs(total_loss.item() - 0.5) > 0.1

    total_loss.backward()

    # Check gradients
    assert gpt.mel_embedding.weight.grad is not None
    assert gpt.mel_embedding.weight.grad.norm().item() > 0.0

    optimizer.step()

    # Check weights actually updated
    updated_param = gpt.mel_embedding.weight.data
    assert not torch.allclose(orig_param, updated_param)


def test_stage2_dataset_and_training_loop(temp_workspace):
    """End-to-end integration test: dataset creation, batching, and training execution."""
    audio_dir = os.path.join(temp_workspace, "audio")
    output_dir = os.path.join(temp_workspace, "checkpoints")
    metadata_path = os.path.join(temp_workspace, "metadata.csv")
    vocab_path = os.path.join(temp_workspace, "vocab.json")
    os.makedirs(audio_dir, exist_ok=True)

    create_dummy_vocab_json(vocab_path)

    # Create 3 synthetic audio files (1s @ 22050)
    metadata_lines = []
    for i in range(3):
        wav_filename = f"sample_{i}.wav"
        wav_path = os.path.join(audio_dir, wav_filename)
        wav = torch.sin(torch.linspace(0, 200 * (i + 1), 22050)).unsqueeze(0)
        torchaudio.save(wav_path, wav, 22050)
        metadata_lines.append(f"{wav_filename}|hello world from xtts|en")

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    # Run train_gpt for 2 epochs with mini architecture for speed
    ckpt_path = train_gpt(
        audio_dir=audio_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
        vocab_path=vocab_path,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        device="cpu",
        gpt_layers=2,
        gpt_model_dim=64,
        gpt_heads=2,
    )

    assert os.path.exists(ckpt_path)
    assert os.path.exists(os.path.join(output_dir, "model.pth"))
    assert os.path.exists(os.path.join(output_dir, "vocab.json"))

    # Verify reloaded checkpoint
    saved_data = torch.load(ckpt_path, map_location="cpu")
    assert "model" in saved_data
    assert "gpt" in saved_data
    assert "optimizer" in saved_data
    assert saved_data["epoch"] == 2
    assert any(k.startswith("gpt.") for k in saved_data["model"].keys())

    # Verify that the saved model.pth can be loaded into native XTTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.model_args.gpt_layers = 2
    config.model_args.gpt_n_model_channels = 64
    config.model_args.gpt_n_heads = 2
    config.model_args.gpt_num_audio_tokens = 1026
    config.model_args.gpt_start_audio_token = 1024
    config.model_args.gpt_stop_audio_token = 1025
    config.model_args.tokenizer_file = vocab_path

    xtts_model = Xtts(config)
    xtts_model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    xtts_model.init_models()
    assert xtts_model.gpt is not None
    missing, unexpected = xtts_model.gpt.load_state_dict(saved_data["gpt"], strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0
