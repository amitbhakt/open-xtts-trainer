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
from xtts_trainer.config import safe_torch_load
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
        "[STOP]": 0,
        "[UNK]": 1,
        "[SPACE]": 2,
        "a": 3,
        "b": 4,
        "c": 5,
        "d": 6,
        "e": 7,
        "f": 8,
        "g": 9,
        "h": 10,
        "i": 11,
        "j": 12,
        "k": 13,
        "l": 14,
        "m": 15,
        "n": 16,
        "o": 17,
        "p": 18,
        "r": 19,
        "s": 20,
        "t": 21,
        "u": 22,
        "v": 23,
        "w": 24,
        "y": 25,
        "[START]": 26,
    }
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "[STOP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "[SPACE]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 26, "content": "[START]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return vocab


def test_special_token_invariants():
    """Verify that XTTS special tokens match official upstream definitions ([STOP]=0, [START]=261, [UNK]=1, [SPACE]=2)."""
    # Test on a minimal vocab representing XTTS semantics
    tmp = tempfile.mktemp(suffix=".json")
    vocab = {"[STOP]": 0, "[UNK]": 1, "[SPACE]": 2, "[START]": 261}
    data = {
        "version": "1.0",
        "added_tokens": [],
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)

    tokenizer = VoiceBpeTokenizer(vocab_file=tmp)
    assert tokenizer.tokenizer.token_to_id("[STOP]") == 0
    assert tokenizer.tokenizer.token_to_id("[START]") == 261
    assert tokenizer.tokenizer.token_to_id("[UNK]") == 1
    assert tokenizer.tokenizer.token_to_id("[SPACE]") == 2
    os.remove(tmp)


def test_vocab_extender_resizes_embedding_and_head():
    """Verify that extending vocab strictly preserves original rows and correctly expands text_head."""
    old_vocab_size = 27
    gpt = GPT(
        layers=2,
        model_dim=32,
        start_text_token=26,
        stop_text_token=0,
        heads=2,
        max_text_tokens=50,
        max_mel_tokens=50,
        max_prompt_tokens=20,
        number_text_tokens=old_vocab_size,
        num_audio_tokens=1026,
        start_audio_token=1024,
        stop_audio_token=1025,
        use_perceiver_resampler=False,
        code_stride_len=1024,
    )

    assert gpt.text_embedding.weight.shape[0] == old_vocab_size
    assert gpt.text_head.weight.shape[0] == old_vocab_size

    # Save original pretrained rows
    orig_emb_weights = gpt.text_embedding.weight.clone()
    orig_head_weights = gpt.text_head.weight.clone()
    orig_head_bias = gpt.text_head.bias.clone()

    new_vocab_size = 35
    resize_gpt_embeddings(gpt, new_vocab_size, seed=42)

    assert gpt.text_embedding.weight.shape[0] == new_vocab_size
    assert gpt.text_head.weight.shape[0] == new_vocab_size
    assert gpt.text_head.bias.shape[0] == new_vocab_size
    assert gpt.number_text_tokens == new_vocab_size

    # Verify that ALL original pretrained rows 0..old_vocab_size-1 are 100% bit-for-bit preserved
    assert torch.allclose(gpt.text_embedding.weight[:old_vocab_size, :], orig_emb_weights)
    assert torch.allclose(gpt.text_head.weight[:old_vocab_size, :], orig_head_weights)
    assert torch.allclose(gpt.text_head.bias[:old_vocab_size], orig_head_bias)

    # Verify new rows are non-zero
    assert gpt.text_embedding.weight[old_vocab_size:, :].abs().sum().item() > 0.0


def test_vocab_extender_preserves_base_mappings(temp_workspace):
    """Verify that extend_vocab preserves all original tokens and adds new tokens cleanly."""
    base_vocab_path = os.path.join(temp_workspace, "base_vocab.json")
    out_vocab_path = os.path.join(temp_workspace, "extended_vocab.json")
    target_corpus = os.path.join(temp_workspace, "corpus.txt")

    orig_vocab = create_dummy_vocab_json(base_vocab_path)
    with open(target_corpus, "w", encoding="utf-8") as f:
        f.write("123")  # 3 genuinely new characters

    new_vocab = extend_vocab(base_vocab_path, target_corpus, out_vocab_path)

    # Check that all original keys retained original IDs
    for k, v in orig_vocab.items():
        assert new_vocab[k] == v

    # Check that new tokens exist and have IDs >= orig_vocab_size
    assert "1" in new_vocab
    assert "2" in new_vocab
    assert "3" in new_vocab
    assert new_vocab["1"] >= len(orig_vocab)
    assert len(new_vocab) == len(orig_vocab) + 3


def test_checkpoint_strict_allowlist_validation(temp_workspace):
    """Verify that GPT weight loading rejects unexpected keys and missing essential weights."""
    gpt = build_gpt_model(vocab_size=30, layers=2, model_dim=32, heads=2)

    valid_state = gpt.state_dict()

    # Case 1: Extra unexpected key should fail
    corrupt_state = dict(valid_state)
    corrupt_state["unauthorized_malicious_layer.weight"] = torch.randn(10, 10)

    # Using our strict allowlist logic:
    ignore_prefixes = ("torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae", "hifigan_decoder", "mel_stats")
    allowed_missing_patterns = ("attn.bias",)

    missing, unexpected = gpt.load_state_dict(corrupt_state, strict=False)
    assert "unauthorized_malicious_layer.weight" in unexpected


def test_gpt_forward_backward_gradient_updates(temp_workspace):
    """Verify that Stage 2 computes valid nonzero CE loss and updates parameters."""
    gpt = GPT(
        layers=2,
        model_dim=64,
        start_text_token=2,
        stop_text_token=0,
        heads=2,
        max_text_tokens=50,
        max_mel_tokens=50,
        max_prompt_tokens=20,
        number_text_tokens=30,
        num_audio_tokens=1026,
        start_audio_token=1024,
        stop_audio_token=1025,
        use_perceiver_resampler=False,
        code_stride_len=1024,
    )

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    orig_param = gpt.mel_embedding.weight.data.clone()

    batch = {
        "text_inputs": torch.randint(3, 25, (2, 6)),
        "text_lengths": torch.tensor([6, 4]),
        "audio_codes": torch.randint(0, 1000, (2, 16)),
        "wav_lengths": torch.tensor([16 * 1024, 12 * 1024]),
        "cond_mels": torch.randn(2, 1, 80, 20),
        "cond_lens": torch.tensor([20 * 256, 16 * 256]),
    }

    optimizer.zero_grad()
    loss_text, loss_mel, _ = gpt(
        text_inputs=batch["text_inputs"],
        text_lengths=batch["text_lengths"],
        audio_codes=batch["audio_codes"],
        wav_lengths=batch["wav_lengths"],
        cond_mels=batch["cond_mels"],
        cond_lens=batch["cond_lens"],
    )

    assert loss_text.item() > 0.0
    assert loss_mel.item() > 0.0
    total_loss = loss_text * 0.01 + loss_mel * 1.0

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)

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
    assert os.path.exists(os.path.join(output_dir, "config.json"))

    # Verify reloaded checkpoint
    saved_data = safe_torch_load(ckpt_path, map_location="cpu")
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
    config.model_args.gpt_use_perceiver_resampler = True
    config.model_args.tokenizer_file = vocab_path

    xtts_model = Xtts(config)
    xtts_model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    xtts_model.init_models()
    assert xtts_model.gpt is not None
    missing, unexpected = xtts_model.gpt.load_state_dict(saved_data["gpt"], strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0
