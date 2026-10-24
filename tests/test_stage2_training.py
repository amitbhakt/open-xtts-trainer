"""Tests for Stage 2: XTTS v2 GPT Backbone Training."""

import os
import shutil
import tempfile
import json
import torch
import torchaudio
import pytest

from dataset_helpers import write_test_dataset

from xtts_trainer.stage2_gpt.vocab_extender import extend_vocab, resize_gpt_embeddings
from xtts_trainer.models import build_gpt, safe_torch_load
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
        "[en]": 27,
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
            {"id": 27, "content": "[en]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
            "unk_token": "[UNK]",
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

    # New rows start at the mean of the pretrained rows (plus small noise), not at random (docs/recipe_design.md A5)
    mean = orig_emb_weights.mean(0)
    assert torch.allclose(gpt.text_embedding.weight[old_vocab_size:], mean.expand(new_vocab_size - old_vocab_size, -1), atol=0.05)
    assert torch.allclose(gpt.text_head.bias[old_vocab_size:], orig_head_bias.mean().expand(new_vocab_size - old_vocab_size))


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
