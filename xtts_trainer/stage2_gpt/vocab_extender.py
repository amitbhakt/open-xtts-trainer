"""Vocabulary Extender for XTTS v2 GPT Backbone.

Expands XTTS tokenizer and token embeddings to support new alphabets, scripts,
and languages (e.g., Arabic, Devanagari, Cyrillic, Greek, Thaana) without breaking
existing pre-trained weights.
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_characters_from_corpus(file_path: str) -> Set[str]:
    """Extracts unique characters from a target language corpus."""
    unique_chars = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            unique_chars.update(line.strip())
    return unique_chars


def extend_vocab(
    base_vocab_path: str,
    target_text_path: str,
    output_vocab_path: str,
    new_languages: List[str] = None,
) -> Dict[str, int]:
    """Adds missing characters and language tokens from target text to base vocab.json without changing old token IDs.

    Uses the native Hugging Face Tokenizers API to preserve complete schema invariants,
    asserting 100% preservation of all original token-to-ID mappings.
    """
    if not os.path.exists(base_vocab_path):
        raise FileNotFoundError(f"Base vocab not found at: {base_vocab_path}")

    corpus_chars = extract_characters_from_corpus(target_text_path)
    logger.info(f"Found {len(corpus_chars)} unique characters in target corpus.")

    # 1. Load with the tokenizers API (XTTS vocab.json is a Hugging Face tokenizer file; nothing else is supported)
    from tokenizers import AddedToken, Tokenizer

    tokenizer = Tokenizer.from_file(base_vocab_path)
    base_vocab = tokenizer.get_vocab()
    orig_vocab_size = len(base_vocab)
    logger.info(f"Loaded base vocabulary with {orig_vocab_size} tokens.")

    # Find missing characters
    new_chars = [c for c in sorted(corpus_chars) if c not in base_vocab and not c.isspace()]
    if new_languages:
        for lang in new_languages:
            lang_tok = f"[{lang}]"
            if lang_tok not in base_vocab and lang_tok not in new_chars:
                new_chars.append(lang_tok)
    logger.info(f"Identified {len(new_chars)} genuinely new tokens (characters and language tags) to add.")

    added_tokens_list = [
        AddedToken(char, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False)
        for char in new_chars
    ]
    tokenizer.add_tokens(added_tokens_list)
    os.makedirs(os.path.dirname(os.path.abspath(output_vocab_path)), exist_ok=True)
    tokenizer.save(output_vocab_path)

    # Validation: round-trip reload and assert integrity
    reloaded = Tokenizer.from_file(output_vocab_path)
    reloaded_vocab = reloaded.get_vocab()

    # Assert every single original token preserved its exact ID
    for token_str, orig_id in base_vocab.items():
        reloaded_id = reloaded_vocab.get(token_str)
        if reloaded_id != orig_id:
            raise ValueError(
                f"Integrity failure: Token '{token_str}' ID changed from {orig_id} to {reloaded_id}!"
            )

    # Assert every new token was assigned a valid contiguous ID
    for char in new_chars:
        new_id = reloaded.token_to_id(char)
        if new_id is None:
            raise ValueError(f"Integrity failure: New character '{char}' was not found in reloaded tokenizer!")
        if new_id < orig_vocab_size:
            raise ValueError(f"Integrity failure: New character '{char}' received collisive ID {new_id} < {orig_vocab_size}!")

    final_vocab = reloaded_vocab

    logger.info(f"Successfully added {len(new_chars)} new tokens.")
    logger.info(f"New total vocabulary size: {len(final_vocab)} (max ID: {max(final_vocab.values())})")
    logger.info(f"Saved extended vocabulary to: {output_vocab_path}")
    return final_vocab


TEXT_ROW_KEYS = ("text_embedding.weight", "text_head.weight", "text_head.bias")
NEW_ROW_NOISE_STD = 0.01


def extend_text_rows(state: Dict[str, torch.Tensor], new_vocab_size: int, seed: int = 42) -> Dict[str, torch.Tensor]:
    """GPT state dict with text rows added for new tokens (docs/recipe_design.md §3, ablation A5).

    Pretrained rows 0..old-1 are kept unchanged. New embedding and text-head rows start at the mean of the
    pretrained rows plus seeded Gaussian noise; new head biases at the mean bias. Upstream GPTTrainer instead
    uses unit-variance random rows and copies the old *last* row onto the new last row, assuming it is the
    start token; in XTTS-v2 the last row is [hi] (6680) and [START] is 261 (finding F6).
    """
    old = state["text_embedding.weight"].shape[0]
    if new_vocab_size < old:
        raise ValueError(f"vocab shrank from {old} to {new_vocab_size}; old token ids would change meaning")
    n_new = new_vocab_size - old
    if n_new == 0:
        return dict(state)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = dict(state)
    for key in TEXT_ROW_KEYS:
        if key not in state:
            continue
        rows = state[key]
        if rows.shape[0] != old:
            raise ValueError(f"{key} has {rows.shape[0]} rows, text_embedding has {old}")
        mean = rows.float().mean(dim=0, keepdim=True)
        new = mean.repeat(n_new, *([1] * (rows.ndim - 1)))
        if rows.ndim == 2:
            new = new + torch.randn(n_new, rows.shape[1], generator=gen) * NEW_ROW_NOISE_STD
        out[key] = torch.cat([rows, new.to(rows.dtype)], dim=0)
    return out


def resize_gpt_embeddings(model: nn.Module, new_vocab_size: int, seed: int = 42):
    """Resize a GPT module's text embedding and text head in place (extend_text_rows on its own state)."""
    gpt = model.gpt if hasattr(model, "gpt") and hasattr(model.gpt, "text_embedding") else model
    old_vocab_size, dim = gpt.text_embedding.weight.shape
    if new_vocab_size <= old_vocab_size:
        logger.info(f"New vocab size ({new_vocab_size}) <= old size ({old_vocab_size}). No expansion needed.")
        return
    state = {k: v.detach().cpu() for k, v in gpt.state_dict().items() if k in TEXT_ROW_KEYS}
    state = extend_text_rows(state, new_vocab_size, seed)
    device, dtype = gpt.text_embedding.weight.device, gpt.text_embedding.weight.dtype
    with torch.no_grad():
        gpt.text_embedding = nn.Embedding(new_vocab_size, dim).to(device=device, dtype=dtype)
        gpt.text_embedding.weight.copy_(state["text_embedding.weight"])
        head = gpt.text_head
        gpt.text_head = nn.Linear(head.in_features, new_vocab_size, bias=head.bias is not None).to(device=device, dtype=dtype)
        gpt.text_head.weight.copy_(state["text_head.weight"])
        if head.bias is not None:
            gpt.text_head.bias.copy_(state["text_head.bias"])
    gpt.number_text_tokens = new_vocab_size
    logger.info(f"Expanded text_embedding and text_head: {old_vocab_size} -> {new_vocab_size}")


def main():
    parser = argparse.ArgumentParser(description="Extend XTTS v2 vocabulary for new scripts.")
    parser.add_argument("--base_vocab", type=str, required=True, help="Path to original vocab.json")
    parser.add_argument("--target_text", type=str, required=True, help="Text corpus containing target language script")
    parser.add_argument("--output_vocab", type=str, required=True, help="Path to save extended vocab.json")
    parser.add_argument("--languages", type=str, nargs="+", default=None, help="New language codes to register tokens for (e.g. bn el fa)")
    args = parser.parse_args()

    extend_vocab(args.base_vocab, args.target_text, args.output_vocab, new_languages=args.languages)


if __name__ == "__main__":
    main()
