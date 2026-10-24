"""Vocabulary Extender for XTTS v2 GPT Backbone.

Expands XTTS tokenizer and token embeddings to support new alphabets, scripts,
and languages (e.g., Arabic, Devanagari, Cyrillic, Greek, Thaana) without breaking
existing pre-trained weights.
"""

import argparse
import json
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

    # 1. Try loading via official Tokenizer API
    is_hf_tokenizer = False
    try:
        from tokenizers import Tokenizer, AddedToken
        tokenizer = Tokenizer.from_file(base_vocab_path)
        base_vocab = tokenizer.get_vocab()
        is_hf_tokenizer = True
    except Exception as ex:
        logger.warning(f"Could not load via tokenizers.Tokenizer.from_file ({ex}). Falling back to JSON parsing.")
        with open(base_vocab_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict) and "model" in raw_data and "vocab" in raw_data["model"]:
            base_vocab = raw_data["model"]["vocab"]
        else:
            base_vocab = raw_data

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

    if is_hf_tokenizer:
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
    else:
        # Fallback for flat dictionaries
        final_vocab = dict(base_vocab)
        next_id = max(final_vocab.values()) + 1 if final_vocab else 0
        for char in new_chars:
            final_vocab[char] = next_id
            next_id += 1

        os.makedirs(os.path.dirname(os.path.abspath(output_vocab_path)), exist_ok=True)
        with open(output_vocab_path, "w", encoding="utf-8") as f:
            json.dump(final_vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully added {len(new_chars)} new tokens.")
    logger.info(f"New total vocabulary size: {len(final_vocab)} (max ID: {max(final_vocab.values())})")
    logger.info(f"Saved extended vocabulary to: {output_vocab_path}")
    return final_vocab


def resize_gpt_embeddings(
    model: nn.Module,
    new_vocab_size: int,
    embedding_layer_name: str = "text_embedding",
    seed: int = 42,
):
    """Resizes both the text embedding and text prediction head of the GPT model for extended vocabularies.

    Strictly preserves all pretrained rows 0..old_vocab_size-1 without modification.
    New rows are initialized with the mean of pretrained embeddings plus small Gaussian noise (seed=42).
    """
    if hasattr(model, "gpt") and hasattr(model.gpt, "text_embedding"):
        gpt = model.gpt
    else:
        gpt = model

    if not hasattr(gpt, embedding_layer_name):
        raise AttributeError(f"Model does not contain embedding layer: {embedding_layer_name}")

    old_emb: nn.Embedding = getattr(gpt, embedding_layer_name)
    old_vocab_size, embedding_dim = old_emb.weight.shape

    if new_vocab_size <= old_vocab_size:
        logger.info(f"New vocab size ({new_vocab_size}) <= old size ({old_vocab_size}). No expansion needed.")
        return

    num_new_tokens = new_vocab_size - old_vocab_size
    logger.info(
        f"Expanding {embedding_layer_name} and text_head from {old_vocab_size} to {new_vocab_size} (+{num_new_tokens} tokens)..."
    )

    device = old_emb.weight.device
    dtype = old_emb.weight.dtype
    gen = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        # 1. Resize text_embedding: preserve all original rows 0..old_vocab_size-1
        old_emb_weights = old_emb.weight.data
        mean_emb = old_emb_weights.mean(dim=0, keepdim=True).cpu()
        # Principled initialization: mean + small reproducible noise
        noise = torch.randn(num_new_tokens, embedding_dim, generator=gen) * 0.01
        new_emb_rows = (mean_emb.repeat(num_new_tokens, 1) + noise).to(device=device, dtype=dtype)
        expanded_emb = torch.cat([old_emb_weights, new_emb_rows], dim=0)

        new_embedding = nn.Embedding(new_vocab_size, embedding_dim).to(device=device, dtype=dtype)
        new_embedding.weight.data = expanded_emb
        setattr(gpt, embedding_layer_name, new_embedding)

        # 2. Resize text_head if present
        if hasattr(gpt, "text_head") and isinstance(gpt.text_head, nn.Linear):
            old_head_w = gpt.text_head.weight.data
            mean_head_w = old_head_w.mean(dim=0, keepdim=True).cpu()
            head_noise = torch.randn(num_new_tokens, old_head_w.shape[1], generator=gen) * 0.01
            new_head_w = (mean_head_w.repeat(num_new_tokens, 1) + head_noise).to(device=device, dtype=dtype)
            expanded_head_w = torch.cat([old_head_w, new_head_w], dim=0)

            has_bias = gpt.text_head.bias is not None
            new_head = nn.Linear(old_head_w.shape[1], new_vocab_size, bias=has_bias).to(device=device, dtype=dtype)
            new_head.weight.data = expanded_head_w

            if has_bias:
                old_head_bias = gpt.text_head.bias.data
                new_head_bias = torch.zeros(num_new_tokens, device=device, dtype=dtype)
                expanded_head_bias = torch.cat([old_head_bias, new_head_bias], dim=0)
                new_head.bias.data = expanded_head_bias

            gpt.text_head = new_head

        # 3. Update number_text_tokens metadata
        if hasattr(gpt, "number_text_tokens"):
            gpt.number_text_tokens = new_vocab_size

    logger.info("Embedding and text_head expansion complete.")


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
