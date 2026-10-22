"""Vocabulary Extender for XTTS v2 GPT Backbone.

Expands XTTS tokenizer and token embeddings to support new alphabets, scripts,
and languages (e.g., Arabic, Devanagari, Cyrillic, Greek, Thaana) without breaking
existing pre-trained weights.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from typing import Dict, List, Set


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
) -> Dict[str, int]:
    """Adds missing characters from target text to base vocab.json without changing old token IDs.

    Supports both Hugging Face Tokenizer JSON format (official XTTS v2) and flat dicts.
    """
    if not os.path.exists(base_vocab_path):
        raise FileNotFoundError(f"Base vocab not found at: {base_vocab_path}")

    with open(base_vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect if it is a HuggingFace tokenizer JSON or flat dictionary
    is_hf_tokenizer = isinstance(data, dict) and "model" in data and "vocab" in data["model"]
    vocab: Dict[str, int] = data["model"]["vocab"] if is_hf_tokenizer else data

    print(f"Loaded base vocabulary ({'HuggingFace Tokenizer' if is_hf_tokenizer else 'Flat Dict'}) with {len(vocab)} tokens.")

    corpus_chars = extract_characters_from_corpus(target_text_path)
    print(f"Found {len(corpus_chars)} unique characters in target corpus.")

    next_id = max(vocab.values()) + 1
    added_tokens = 0

    for char in sorted(corpus_chars):
        if char not in vocab and not char.isspace():
            vocab[char] = next_id
            if is_hf_tokenizer and "added_tokens" in data and isinstance(data["added_tokens"], list):
                data["added_tokens"].append({
                    "id": next_id,
                    "special": False,
                    "content": char,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                })
            next_id += 1
            added_tokens += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_vocab_path)), exist_ok=True)
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Successfully added {added_tokens} new tokens.")
    print(f"New total vocabulary size: {len(vocab)}")
    print(f"Saved extended vocabulary to: {output_vocab_path}")
    return vocab


def resize_gpt_embeddings(
    model: nn.Module,
    new_vocab_size: int,
    embedding_layer_name: str = "text_embedding",
):
    """Resizes the text embedding layer of the GPT model to accommodate the new vocabulary size."""
    old_emb: nn.Embedding = getattr(model, embedding_layer_name)
    old_vocab_size, embedding_dim = old_emb.weight.shape

    if new_vocab_size <= old_vocab_size:
        print(f"New vocab size ({new_vocab_size}) <= old size ({old_vocab_size}). No expansion needed.")
        return

    print(f"Expanding {embedding_layer_name} from {old_vocab_size} to {new_vocab_size}...")
    new_emb = nn.Embedding(new_vocab_size, embedding_dim)

    with torch.no_grad():
        # Copy old weights
        new_emb.weight[:old_vocab_size] = old_emb.weight
        # Initialize new weights with mean of old weights
        mean_weight = old_emb.weight.mean(dim=0, keepdim=True)
        new_emb.weight[old_vocab_size:] = mean_weight.repeat(new_vocab_size - old_vocab_size, 1)

    setattr(model, embedding_layer_name, new_emb)
    print("Embedding expansion complete.")


def main():
    parser = argparse.ArgumentParser(description="Extend XTTS v2 vocabulary for new scripts.")
    parser.add_argument("--base_vocab", type=str, required=True, help="Path to original vocab.json")
    parser.add_argument("--target_text", type=str, required=True, help="Text corpus containing target language script")
    parser.add_argument("--output_vocab", type=str, required=True, help="Path to save extended vocab.json")
    args = parser.parse_args()

    extend_vocab(args.base_vocab, args.target_text, args.output_vocab)


if __name__ == "__main__":
    main()
