"""Tests for Ablation A3 Evaluation Parity and Arithmetic Audit.

Verifies:
  1. Both upstream and per-sample evaluation runs used identical batches, sample identifiers, and token positions.
  2. Mathematical reconciliation between partition sums, token-weighted means, and trainer batch averages.
  3. Proper indexing and mask exclusion semantics for conditioning prompt tokens.
"""

import json
import math
import os
import pytest
import torch

from xtts_trainer.stage2_gpt.gpt_trainer import (
    prompt_code_span, IGNORE_INDEX
)

EVIDENCE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "evidence", "ablation_a3", "loss_decomposition_reconciled.json"
)

def test_evidence_file_exists():
    assert os.path.exists(EVIDENCE_FILE), f"Missing evidence file: {EVIDENCE_FILE}"

def test_evaluation_parity_and_token_reconciliation():
    with open(EVIDENCE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Parity between runs and validation against canonical batches records
    up_batches = data["raw_losses"]["upstream_model"].get("evaluated_batches")
    ps_batches = data["raw_losses"]["persample_model"].get("evaluated_batches")
    batches = data["batches"]
    assert up_batches is not None, "Missing evaluated_batches in upstream_model"
    assert ps_batches is not None, "Missing evaluated_batches in persample_model"
    assert len(up_batches) == len(ps_batches) == len(batches) == 2, "Expected 2 evaluation batches across all records"

    for b_idx in range(len(up_batches)):
        b_up = up_batches[b_idx]
        b_ps = ps_batches[b_idx]
        b_canonical = batches[b_idx]
        assert b_up["batch_idx"] == b_ps["batch_idx"] == b_canonical["batch_idx"] == b_idx
        # Parity check: identical sample identifiers across both runs and canonical batches record
        assert b_up["sample_ids"] == b_ps["sample_ids"] == b_canonical["sample_ids"], (
            f"Batch {b_idx} sample IDs mismatch between runs or canonical record"
        )
        assert len(b_up["sample_ids"]) == 4, f"Batch {b_idx} expected 4 samples"
        # Parity check: identical token positions and prompt spans across both runs and canonical record
        assert b_up["token_positions"] == b_ps["token_positions"] == b_canonical["token_positions"], (
            f"Batch {b_idx} token positions mismatch between runs or canonical record"
        )

    # 2. Parity between batches and sample-level token consistency
    assert batches[0]["num_valid_tokens"] == 423
    assert batches[1]["num_valid_tokens"] == 387

    # Check sample-level token consistency within batches
    for b_idx, batch in enumerate(batches):
        token_positions = batch["token_positions"]
        batch_valid = sum(s["num_valid_tokens"] for s in token_positions)
        assert batch_valid == batch["num_valid_tokens"], f"Batch {b_idx} token count mismatch with samples"
        for s in token_positions:
            assert s["num_prompt_tokens"] + s["num_gen_tokens"] == s["num_valid_tokens"]
            span = s["prompt_token_span"]
            assert len(span) == 2 and span[0] >= 0 and span[1] >= span[0]
            # Derive clipped span length from valid-token range and compare with num_prompt_tokens
            clipped_prompt_len = max(0, min(span[1], s["num_valid_tokens"]) - max(0, span[0]))
            assert clipped_prompt_len == s["num_prompt_tokens"], (
                f"Sample {s['sample_id']} clipped prompt span length {clipped_prompt_len} "
                f"does not match num_prompt_tokens {s['num_prompt_tokens']}"
            )

    total_valid = sum(b["num_valid_tokens"] for b in batches)
    assert total_valid == 810

    # 3. Token partitions partition the total exactly
    partitions = data["token_partitions"]
    assert partitions["num_prompt_tokens"] == 559
    assert partitions["num_gen_tokens"] == 251
    assert partitions["num_total_tokens"] == 810
    assert partitions["num_prompt_tokens"] + partitions["num_gen_tokens"] == partitions["num_total_tokens"] == total_valid

    # Verify that sum of sample prompt and gen tokens match partitions
    total_prompt = sum(s["num_prompt_tokens"] for b in batches for s in b["token_positions"])
    total_gen = sum(s["num_gen_tokens"] for b in batches for s in b["token_positions"])
    assert total_prompt == 559
    assert total_gen == 251

    # 4. Arithmetic consistency for upstream model
    up = data["raw_losses"]["upstream_model"]
    assert math.isclose(up["sum_prompt_loss"] + up["sum_gen_loss"], up["sum_total_loss"], rel_tol=1e-7)
    assert math.isclose(up["sum_prompt_loss"] / 559, up["mean_prompt_loss"], rel_tol=1e-7)
    assert math.isclose(up["sum_gen_loss"] / 251, up["mean_gen_loss"], rel_tol=1e-7)
    assert math.isclose(up["sum_total_loss"] / 810, up["weighted_all_loss"], rel_tol=1e-7)
    
    # Check unweighted batch mean
    batch_mean_up = (batches[0]["upstream_loss"] + batches[1]["upstream_loss"]) / 2
    assert math.isclose(batch_mean_up, up["trainer_batch_averaged_eval_loss"], rel_tol=1e-5)

    # 5. Arithmetic consistency for per-sample model
    ps = data["raw_losses"]["persample_model"]
    assert math.isclose(ps["sum_prompt_loss"] + ps["sum_gen_loss"], ps["sum_total_loss"], rel_tol=1e-7)
    assert math.isclose(ps["sum_prompt_loss"] / 559, ps["mean_prompt_loss"], rel_tol=1e-7)
    assert math.isclose(ps["sum_gen_loss"] / 251, ps["mean_gen_loss"], rel_tol=1e-7)
    assert math.isclose(ps["sum_total_loss"] / 810, ps["weighted_all_loss"], rel_tol=1e-7)
    
    # Check unweighted batch mean
    batch_mean_ps = (batches[0]["persample_loss"] + batches[1]["persample_loss"]) / 2
    assert math.isclose(batch_mean_ps, ps["trainer_batch_averaged_eval_loss"], rel_tol=1e-5)

    # 6. Core empirical findings
    # Prompt loss: upstream < per_sample
    assert up["mean_prompt_loss"] < ps["mean_prompt_loss"]
    # Generation loss: per_sample < upstream
    assert ps["mean_gen_loss"] < up["mean_gen_loss"]

def test_prompt_code_span_logic():
    stride = 1024
    # Prompt of 3 seconds at 24000 Hz: [0, 72000)
    lo, hi = prompt_code_span(0, 72000, stride)
    assert lo == 0
    assert hi == math.ceil(72000 / 1024) == 71
    assert (hi - lo) == 71

    # Prompt offset at [24000, 96000)
    lo2, hi2 = prompt_code_span(24000, 96000, stride)
    assert lo2 == 24000 // 1024 == 23
    assert hi2 == math.ceil(96000 / 1024) == 94
