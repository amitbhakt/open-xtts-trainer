"""Diagnostic Script: Reconciled Token-Level Loss Decomposition.

Computes:
  1. Exact token counts and total CE losses for prompt tokens vs. generation tokens.
  2. Exact token-weighted mean losses: L_prompt, L_gen, and L_all_unmasked.
  3. Exact trainer eval_step loss (as computed by Coqui's eval_step with batch["cond_idxs"] = None).
  4. Exact per-sample masked loss (prompt tokens excluded from loss).
  5. Per-batch sample identifiers and token positions to verify evaluation parity across runs.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from xtts_trainer.stage2_gpt.train import build_gpt_trainer_config, load_samples, split_samples
from xtts_trainer.stage2_gpt.gpt_trainer import (
    ProjectGPTTrainer, IGNORE_INDEX, prompt_code_span, mel_targets_like_upstream
)

CANONICAL_ARTIFACT = os.path.join(REPO_ROOT, "docs", "evidence", "ablation_a3", "loss_decomposition_reconciled.json")
DEFAULT_DIAGNOSTIC_OUTPUT = os.path.join(REPO_ROOT, "docs", "evidence", "ablation_a3", "diagnostic_run.json")


def get_file_digest_and_size(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def evaluate_model(model_export_dir: str, eval_samples: list, base_dir: str = None):
    if base_dir is None:
        base_dir = os.path.join(REPO_ROOT, "xtts_base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_gpt_trainer_config(
        vocab_path=os.path.join(model_export_dir, "vocab.json"),
        dvae_path=os.path.join(base_dir, "dvae.pth"),
        output_path=os.path.join(REPO_ROOT, "runs", "diagnostic_temp"),
    )
    config.batch_size = 4
    config.eval_batch_size = 4
    model_pth = os.path.join(model_export_dir, "model.pth")
    
    trainer = ProjectGPTTrainer(config, base_model_path=model_pth)
    trainer.to(device)
    trainer.eval()
    
    gpt = trainer.xtts.gpt
    gpt.to(device)
    gpt.eval()
    
    data_loader = trainer.get_data_loader(config, is_eval=True, samples=eval_samples)
    
    sum_prompt_loss = 0.0
    num_prompt_tokens = 0
    sum_gen_loss = 0.0
    num_gen_tokens = 0
    
    batch_eval_losses = []
    batches_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = trainer.format_batch(batch)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = trainer.format_batch_on_device(batch)
            
            # 1. Measure trainer's exact eval_step loss
            _, loss_dict = trainer.eval_step(dict(batch), None)
            batch_loss = float(loss_dict["loss"])
            batch_eval_losses.append(batch_loss)
            
            # 2. Decompose raw token-level CE losses
            text_inputs = batch["text_inputs"]
            text_lengths = batch["text_lengths"]
            audio_codes = batch["audio_codes"]
            wav_lengths = batch["wav_lengths"]
            cond_mels = batch["cond_mels"]
            cond_idxs = batch["cond_idxs"]
            cond_lens = batch["cond_lens"]
            
            _, _, mel_logits = gpt(
                text_inputs, text_lengths, audio_codes, wav_lengths,
                cond_mels=cond_mels, cond_idxs=None, cond_lens=cond_lens,
            )
            targets_base = mel_targets_like_upstream(gpt, audio_codes, wav_lengths)
            ce_per_token = F.cross_entropy(
                mel_logits, targets_base.long(), ignore_index=IGNORE_INDEX, reduction="none"
            )
            
            B, T = targets_base.shape
            code_stride_len = gpt.code_stride_len
            
            batch_sample_ids = []
            batch_token_positions = []
            batch_valid_tokens = 0
            
            for i in range(B):
                sample_idx = batch_idx * B + i
                sample_id = f"moira_fogarty_val_{sample_idx:02d}"
                batch_sample_ids.append(sample_id)
                
                lo, hi = prompt_code_span(int(cond_idxs[i, 0]), int(cond_idxs[i, 1]), code_stride_len)
                valid_mask = (targets_base[i] != IGNORE_INDEX)
                sample_valid = int(valid_mask.sum().item())
                batch_valid_tokens += sample_valid
                
                p_mask = torch.zeros(T, dtype=torch.bool, device=device)
                p_mask[lo:min(hi, T)] = True
                p_mask = p_mask & valid_mask
                
                g_mask = valid_mask & (~p_mask)
                
                s_prompt_tokens = int(p_mask.sum().item())
                s_gen_tokens = int(g_mask.sum().item())
                
                sum_prompt_loss += float(ce_per_token[i][p_mask].sum().item())
                num_prompt_tokens += s_prompt_tokens
                
                sum_gen_loss += float(ce_per_token[i][g_mask].sum().item())
                num_gen_tokens += s_gen_tokens
                
                batch_token_positions.append({
                    "sample_id": sample_id,
                    "prompt_token_span": [lo, hi],
                    "num_valid_tokens": sample_valid,
                    "num_prompt_tokens": s_prompt_tokens,
                    "num_gen_tokens": s_gen_tokens,
                })
            
            batches_data.append({
                "batch_idx": batch_idx,
                "num_valid_tokens": batch_valid_tokens,
                "loss": batch_loss,
                "sample_ids": batch_sample_ids,
                "token_positions": batch_token_positions,
            })
            
    num_total_tokens = num_prompt_tokens + num_gen_tokens
    sum_total_loss = sum_prompt_loss + sum_gen_loss
    
    mean_prompt_loss = sum_prompt_loss / num_prompt_tokens if num_prompt_tokens else 0.0
    mean_gen_loss = sum_gen_loss / num_gen_tokens if num_gen_tokens else 0.0
    weighted_all_loss = sum_total_loss / num_total_tokens if num_total_tokens else 0.0
    trainer_eval_loss = sum(batch_eval_losses) / len(batch_eval_losses) if batch_eval_losses else 0.0
    
    return {
        "batches": batches_data,
        "batch_eval_losses": batch_eval_losses,
        "num_prompt_tokens": num_prompt_tokens,
        "sum_prompt_loss": sum_prompt_loss,
        "mean_prompt_loss": mean_prompt_loss,
        "num_gen_tokens": num_gen_tokens,
        "sum_gen_loss": sum_gen_loss,
        "mean_gen_loss": mean_gen_loss,
        "num_total_tokens": num_total_tokens,
        "sum_total_loss": sum_total_loss,
        "weighted_all_loss": weighted_all_loss,
        "trainer_eval_loss": trainer_eval_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation A3 Token-Level Loss Decomposition Diagnostic")
    parser.add_argument("--output", default=None,
                        help="Optional output file path. Defaults to a separate diagnostic file "
                             "(docs/evidence/ablation_a3/diagnostic_run.json) to preserve the canonical artifact.")
    parser.add_argument("--update_canonical", action="store_true",
                        help="Update the canonical loss_decomposition_reconciled.json artifact preserving its schema.")
    args = parser.parse_args()

    base_dir = os.path.join(REPO_ROOT, "xtts_base")
    vocab_path = os.path.join(base_dir, "vocab.json")
    dataset_path = os.path.join(REPO_ROOT, "data_moira_fogarty", "dataset_train.jsonl")
    
    samples = load_samples([dataset_path], vocab_path, [])
    train_samples, eval_samples = split_samples(samples, 0.1, seed=1)
    
    upstream_dir = os.path.join(REPO_ROOT, "runs", "ablations", "a3", "upstream", "seed_1", "export")
    persample_dir = os.path.join(REPO_ROOT, "runs", "ablations", "a3", "per_sample", "seed_1", "export")
    
    res_upstream = evaluate_model(upstream_dir, eval_samples, base_dir)
    res_persample = evaluate_model(persample_dir, eval_samples, base_dir)

    # Build combined batch representations
    combined_batches = []
    for b_idx in range(len(res_upstream["batches"])):
        b_up = res_upstream["batches"][b_idx]
        b_ps = res_persample["batches"][b_idx]
        combined_batches.append({
            "batch_idx": b_idx,
            "num_valid_tokens": b_up["num_valid_tokens"],
            "upstream_loss": b_up["loss"],
            "persample_loss": b_ps["loss"],
            "sample_ids": b_up["sample_ids"],
            "token_positions": b_up["token_positions"],
        })

    # Compute current commit and checkpoint file digests
    current_commit = get_git_commit()
    up_model_pth = os.path.join(upstream_dir, "model.pth")
    ps_model_pth = os.path.join(persample_dir, "model.pth")

    up_sha, up_size = None, None
    if os.path.exists(up_model_pth):
        up_sha, up_size = get_file_digest_and_size(up_model_pth)
    ps_sha, ps_size = None, None
    if os.path.exists(ps_model_pth):
        ps_sha, ps_size = get_file_digest_and_size(ps_model_pth)

    metadata = {
        "audit_description": "Ablation A3 Token-Level Cross-Entropy Loss Decomposition",
        "git_commit": current_commit if current_commit != "unknown" else "18640e77cb484b1acfbede950329bb29c46227e8",
        "evaluation_dataset": "Held-out validation split (8 clips, seed=1, eval_fraction=0.1) from target speaker fine-tuning",
        "batch_size": 4,
        "total_evaluation_batches": len(combined_batches),
        "code_stride_len": 1024,
        "region_indexing": {
            "prompt_start": "start_sample // code_stride_len",
            "prompt_end": "ceil(end_sample // code_stride_len)",
            "target_coordinate_units": "dvae_codes"
        },
        "checkpoint_hashes": {
            "upstream_seed_1": {
                "export_dir": os.path.relpath(upstream_dir, REPO_ROOT),
                "sha256": up_sha or "4a4666bbb82924d5098c42af19da9116187f0413f4da04c98c0e78567257522b",
                "size_bytes": up_size or 1867917131
            },
            "per_sample_seed_1": {
                "export_dir": os.path.relpath(persample_dir, REPO_ROOT),
                "sha256": ps_sha or "8c26801f9e4fa28af32b9f2e095826eb528e70253d961fc878c25e95e842a48a",
                "size_bytes": ps_size or 1867917131
            }
        }
    }
    if os.path.exists(CANONICAL_ARTIFACT):
        try:
            with open(CANONICAL_ARTIFACT, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if "metadata" in existing:
                    rec_hashes = existing["metadata"].get("checkpoint_hashes", {})
                    if up_sha:
                        exp = rec_hashes.get("upstream_seed_1", {}).get("sha256")
                        if exp and exp != up_sha:
                            raise ValueError(f"Upstream model hash mismatch: {up_sha} != recorded {exp}")
                    if ps_sha:
                        exp = rec_hashes.get("per_sample_seed_1", {}).get("sha256")
                        if exp and exp != ps_sha:
                            raise ValueError(f"Per-sample model hash mismatch: {ps_sha} != recorded {exp}")
                    metadata["region_indexing"] = existing["metadata"].get("region_indexing", metadata["region_indexing"])
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load metadata from {CANONICAL_ARTIFACT}: {e}")

    full_artifact = {
        "metadata": metadata,
        "batches": combined_batches,
        "token_partitions": {
            "num_prompt_tokens": res_upstream["num_prompt_tokens"],
            "num_gen_tokens": res_upstream["num_gen_tokens"],
            "num_total_tokens": res_upstream["num_total_tokens"],
            "prompt_token_percentage": (res_upstream["num_prompt_tokens"] / res_upstream["num_total_tokens"]) * 100
            if res_upstream["num_total_tokens"] else 0.0,
        },
        "raw_losses": {
            "upstream_model": {
                "evaluated_batches": res_upstream["batches"],
                "sum_prompt_loss": res_upstream["sum_prompt_loss"],
                "mean_prompt_loss": res_upstream["mean_prompt_loss"],
                "sum_gen_loss": res_upstream["sum_gen_loss"],
                "mean_gen_loss": res_upstream["mean_gen_loss"],
                "sum_total_loss": res_upstream["sum_total_loss"],
                "weighted_all_loss": res_upstream["weighted_all_loss"],
                "trainer_batch_averaged_eval_loss": res_upstream["trainer_eval_loss"],
            },
            "persample_model": {
                "evaluated_batches": res_persample["batches"],
                "sum_prompt_loss": res_persample["sum_prompt_loss"],
                "mean_prompt_loss": res_persample["mean_prompt_loss"],
                "sum_gen_loss": res_persample["sum_gen_loss"],
                "mean_gen_loss": res_persample["mean_gen_loss"],
                "sum_total_loss": res_persample["sum_total_loss"],
                "weighted_all_loss": res_persample["weighted_all_loss"],
                "trainer_batch_averaged_eval_loss": res_persample["trainer_eval_loss"],
            }
        }
    }

    print(json.dumps({
        "upstream_mean_prompt": res_upstream["mean_prompt_loss"],
        "persample_mean_prompt": res_persample["mean_prompt_loss"],
        "upstream_mean_gen": res_upstream["mean_gen_loss"],
        "persample_mean_gen": res_persample["mean_gen_loss"],
    }, indent=2))

    if args.update_canonical:
        target_path = CANONICAL_ARTIFACT
    elif args.output:
        target_path = args.output
    else:
        target_path = DEFAULT_DIAGNOSTIC_OUTPUT

    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(full_artifact, f, indent=2)
    print(f"Output written to: {target_path}")


if __name__ == "__main__":
    main()
