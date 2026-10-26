"""Diagnostic Script: Reconciled Token-Level Loss Decomposition.

Computes:
  1. Exact token counts and total CE losses for prompt tokens vs. generation tokens.
  2. Exact token-weighted mean losses: L_prompt, L_gen, and L_all_unmasked.
  3. Exact trainer eval_step loss (as computed by Coqui's eval_step with batch["cond_idxs"] = None).
  4. Exact per-sample masked loss (prompt tokens excluded from loss).
"""

import json
import os
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

def evaluate_model(model_export_dir: str, eval_samples: list, base_dir: str = "xtts_base"):
    config = build_gpt_trainer_config(
        vocab_path=os.path.join(model_export_dir, "vocab.json"),
        dvae_path=os.path.join(base_dir, "dvae.pth"),
        output_path="runs/diagnostic_temp",
    )
    config.batch_size = 4
    config.eval_batch_size = 4
    model_pth = os.path.join(model_export_dir, "model.pth")
    
    trainer = ProjectGPTTrainer(config, base_model_path=model_pth)
    trainer.eval()
    
    gpt = trainer.xtts.gpt
    gpt.eval()
    device = "cpu"
    
    data_loader = trainer.get_data_loader(config, is_eval=True, samples=eval_samples)
    
    sum_prompt_loss = 0.0
    num_prompt_tokens = 0
    sum_gen_loss = 0.0
    num_gen_tokens = 0
    
    batch_eval_losses = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = trainer.format_batch(batch)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = trainer.format_batch_on_device(batch)
            
            # 1. Measure trainer's exact eval_step loss
            _, loss_dict = trainer.eval_step(dict(batch), None)
            batch_eval_losses.append(float(loss_dict["loss"]))
            
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
            
            for i in range(B):
                lo, hi = prompt_code_span(int(cond_idxs[i, 0]), int(cond_idxs[i, 1]), code_stride_len)
                valid_mask = (targets_base[i] != IGNORE_INDEX)
                
                p_mask = torch.zeros(T, dtype=torch.bool)
                p_mask[lo:min(hi, T)] = True
                p_mask = p_mask & valid_mask
                
                g_mask = valid_mask & (~p_mask)
                
                sum_prompt_loss += float(ce_per_token[i][p_mask].sum().item())
                num_prompt_tokens += int(p_mask.sum().item())
                
                sum_gen_loss += float(ce_per_token[i][g_mask].sum().item())
                num_gen_tokens += int(g_mask.sum().item())
                
    num_total_tokens = num_prompt_tokens + num_gen_tokens
    sum_total_loss = sum_prompt_loss + sum_gen_loss
    
    mean_prompt_loss = sum_prompt_loss / num_prompt_tokens
    mean_gen_loss = sum_gen_loss / num_gen_tokens
    weighted_all_loss = sum_total_loss / num_total_tokens
    trainer_eval_loss = sum(batch_eval_losses) / len(batch_eval_losses)
    
    return {
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
    base_dir = "xtts_base"
    vocab_path = os.path.join(base_dir, "vocab.json")
    dataset_path = "data_moira_fogarty/dataset_train.jsonl"
    
    samples = load_samples([dataset_path], vocab_path, [])
    train_samples, eval_samples = split_samples(samples, 0.1, seed=1)
    
    upstream_dir = "runs/ablations/a3/upstream/seed_1/export"
    persample_dir = "runs/ablations/a3/per_sample/seed_1/export"
    
    res_upstream = evaluate_model(upstream_dir, eval_samples, base_dir)
    res_persample = evaluate_model(persample_dir, eval_samples, base_dir)
    
    summary = {
        "upstream_model": res_upstream,
        "persample_model": res_persample,
    }
    print(json.dumps(summary, indent=2))
    
    out_file = "docs/evidence/ablation_a3/loss_decomposition_reconciled.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
