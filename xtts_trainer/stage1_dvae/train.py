"""Stage 1: train or fine-tune the Discrete VAE audio tokenizer (docs/recipe_design.md §6, phase 7).

Off by default in the recipe (11.3): every training forward pass moves the codebook (F1), so a fine-tuned
DVAE changes what the GPT's audio tokens mean. What this trainer guarantees:
- validation on held-out clips, with the DVAE in eval mode (eval-mode get_codebook_indices doesn't touch the
  codebook, and eval-mode forward decodes from the quantized codes, which is what the GPT actually sees);
- codebook usage (active codes, perplexity) logged and gated: a collapse fails the run;
- code drift: the fraction of validation frames whose code differs from the base DVAE's;
- an optional frozen-codebook mode (quantizer kept in eval mode: only the encoder and decoder adapt);
- the best checkpoint by validation reconstruction loss, marked requires_gpt_retrain so Stage 2 refuses
  to pair it with the pretrained GPT unless explicitly retraining for it;
- seeded runs, warm-up LR schedule, optional mixed precision and full resume (xtts_trainer.training).
"""

import argparse
import copy
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from xtts_trainer.config import DVAEConfig
from xtts_trainer.data import load_validated_dataset
from xtts_trainer.data.latents import sha256_file
from xtts_trainer.models import (
    REQUIRES_GPT_RETRAIN_KEY, build_dvae, check_spec_hash, dvae_num_tokens, load_dvae_state, resolve_dvae_state,
    safe_torch_load, stamp,
)
from xtts_trainer.stage1_dvae.dataset import MelDataset, collate_mels
from xtts_trainer.training import (
    Precision, decay_per_step, is_improvement, load_training_state, save_training_state, seed_everything,
    split_records, warmup_lambda,
)

logger = logging.getLogger(__name__)

TRAINING_STATE_FILENAME = "training_state.pth"


class CodebookCollapseError(RuntimeError):
    """Too few codebook entries are in use on validation data."""


def dvae_training_loss(dvae, mels: torch.Tensor):
    """The Stage 1 objective: native DVAE reconstruction + commitment loss. Returns (loss, recon, commit)."""
    recon_loss, commitment_loss, _ = dvae(mels)
    recon, commit = recon_loss.mean(), commitment_loss.mean()
    return recon + commit, recon, commit


def codebook_stats(codes: torch.Tensor, num_tokens: int) -> Dict[str, float]:
    """Active codes and perplexity (exp of the code-usage entropy) over a flat tensor of code ids."""
    counts = torch.bincount(codes.flatten().long(), minlength=num_tokens).double()
    p = counts / counts.sum()
    entropy = float(-(p[p > 0] * p[p > 0].log()).sum())
    return {"active_codes": int((counts > 0).sum()), "perplexity": math.exp(entropy)}


@torch.no_grad()
def evaluate_dvae(dvae, val_mels: List[torch.Tensor], base_dvae=None) -> Dict[str, Any]:
    """Eval-mode reconstruction loss, codebook usage, and drift against base_dvae, over whole val clips."""
    was_training = dvae.training
    dvae.eval()
    recon, codes, drift_num, drift_den = [], [], 0, 0
    for mel in val_mels:
        mel = mel.unsqueeze(0)
        r, _, _ = dvae(mel)
        recon.append(float(r.mean()))
        c = dvae.get_codebook_indices(mel)
        codes.append(c.flatten())
        if base_dvae is not None:
            b = base_dvae.get_codebook_indices(mel)
            drift_num += int((b != c).sum())
            drift_den += c.numel()
    dvae.train(was_training)
    stats = codebook_stats(torch.cat(codes), dvae.num_tokens)
    stats["val_recon"] = sum(recon) / len(recon)
    stats["code_drift"] = drift_num / drift_den if base_dvae is not None else None
    return stats


def collapse_floor(config: DVAEConfig, num_tokens: int, base_active: Optional[int]) -> int:
    """Minimum active codes on validation data. [E5] until ablation A2 calibrates it."""
    floor = config.min_active_codes
    if base_active is not None:
        floor = max(floor, math.ceil(config.min_active_fraction_of_base * base_active))
    return min(floor, num_tokens)


def train_dvae(
    dataset_path: str,
    output_dir: str,
    config: Optional[DVAEConfig] = None,
    base_checkpoint: str = "",
    resume: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Train, keeping the best checkpoint by validation loss in output_dir/dvae.pth. Returns its path.

    base_checkpoint: the DVAE to fine-tune (the released dvae.pth); empty trains from scratch.
    resume: continue from output_dir/training_state.pth.
    """
    config = config or DVAEConfig()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    os.makedirs(output_dir, exist_ok=True)
    loader_gen = seed_everything(config.seed)

    records, dataset_dir, _ = load_validated_dataset(dataset_path, None, require_text=False, logger=logger)
    train_records, val_records = split_records(records, config.val_fraction, config.seed)
    train_set = MelDataset([r.audio_path(dataset_dir) for r in train_records], config.max_mel_frames)
    val_mels = [m.to(device) for m in MelDataset([r.audio_path(dataset_dir) for r in val_records],
                                                 config.max_val_mel_frames, random_crop=False)]
    batch_size = max(1, min(config.batch_size, len(train_set)))
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_mels,
                        generator=loader_gen, num_workers=config.num_workers,
                        drop_last=len(train_set) >= 2 * batch_size)
    logger.info("Stage 1: %d train / %d val clips, %d steps per epoch", len(train_set), len(val_mels), len(loader))

    base_state, base_sha = None, None
    num_tokens = config.num_tokens
    if base_checkpoint:
        ckpt = safe_torch_load(base_checkpoint, map_location="cpu")
        check_spec_hash(ckpt, base_checkpoint)
        base_state = resolve_dvae_state(ckpt)
        num_tokens = dvae_num_tokens(base_state)
        base_sha = sha256_file(base_checkpoint)
    dvae = build_dvae(num_tokens=num_tokens, codebook_dim=config.codebook_dim, hidden_dim=config.hidden_dim).to(device)
    base_dvae = None
    if base_state is not None:
        load_dvae_state(dvae, base_state, f"DVAE ({base_checkpoint})")
        base_dvae = copy.deepcopy(dvae).eval()

    optimizer = torch.optim.Adam(dvae.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_lambda(config.lr_warmup_steps, decay_per_step(config.lr_decay_per_epoch, len(loader))))
    precision = Precision(device, config.mixed_precision)
    scaler = precision.scaler()
    state_path = os.path.join(output_dir, TRAINING_STATE_FILENAME)
    run = dict(models={"dvae": dvae}, optimizers={"adam": optimizer}, schedulers={"lr": scheduler},
               scalers={"amp": scaler})

    step, start_epoch, best = 0, 1, None
    if resume:
        restored = load_training_state(state_path, **run)
        step, start_epoch, best = restored["step"], restored["epoch"] + 1, restored["best"]
        loader_gen.set_state(restored["extra"]["loader_generator"])
        reference = restored["extra"]["reference"]
        logger.info("Resumed at epoch %d, step %d (best val %.4f)", start_epoch, step, best)
    else:
        reference = evaluate_dvae(dvae, val_mels, base_dvae)  # the starting model's numbers, to beat
        logger.info("Stage 1 reference before training: %s", reference)
    base_active = reference["active_codes"] if base_dvae is not None else None
    floor = collapse_floor(config, dvae.num_tokens, base_active)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    best_path = os.path.join(output_dir, "dvae.pth")

    def validate(epoch: int) -> None:
        nonlocal best
        stats = evaluate_dvae(dvae, val_mels, base_dvae)
        stats.update(step=step, epoch=epoch, lr=scheduler.get_last_lr()[0])
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        logger.info("Validation step %d: recon %.4f, active codes %d (floor %d), perplexity %.1f, drift %s",
                    step, stats["val_recon"], stats["active_codes"], floor, stats["perplexity"], stats["code_drift"])
        if stats["active_codes"] < floor:
            raise CodebookCollapseError(
                f"only {stats['active_codes']}/{dvae.num_tokens} codes active on validation data (floor {floor}, "
                f"reference {reference['active_codes']}); the codebook collapsed. See {metrics_path}"
            )
        if is_improvement(stats["val_recon"], best):
            best = stats["val_recon"]
            torch.save(stamp({
                "dvae": {k: v.cpu() for k, v in dvae.state_dict().items()},
                REQUIRES_GPT_RETRAIN_KEY: True, "base_dvae_sha256": base_sha,
                "codebook_frozen": config.frozen_codebook, "val": stats, "reference": reference,
            }), best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)

    for epoch in range(start_epoch, config.epochs + 1):
        dvae.train()
        if config.frozen_codebook:
            dvae.codebook.eval()  # no EMA codebook updates (F1); encoder/decoder still learn
        for mels in loader:
            mels = mels.to(device)
            with precision.autocast():
                loss, recon, commit = dvae_training_loss(dvae, mels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite DVAE loss at step {step}: recon {recon.item()}, commit {commit.item()}")
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dvae.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % config.val_every_steps == 0:
                validate(epoch)
        if step % config.val_every_steps:  # end of epoch, unless that step was just validated
            validate(epoch)
        save_training_state(state_path, **run, step=step, epoch=epoch, best=best,
                            extra={"loader_generator": loader_gen.get_state(), "reference": reference})

    logger.info("Stage 1 finished; best checkpoint (val recon %.4f): %s", best, best_path)
    return best_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train or fine-tune the Stage 1 Discrete VAE (DVAE).")
    parser.add_argument("--dataset", required=True, help="dataset.jsonl (transcripts not required)")
    parser.add_argument("--output_dir", default="./checkpoints/dvae")
    parser.add_argument("--base_checkpoint", default="", help="DVAE to fine-tune (e.g. xtts_base/dvae.pth)")
    parser.add_argument("--resume", action="store_true", help="Continue from output_dir/training_state.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--frozen_codebook", action="store_true", help="Keep the quantizer fixed (encoder-only adaptation)")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = DVAEConfig(learning_rate=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs,
                        frozen_codebook=args.frozen_codebook, mixed_precision=args.mixed_precision, seed=args.seed)
    train_dvae(args.dataset, args.output_dir, config, args.base_checkpoint, args.resume, args.device)


if __name__ == "__main__":
    main()
