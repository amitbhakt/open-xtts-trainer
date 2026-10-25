"""Stage 2: fine-tune the XTTS-v2 GPT through upstream GPTTrainer (docs/recipe_design.md §6, phase 7).

The coqui Trainer runs the loop; ProjectGPTTrainer (gpt_trainer.py) carries the project's fixes. This module
prepares the data (validated dataset.jsonl, optional replay data, utterance-level train/eval split), runs
the trainer, reloads the best checkpoint by eval loss and exports it in the upstream inference format.

    python -m xtts_trainer.stage2_gpt.train --dataset data/dataset.jsonl --base_dir xtts_base \\
        --output_dir runs/gpt [--vocab_path vocab_extended.json --languages dv --replay en_data/dataset.jsonl]
"""

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from trainer import Trainer, TrainerArgs
from trainer.trainer import rank_zero_only

from xtts_trainer.data import load_validated_dataset
from xtts_trainer.models import load_state_strict, safe_torch_load
from xtts_trainer.stage2_gpt.export import export_for_inference
from xtts_trainer.stage2_gpt.gpt_trainer import (
    ProjectGPTTrainer, build_gpt_trainer_config, language_shares, records_to_samples,
)

logger = logging.getLogger(__name__)


class ProjectTrainer(Trainer):
    """Trainer subclass that stamps spec_hash on all intermediate checkpoints (Phase 2 audit requirement)."""

    @rank_zero_only
    def save_checkpoint(self) -> None:
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)

        from trainer.io import save_checkpoint as trainer_save_checkpoint
        from xtts_trainer.models.checkpoint_io import SPEC_HASH_KEY, spec_hash

        trainer_save_checkpoint(
            self.config,
            self._get_model(),
            self.output_path,
            current_step=self.total_steps_done,
            epoch=self.epochs_done,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp_scaler else None,
            model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
            save_n_checkpoints=self.config.save_n_checkpoints,
            **{SPEC_HASH_KEY: spec_hash()},
        )

    @rank_zero_only
    def save_best_model(self) -> None:
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train) or float("inf")

        from trainer.io import save_best_model as trainer_save_best_model
        from xtts_trainer.models.checkpoint_io import SPEC_HASH_KEY, spec_hash

        self.best_loss = trainer_save_best_model(
            {"train_loss": train_loss, "eval_loss": eval_loss},
            self.best_loss,
            self.config,
            self._get_model(),
            self.output_path,
            current_step=self.total_steps_done,
            epoch=self.epochs_done,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp_scaler else None,
            keep_all_best=self.config.save_all_best,
            keep_after=self.config.save_best_after,
            **{SPEC_HASH_KEY: spec_hash()},
        )


def split_samples(samples: List[Dict[str, Any]], eval_fraction: float, seed: int) -> Tuple[List, List]:
    """Utterance-level split, seeded; at least one eval sample per language when there are two or more."""
    rng = random.Random(seed)
    train, evals = [], []
    for lang in sorted({s["language"] for s in samples}):
        items = [s for s in samples if s["language"] == lang]
        rng.shuffle(items)
        n_eval = max(1, round(len(items) * eval_fraction)) if len(items) >= 2 else 0
        evals += items[:n_eval]
        train += items[n_eval:]
    if not evals:
        raise ValueError("no eval samples: every language needs at least 2 clips (best checkpoint is chosen by eval loss)")
    return train, evals


@torch.no_grad()
def eval_loss(model, config, eval_samples) -> float:
    """Mean eval loss over eval_samples, exactly as the trainer's eval epoch computes it (eval_step, no prompt
    masking). Run before training, it is the base model's score that training has to beat."""
    model.eval()
    device = next(model.parameters()).device
    losses = []
    for batch in model.get_data_loader(config, is_eval=True, samples=eval_samples):
        # As trainer.Trainer.format_batch does: format, move every tensor to the model's device, then the
        # on-device step (mel spectrograms, DVAE codes). After fit() the model is on the GPU, the loader isn't.
        batch = model.format_batch(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = model.format_batch_on_device(batch)
        _, loss_dict = model.eval_step(batch, None)
        losses.append(float(loss_dict["loss"]))
    return sum(losses) / len(losses)


def load_samples(dataset_paths: Sequence[str], vocab_path: str, languages: Sequence[str]) -> List[Dict[str, Any]]:
    samples = []
    for path in dataset_paths:
        records, dataset_dir, _ = load_validated_dataset(path, vocab_path, languages, logger=logger)
        samples += records_to_samples(records, dataset_dir)
    return samples


def train_gpt(
    dataset_path: str,
    output_dir: str,
    base_dir: str,
    vocab_path: Optional[str] = None,
    languages: Sequence[str] = (),
    replay_paths: Sequence[str] = (),
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lr: float = 5e-6,
    eval_fraction: float = 0.1,
    seed: int = 1,
    arch_overrides: Optional[Dict[str, Any]] = None,
    load_base_weights: bool = True,
    dvae_path: Optional[str] = None,
    retrain_for_new_dvae: bool = False,
    num_workers: int = 0,
) -> str:
    """Train, then export the best checkpoint (by eval loss) to output_dir/export. Returns the export dir.

    base_dir holds the released model.pth, dvae.pth and vocab.json. load_base_weights=False starts the GPT
    from random weights (tests with a reduced architecture only). dvae_path overrides base_dir/dvae.pth; a
    Stage-1-trained DVAE needs retrain_for_new_dvae (see ProjectGPTTrainer).
    """
    vocab_path = vocab_path or os.path.join(base_dir, "vocab.json")
    languages = [l.lower() for l in languages]
    samples = load_samples([dataset_path], vocab_path, languages)
    replay = load_samples(replay_paths, vocab_path, languages)
    if replay:
        overlap = {s["language"] for s in replay} & {s["language"] for s in samples}
        if overlap:
            raise ValueError(f"replay data must be in other languages than the target data; both have {sorted(overlap)}")
    train_samples, eval_samples = split_samples(samples + replay, eval_fraction, seed)
    logger.info("Stage 2: %d train / %d eval clips; language share of training draws: %s",
                len(train_samples), len(eval_samples), language_shares(train_samples))

    config = build_gpt_trainer_config(
        vocab_path, dvae_path or os.path.join(base_dir, "dvae.pth"), output_dir, extra_languages=languages,
        batch_size=batch_size, lr=lr, epochs=epochs, seed=seed,
        arch_overrides=arch_overrides, num_workers=num_workers,
    )
    base_model = os.path.join(base_dir, "model.pth") if load_base_weights else None
    model = ProjectGPTTrainer(config, extra_languages=languages, base_model_path=base_model,
                              retrain_for_new_dvae=retrain_for_new_dvae)
    if torch.cuda.is_available():
        model.cuda()  # the Trainer would move it anyway; evaluating the base model on CPU is needlessly slow
    loss_before = eval_loss(model, config, eval_samples)
    logger.info("Eval loss before training: %.4f", loss_before)
    trainer = ProjectTrainer(
        # start_with_eval would make epoch 0 eval-only (trainer._fit skips train_epoch), silently training N-1 epochs
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=grad_accum_steps),
        config, output_path=output_dir, model=model, train_samples=train_samples, eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    trainer.fit()

    best = os.path.join(trainer.output_path, "best_model.pth")
    if not os.path.exists(best):
        raise RuntimeError(f"no best_model.pth in {trainer.output_path}; training did not complete an eval")
    best_ckpt = safe_torch_load(best, map_location="cpu")
    load_state_strict(model, best_ckpt["model"], f"best checkpoint ({best})")
    loss_after = eval_loss(model, config, eval_samples)
    export_dir = os.path.join(output_dir, "export")
    export_for_inference(model, export_dir, base_model_path=base_model)
    with open(os.path.join(export_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_checkpoint": best, "best_step": best_ckpt.get("step"),
                   "eval_loss_before": loss_before, "eval_loss_best": loss_after, "improved": loss_after < loss_before,
                   "train_clips": len(train_samples), "eval_clips": len(eval_samples),
                   "language_shares": language_shares(train_samples), "languages": list(config.languages), "retrain_for_new_dvae": retrain_for_new_dvae}, f, indent=2)
    logger.info("Exported best checkpoint to %s", export_dir)
    return export_dir


def main(argv: Optional[List[str]] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fine-tune the XTTS-v2 GPT (Stage 2) through upstream GPTTrainer.")
    parser.add_argument("--dataset", required=True, help="dataset.jsonl (xtts_trainer/data/schema.py)")
    parser.add_argument("--base_dir", required=True, help="Directory with the released model.pth, dvae.pth, vocab.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_path", default=None, help="Extended vocab (default: base_dir/vocab.json)")
    parser.add_argument("--languages", nargs="+", default=[], help="New language codes registered in the vocab")
    parser.add_argument("--replay", nargs="+", default=[], help="dataset.jsonl files of base-language replay data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Effective batch = batch_size x this (>= 32)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dvae_checkpoint", default=None, help="DVAE to code audio with (default: base_dir/dvae.pth)")
    parser.add_argument("--num_workers", type=int, default=2, help="Data loader worker processes")
    parser.add_argument("--retrain_for_new_dvae", action="store_true",
                        help="Required with a Stage-1-trained DVAE: retrain the GPT on its codes")
    args = parser.parse_args(argv)
    if args.batch_size * args.grad_accum_steps < 32:
        logger.warning("effective batch %d < 32 (docs/recipe_design.md §3 starting point)", args.batch_size * args.grad_accum_steps)
    train_gpt(args.dataset, args.output_dir, args.base_dir, args.vocab_path, args.languages, args.replay,
              args.epochs, args.batch_size, args.grad_accum_steps, args.lr, args.eval_fraction, args.seed,
              dvae_path=args.dvae_checkpoint, retrain_for_new_dvae=args.retrain_for_new_dvae,
              num_workers=args.num_workers)


if __name__ == "__main__":
    main()
