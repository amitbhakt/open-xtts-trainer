"""Stage 3: fine-tune the XTTS-v2 HiFi-GAN decoder on GPT latents (docs/recipe_design.md §6, phase 7).

Optional in the recipe (11.3): keep the result only if it beats the stock decoder in the Phase 8 A/B.
What this trainer guarantees:
- decoder input interpolated over each whole clip before slicing (dataset.py), so segments align exactly;
- validation on held-out whole clips: log-mel L1 of the full-clip resynthesis, measured for the starting
  decoder first (the number to beat; for the released decoder this is the Phase 5 round-trip canary's
  measurement) and every val_every_steps after; audio samples written periodically; the best checkpoint by
  validation error;
- discriminator warm-up in steps (fresh discriminators, generator on mel loss only), refused if the run
  would never leave it, unless mel_only;
- optional Speaker Consistency Loss (the decoder's own speaker encoder, frozen) and full-band mel loss;
- seeded runs, warm-up LR schedule, mixed precision on CUDA and full resume (xtts_trainer.training).
Latents must come from extract_latents (eval-mode GPT), which the latent schema checks.
"""

import argparse
import dataclasses
import itertools
import json
import logging
import os
from typing import Any, Dict, List, Optional

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from xtts_trainer.config import HiFiGANConfig
from xtts_trainer.data import load_validated_dataset
from xtts_trainer.frontend import LogMel, vocoder_loss_mel
from xtts_trainer.models import build_hifigan, check_spec_hash, load_state_strict, safe_torch_load, stamp
from xtts_trainer.spec import SPEC
from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset, collate_segments
from xtts_trainer.stage3_hifigan.models import (
    MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, feature_loss, generator_loss,
)
from xtts_trainer.training import (
    Precision, decay_per_step, is_improvement, load_training_state, save_training_state, seed_everything,
    split_records, warmup_lambda,
)

logger = logging.getLogger(__name__)

MEL_LOSS_WEIGHT = 45.0  # HiFi-GAN (Kong et al., 2020) mel-L1 weight
TRAINING_STATE_FILENAME = "training_state.pth"


def generator_mel_loss(mel_transform, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Weighted L1 between log-mels of generated and target audio ([B, 1, T] each)."""
    return F.l1_loss(mel_transform(y_hat.squeeze(1)), mel_transform(y.squeeze(1))) * MEL_LOSS_WEIGHT


def full_band_loss_mel(device: str = "cpu") -> LogMel:
    """The Stage 3 loss mel extended to Nyquist (the default stops at 8 kHz)."""
    return LogMel(dataclasses.replace(SPEC.vocoder_loss_mel, fmax=SPEC.output_sample_rate / 2)).to(device)


def speaker_consistency_loss(speaker_encoder, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity of speaker embeddings (decoder's own encoder, 16 kHz), batch mean.

    Training segments are short (segment_frames x 256 samples, ~0.7 s by default), and speaker embeddings of
    sub-second audio are unreliable (Colab run 2: SECS on 0.7-1.0 s clips); hence off by default. [E2] the
    XTTS paper uses it; its value for fine-tuning is ablation material.
    """
    def embed(audio):
        audio_16k = torchaudio.functional.resample(audio.float(), SPEC.output_sample_rate, SPEC.speaker_encoder_sample_rate)
        return speaker_encoder(audio_16k.clone(), l2_norm=True)  # forward squeezes its input in place

    with torch.no_grad():
        target = embed(y)
    return (1 - F.cosine_similarity(embed(y_hat), target, dim=-1)).mean()


def load_base_decoder(generator, model_path: str) -> None:
    ckpt = safe_torch_load(model_path, map_location="cpu")
    check_spec_hash(ckpt, model_path)
    weights = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k[len("hifigan_decoder."):]: v for k, v in weights.items() if k.startswith("hifigan_decoder.")}
    if not state:
        raise KeyError(f"{model_path} has no hifigan_decoder.* tensors")
    load_state_strict(generator, state, f"HiFi-GAN decoder ({model_path})")


@torch.no_grad()
def evaluate_decoder(generator, val_clips: List, mel, device: str, sample_dir: Optional[str] = None) -> float:
    """Mean log-mel L1 between whole-clip resynthesis and the original, over the validation clips."""
    was_training = generator.training
    generator.eval()
    errors = []
    for name, (audio, latents, speaker) in val_clips:
        y_hat = generator(latents.to(device), g=speaker.unsqueeze(0).to(device)).reshape(-1).float().cpu()
        target = audio.reshape(-1)
        n = min(len(y_hat), len(target))
        errors.append(float((mel(y_hat[:n].unsqueeze(0)) - mel(target[:n].unsqueeze(0))).abs().mean()))
        if sample_dir:
            os.makedirs(sample_dir, exist_ok=True)
            sf.write(os.path.join(sample_dir, f"{name}.wav"), y_hat[:n].numpy(), SPEC.output_sample_rate)
    generator.train(was_training)
    return sum(errors) / len(errors)


def train_hifigan(
    dataset_path: str,
    latents_dir: str,
    output_dir: str,
    config: Optional[HiFiGANConfig] = None,
    checkpoint_dir: str = "",
    resume: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    allow_synthetic_latents: bool = False,
) -> str:
    """Train, keeping the best decoder by validation error in output_dir/hifigan.pth. Returns its path.

    checkpoint_dir: base model directory whose model.pth decoder is fine-tuned; empty trains from scratch.
    resume: continue from output_dir/training_state.pth. allow_synthetic_latents: unit tests only.
    """
    config = config or HiFiGANConfig()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    os.makedirs(output_dir, exist_ok=True)
    loader_gen = seed_everything(config.seed)

    generator = build_hifigan().to(device)
    finetuning = bool(checkpoint_dir)
    if finetuning:
        load_base_decoder(generator, os.path.join(checkpoint_dir, "model.pth"))
    else:
        logger.warning("Training the decoder from scratch needs hundreds of hours of data; for fine-tuning pass checkpoint_dir.")
    speaker_encoder = generator.speaker_encoder.eval().requires_grad_(False)

    records, dataset_dir, _ = load_validated_dataset(dataset_path, None, require_text=False, logger=logger)
    train_records, val_records = split_records(records, config.val_fraction, config.seed)
    train_set = LatentHiFiGANDataset(train_records, dataset_dir, latents_dir, generator, config.segment_frames,
                                     allow_synthetic_latents)
    val_set = LatentHiFiGANDataset(val_records[:config.max_val_clips], dataset_dir, latents_dir, generator,
                                   config.segment_frames, allow_synthetic_latents)
    if val_set.checkpoint_sha256 != train_set.checkpoint_sha256:
        raise ValueError("train and validation latents come from different GPT checkpoints")
    val_clips = [(os.path.splitext(os.path.basename(a))[0], val_set.full_clip(i)) for i, (a, _) in enumerate(val_set.items)]
    batch_size = max(1, min(config.batch_size, len(train_set)))
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_segments,
                        generator=loader_gen, num_workers=config.num_workers,
                        drop_last=len(train_set) >= 2 * batch_size)

    total_steps = config.epochs * len(loader)
    disc_warmup = config.disc_warmup_steps if finetuning else 0
    if not config.mel_only and total_steps <= disc_warmup:
        raise ValueError(
            f"{total_steps} steps never leave the {disc_warmup}-step discriminator warm-up, so no adversarial "
            f"training would happen; train longer, lower disc_warmup_steps, or set mel_only for a mel-only fine-tune"
        )
    logger.info("Stage 3: %d train / %d val clips, %d steps (%d discriminator warm-up%s)", len(train_set),
                len(val_clips), total_steps, disc_warmup, ", mel only" if config.mel_only else "")

    mpd, msd = MultiPeriodDiscriminator().to(device), MultiScaleDiscriminator().to(device)
    lr_g = config.finetune_learning_rate if finetuning else config.learning_rate
    optim_g = torch.optim.AdamW(generator.waveform_decoder.parameters(), lr=lr_g, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()),
                                lr=lr_g * config.discriminator_lr_ratio, betas=(0.8, 0.99))
    schedule = warmup_lambda(config.lr_warmup_steps, decay_per_step(config.lr_decay, len(loader)))
    sched_g = torch.optim.lr_scheduler.LambdaLR(optim_g, schedule)
    sched_d = torch.optim.lr_scheduler.LambdaLR(optim_d, schedule)
    precision = Precision(device, config.mixed_precision)
    scaler_g, scaler_d = precision.scaler(), precision.scaler()
    run = dict(models={"generator": generator, "mpd": mpd, "msd": msd},
               optimizers={"g": optim_g, "d": optim_d}, schedulers={"g": sched_g, "d": sched_d},
               scalers={"g": scaler_g, "d": scaler_d})

    train_mel = full_band_loss_mel(device) if config.full_band_mel else vocoder_loss_mel(device)
    val_mel = vocoder_loss_mel("cpu")  # always the standard band, so runs stay comparable
    state_path = os.path.join(output_dir, TRAINING_STATE_FILENAME)
    best_path = os.path.join(output_dir, "hifigan.pth")
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    step, start_epoch, best = 0, 1, None
    if resume:
        restored = load_training_state(state_path, **run)
        step, start_epoch, best = restored["step"], restored["epoch"] + 1, restored["best"]
        loader_gen.set_state(restored["extra"]["loader_generator"])
        reference = restored["extra"]["reference"]
    else:
        reference = evaluate_decoder(generator, val_clips, val_mel, device, os.path.join(output_dir, "samples", "step_0"))
        logger.info("Validation error of the starting decoder: %.4f (to beat)", reference)

    def validate(epoch: int) -> None:
        nonlocal best
        samples = os.path.join(output_dir, "samples", f"step_{step}") if step % config.sample_every_steps == 0 else None
        error = evaluate_decoder(generator, val_clips, val_mel, device, samples)
        record = {"step": step, "epoch": epoch, "val_mel_l1": error, "reference": reference,
                  "lr_g": sched_g.get_last_lr()[0]}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Validation step %d: mel L1 %.4f (start %.4f)", step, error, reference)
        if is_improvement(error, best):
            best = error
            torch.save(stamp({
                "hifigan_decoder": {k: v.cpu() for k, v in generator.state_dict().items()},
                "val_mel_l1": error, "reference_val_mel_l1": reference, "beats_start": error < reference,
                "step": step, "latents_checkpoint_sha256": train_set.checkpoint_sha256,
            }), best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)

    for epoch in range(start_epoch, config.epochs + 1):
        generator.train()
        speaker_encoder.eval()
        mpd.train()
        msd.train()
        for y, frames, speakers in loader:
            y, frames, speakers = y.to(device), frames.to(device), speakers.to(device)
            adversarial = not config.mel_only and step >= disc_warmup
            with precision.autocast():
                y_hat = generator.waveform_decoder(frames, g=speakers)
            n = min(y.shape[-1], y_hat.shape[-1])
            y, y_hat = y[..., :n], y_hat[..., :n]

            if not config.mel_only:  # discriminators train from step 0, including during warm-up
                with precision.autocast():
                    r_f, g_f, _, _ = mpd(y, y_hat.detach())
                    r_s, g_s, _, _ = msd(y, y_hat.detach())
                    loss_d = discriminator_loss(r_f, g_f) + discriminator_loss(r_s, g_s)
                optim_d.zero_grad()
                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(optim_d)
                torch.nn.utils.clip_grad_norm_(itertools.chain(mpd.parameters(), msd.parameters()), config.grad_clip)
                scaler_d.step(optim_d)
                scaler_d.update()

            loss_g = generator_mel_loss(train_mel, y_hat.float(), y)
            if config.scl_weight > 0:
                loss_g = loss_g + config.scl_weight * speaker_consistency_loss(speaker_encoder, y_hat, y)
            if adversarial:
                with precision.autocast():
                    _, g_f, fr_f, fg_f = mpd(y, y_hat)
                    _, g_s, fr_s, fg_s = msd(y, y_hat)
                    loss_g = loss_g + generator_loss(g_f) + generator_loss(g_s) + feature_loss(fr_f, fg_f) + feature_loss(fr_s, fg_s)
            if not torch.isfinite(loss_g):
                raise FloatingPointError(f"non-finite generator loss at step {step}")
            optim_g.zero_grad()
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(generator.waveform_decoder.parameters(), config.grad_clip)
            scaler_g.step(optim_g)
            scaler_g.update()
            sched_g.step()
            sched_d.step()
            step += 1
            if step % config.val_every_steps == 0:
                validate(epoch)
        if step % config.val_every_steps:
            validate(epoch)
        save_training_state(state_path, **run, step=step, epoch=epoch, best=best,
                            extra={"loader_generator": loader_gen.get_state(), "reference": reference})

    logger.info("Stage 3 finished; best validation mel L1 %.4f (starting decoder %.4f): %s", best, reference, best_path)
    return best_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fine-tune the Stage 3 HiFi-GAN decoder on GPT latents.")
    parser.add_argument("--dataset", required=True, help="dataset.jsonl the latents were extracted from")
    parser.add_argument("--latents_dir", required=True, help="Directory of extract_latents output")
    parser.add_argument("--output_dir", default="./checkpoints/hifigan")
    parser.add_argument("--checkpoint_dir", default="", help="Base model directory (model.pth) to fine-tune")
    parser.add_argument("--resume", action="store_true", help="Continue from output_dir/training_state.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--disc_warmup_steps", type=int, default=1000)
    parser.add_argument("--mel_only", action="store_true", help="No adversarial training (mel-only fine-tune)")
    parser.add_argument("--scl_weight", type=float, default=0.0, help="Speaker Consistency Loss weight (0 = off)")
    parser.add_argument("--full_band_mel", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = HiFiGANConfig(epochs=args.epochs, batch_size=args.batch_size, disc_warmup_steps=args.disc_warmup_steps,
                           mel_only=args.mel_only, scl_weight=args.scl_weight, full_band_mel=args.full_band_mel,
                           mixed_precision=not args.no_mixed_precision, seed=args.seed)
    train_hifigan(args.dataset, args.latents_dir, args.output_dir, config, args.checkpoint_dir, args.resume, args.device)


if __name__ == "__main__":
    main()
