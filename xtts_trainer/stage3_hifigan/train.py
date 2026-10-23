"""Training script for Stage 3: Latent HiFi-GAN Vocoder Fine-Tuning."""

import argparse
import itertools
import logging
import os
import time
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from typing import Optional

from xtts_trainer.config import HiFiGANConfig, safe_torch_load
from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset
from xtts_trainer.stage3_hifigan.models import (
    build_hifigan_decoder,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["COQUI_TOS_AGREED"] = "1"


def train_hifigan(
    audio_dir: str,
    latents_dir: str,
    output_dir: str,
    config: Optional[HiFiGANConfig] = None,
    checkpoint_dir: str = "",
    continue_path: str = "",
    epochs: int = 50,
    batch_size: int = 16,
    latent_frames_slice: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Executes dual-optimizer fine-tuning for native XTTS HiFi-GAN Vocoder."""
    if config is None:
        config = HiFiGANConfig()

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting Stage 3 HiFi-GAN Vocoder training on device: {device}")

    # 1. Dataset & DataLoader (aligned temporal slicing)
    dataset = LatentHiFiGANDataset(
        audio_dir=audio_dir,
        latents_dir=latents_dir,
        latent_frames_slice=latent_frames_slice,
        sample_rate=config.audio.sample_rate,
    )
    if len(dataset) == 0:
        raise ValueError(f"No valid audio-latent paired samples found in {audio_dir} and {latents_dir}")
    effective_batch_size = max(1, min(batch_size, len(dataset)))
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=(len(dataset) >= effective_batch_size * 2),
    )
    logger.info(f"Loaded {len(dataset)} audio-latent paired samples for vocoder training (batch size: {effective_batch_size}, steps per epoch: {len(dataloader)}).")

    # 2. Native Generator & Discriminators
    generator: HifiDecoder = build_hifigan_decoder().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # 3. Load Pretrained Weights
    # Check if checkpoint_dir (with model.pth) or continue_path is provided
    model_pth_path = continue_path or (os.path.join(checkpoint_dir, "model.pth") if checkpoint_dir else "")
    if model_pth_path and os.path.exists(model_pth_path):
        logger.info(f"Loading pretrained vocoder weights from: {model_pth_path}")
        ckpt = safe_torch_load(model_pth_path, map_location=device)
        raw_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # Filter hifigan_decoder keys
        hifi_state = {}
        for k, v in raw_model.items():
            if k.startswith("hifigan_decoder."):
                hifi_state[k[16:]] = v
            elif k.startswith("waveform_decoder."):
                hifi_state[k] = v

        if hifi_state:
            missing, unexpected = generator.load_state_dict(hifi_state, strict=False)
            logger.info(f"Restored pretrained vocoder weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
        elif "generator" in ckpt:
            generator.waveform_decoder.load_state_dict(ckpt["generator"], strict=False)
            logger.info("Restored generator weights from checkpoint.")
        else:
            logger.warning("No hifigan_decoder keys found in checkpoint. Starting from initialized weights.")
    else:
        logger.info("No pretrained vocoder checkpoint provided. Training from initialized weights.")

    # 4. Mel-Spectrogram transform for mel-loss (24kHz, 80 mel channels)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000,
    ).to(device)

    # 5. Optimizers with AdamW
    optim_g = torch.optim.AdamW(generator.parameters(), lr=config.learning_rate, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), msd.parameters()),
        lr=config.discriminator_learning_rate,
        betas=(0.8, 0.99),
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay)

    generator.train()
    mpd.train()
    msd.train()

    last_ckpt_path = ""
    global_step = 0

    logger.info(f"Beginning Stage 3 training for {epochs} epochs (Batch size: {batch_size}, Steps per epoch: {len(dataloader)})...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss_g = 0.0
        total_loss_d = 0.0
        total_loss_mel = 0.0

        for i, (y, latents, speaker_emb) in enumerate(dataloader):
            global_step += 1
            y = y.to(device)                        # [B, 1, T_audio]
            latents = latents.to(device)            # [B, 16, 1024]
            speaker_emb = speaker_emb.to(device)    # [B, 512, 1]

            # ---------------------------
            # Forward Generator
            # ---------------------------
            y_g_hat = generator(latents, g=speaker_emb)  # [B, 1, T_audio]

            # Compute Mel loss
            y_mel = torch.log(torch.clamp(mel_transform(y.squeeze(1)), min=1e-5))
            y_g_hat_mel = torch.log(torch.clamp(mel_transform(y_g_hat.squeeze(1)), min=1e-5))
            loss_mel = F.l1_loss(y_g_hat_mel, y_mel) * 45.0

            # ---------------------------
            # Train Discriminator
            # ---------------------------
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_d = loss_disc_f + loss_disc_s
            loss_d.backward()
            optim_d.step()

            # ---------------------------
            # Train Generator
            # ---------------------------
            optim_g.zero_grad()

            # L1 Mel Loss + GAN Loss + Feature Matching Loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s = generator_loss(y_ds_hat_g)

            loss_g = loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + loss_mel
            loss_g.backward()
            optim_g.step()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            total_loss_mel += loss_mel.item()

            if (i + 1) % max(1, len(dataloader) // 5) == 0 or (i + 1) == len(dataloader):
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Step [{i+1}/{len(dataloader)}] | "
                    f"Loss Gen: {loss_g.item():.4f} (Mel: {loss_mel.item():.4f}) | Loss Disc: {loss_d.item():.4f}"
                )

        scheduler_g.step()
        scheduler_d.step()

        avg_loss_g = total_loss_g / len(dataloader)
        avg_loss_d = total_loss_d / len(dataloader)
        avg_loss_mel = total_loss_mel / len(dataloader)
        elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch} finished in {elapsed:.2f}s | Avg Gen Loss: {avg_loss_g:.4f} (Mel: {avg_loss_mel:.4f}) | Avg Disc Loss: {avg_loss_d:.4f}")

        # 6. Save Checkpoints
        last_ckpt_path = os.path.join(output_dir, f"hifigan_epoch_{epoch}.pt")
        save_payload = {
            "hifigan_decoder": generator.state_dict(),
            "generator": generator.waveform_decoder.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "epoch": epoch,
            "step": global_step,
        }
        torch.save(save_payload, last_ckpt_path)

        # Also maintain latest hifigan.pth
        torch.save(save_payload, os.path.join(output_dir, "hifigan.pth"))

    logger.info(f"Stage 3 HiFi-GAN training finished. Final model saved to: {output_dir}")
    return last_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train Stage 3 Latent HiFi-GAN Vocoder for XTTS v2.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio (.wav) files")
    parser.add_argument("--latents_dir", type=str, required=True, help="Directory containing extracted .pt latents")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hifigan")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Path to base XTTS checkpoint directory")
    parser.add_argument("--continue_path", type=str, default="", help="Path to vocoder checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_hifigan(
        audio_dir=args.audio_dir,
        latents_dir=args.latents_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        continue_path=args.continue_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
