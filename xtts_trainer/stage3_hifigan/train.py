"""Training loop for Stage 3: Latent-Conditioned HiFi-GAN Vocoder."""

import argparse
import os
import time
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from typing import Optional

from xtts_trainer.config import HiFiGANConfig
from xtts_trainer.stage3_hifigan.models import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    generator_loss,
    feature_loss,
)
from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> torch.Tensor:
    """Computes log-mel spectrogram for mel reconstruction loss."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=win_size,
        hop_length=hop_size,
        f_min=fmin,
        f_max=fmax,
        n_mels=num_mels,
        power=1.0,
        center=False,
    ).to(y.device)
    mel = mel_transform(y)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    return log_mel


def train_hifigan(
    audio_dir: str,
    latents_dir: str,
    output_dir: str,
    config: Optional[HiFiGANConfig] = None,
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    continue_path: str = "",
):
    """Executes the dual-optimizer training loop for Latent HiFi-GAN."""
    if config is None:
        config = HiFiGANConfig()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting HiFi-GAN Vocoder training on device: {device}")

    # 1. Dataset & Dataloader
    dataset = LatentHiFiGANDataset(
        audio_dir=audio_dir,
        latents_dir=latents_dir,
        segment_size=config.segment_size,
        hop_length=config.audio.hop_length,
        sample_rate=config.audio.sample_rate,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} paired samples.")

    # 2. Models: Generator + Discriminators
    # Attempt to load Coqui's HiFi-GAN generator architecture
    try:
        from TTS.vocoder.models.hifigan_generator import HifiganGenerator
        generator = HifiganGenerator(
            in_channels=1024,  # GPT latent dimension
            out_channels=1,
            resblock_type="1",
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_factors=config.upsample_rates,
        ).to(device)
    except Exception:
        # Fallback minimal generator for CPU tests / independent runs
        import torch.nn as nn
        class MockGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.ConvTranspose1d(1024, 1, kernel_size=512, stride=256, padding=128)
            def forward(self, x, g=None):
                return self.conv(x)
        generator = MockGenerator().to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # 3. Optimizers with AdamW
    optim_g = torch.optim.AdamW(generator.parameters(), lr=config.learning_rate, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=config.discriminator_learning_rate,
        betas=(0.8, 0.99),
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay)

    start_epoch = 0
    global_step = 0

    if continue_path and os.path.exists(continue_path):
        print(f"Resuming checkpoint from {continue_path}...")
        ckpt = torch.load(continue_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        msd.load_state_dict(ckpt["msd"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)

    # 4. Training Loop
    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        for i, (audio, latents, speaker_emb) in enumerate(dataloader):
            global_step += 1
            audio = audio.to(device)
            latents = latents.to(device)
            speaker_emb = speaker_emb.to(device)

            # Generate audio waveform from latents
            # generator takes latents + optional speaker embedding
            try:
                audio_hat = generator(latents, g=speaker_emb)
            except TypeError:
                audio_hat = generator(latents)

            # Make sure shapes match exactly
            if audio_hat.shape[-1] != audio.shape[-1]:
                min_len = min(audio_hat.shape[-1], audio.shape[-1])
                audio_hat = audio_hat[:, :, :min_len]
                audio = audio[:, :, :min_len]

            # ---------------------
            # Train Discriminators
            # ---------------------
            optim_d.zero_grad()

            # MPD Loss
            df_r_rs, df_r_gs, _, _ = mpd(audio, audio_hat.detach())
            loss_disc_f = discriminator_loss(df_r_rs, df_r_gs)

            # MSD Loss
            df_s_rs, df_s_gs, _, _ = msd(audio, audio_hat.detach())
            loss_disc_s = discriminator_loss(df_s_rs, df_s_gs)

            loss_d = loss_disc_f + loss_disc_s
            loss_d.backward()
            optim_d.step()

            # -----------------
            # Train Generator
            # -----------------
            optim_g.zero_grad()

            # L1 Mel Spectrogram Loss
            mel_real = mel_spectrogram(audio.squeeze(1))
            mel_gen = mel_spectrogram(audio_hat.squeeze(1))
            loss_mel = F.l1_loss(mel_gen, mel_real) * 45.0

            # Adversarial and Feature Matching Loss
            _, df_r_gs, fmap_f_r, fmap_f_g = mpd(audio, audio_hat)
            _, df_s_gs, fmap_s_r, fmap_s_g = msd(audio, audio_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f = generator_loss(df_r_gs)
            loss_gen_s = generator_loss(df_s_gs)

            loss_g = loss_mel + loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s
            loss_g.backward()
            optim_g.step()

            if global_step % 10 == 0 or global_step == 1:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step {global_step} | "
                    f"Loss Gen: {loss_g.item():.4f} (Mel: {loss_mel.item():.4f}) | "
                    f"Loss Disc: {loss_d.item():.4f}"
                )

        scheduler_g.step()
        scheduler_d.step()
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(output_dir, f"hifigan_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "generator": generator.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Latent HiFi-GAN Vocoder for XTTS v2.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to ground truth audio (.wav)")
    parser.add_argument("--latents_dir", type=str, required=True, help="Path to extracted GPT latents (.pt)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hifigan")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--continue_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_hifigan(
        audio_dir=args.audio_dir,
        latents_dir=args.latents_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        continue_path=args.continue_path,
    )


if __name__ == "__main__":
    main()
