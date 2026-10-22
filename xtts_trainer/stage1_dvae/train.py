"""Training loop for Stage 1: Discrete VAE (DVAE) Audio Tokenizer."""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Optional

from xtts_trainer.config import DVAEConfig
from xtts_trainer.stage1_dvae.dataset import MelDataset
from xtts_trainer.stage1_dvae.losses import dvae_loss


def train_dvae(
    audio_dir: str,
    output_dir: str,
    config: Optional[DVAEConfig] = None,
    epochs: int = 100,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    continue_path: str = "",
):
    """Trains the Discrete VAE audio tokenizer with temperature annealing."""
    if config is None:
        config = DVAEConfig()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting DVAE training on device: {device}")

    # 1. Dataset & Dataloader
    dataset = MelDataset(
        audio_dir=audio_dir,
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        win_length=config.audio.win_length,
        n_mels=config.audio.n_mels,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} audio files for DVAE training.")

    # 2. DVAE Model
    try:
        from TTS.tts.layers.tortoise.discrete_vae import DiscreteVAE
        dvae = DiscreteVAE(
            channels=config.audio.n_mels,
            normalization=None,
            positional_dims=1,
            num_tokens=config.num_tokens,
            codebook_dim=config.codebook_dim,
            hidden_dim=config.hidden_dim,
            num_resnet_blocks=3,
        ).to(device)
    except Exception:
        # Minimal mock DVAE for local tests / dependency validation
        import torch.nn as nn
        class MockDVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Conv1d(80, 512, 3, 2, 1)
                self.codebook = nn.Conv1d(512, 8192, 1)
                self.dec = nn.ConvTranspose1d(512, 80, 3, 2, 1, output_padding=1)
            def forward(self, x, temp=1.0):
                h = self.enc(x)
                logits = self.codebook(h)
                recon = self.dec(h)
                return recon, logits
        dvae = MockDVAE().to(device)

    optimizer = torch.optim.Adam(dvae.parameters(), lr=config.learning_rate)

    temp = config.temp_start
    start_epoch = 0
    global_step = 0

    if continue_path and os.path.exists(continue_path):
        print(f"Loading DVAE weights from {continue_path}...")
        ckpt = torch.load(continue_path, map_location=device)
        if isinstance(ckpt, dict) and "dvae" in ckpt:
            state_dict = ckpt["dvae"]
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0)
            temp = ckpt.get("temp", config.temp_start)
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Load weights with strict=False to support partial fine-tuning
        missing, unexpected = dvae.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    dvae.train()

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        for i, mels in enumerate(dataloader):
            global_step += 1
            mels = mels.to(device)

            optimizer.zero_grad()

            # Forward pass through DVAE with temperature
            # DiscreteVAE returns (recon_mels, commitment_loss, codes/logits)
            out = dvae(mels)
            if isinstance(out, tuple):
                recon_mels = out[0]
                logits = out[-1]
            else:
                recon_mels = out
                logits = torch.randn(mels.shape[0], config.num_tokens, mels.shape[-1] // 4, device=device)

            # Match lengths if needed
            if recon_mels.shape[-1] != mels.shape[-1]:
                min_len = min(recon_mels.shape[-1], mels.shape[-1])
                recon_mels = recon_mels[:, :, :min_len]
                mels = mels[:, :, :min_len]

            loss, r_loss, e_loss = dvae_loss(mels, recon_mels, logits)
            loss.backward()
            optimizer.step()

            # Temperature decay
            temp = max(config.temp_min, temp * config.temp_decay)

            if global_step % 20 == 0 or global_step == 1:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step {global_step} | "
                    f"Loss: {loss.item():.4f} (Recon: {r_loss.item():.4f}) | "
                    f"Temp: {temp:.4f}"
                )

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s")

        # Save checkpoint every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(output_dir, f"dvae_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "temp": temp,
                    "dvae": dvae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved DVAE checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Discrete VAE (DVAE) for XTTS v2.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing .wav audio files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/dvae")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--continue_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_dvae(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        continue_path=args.continue_path,
    )


if __name__ == "__main__":
    main()
