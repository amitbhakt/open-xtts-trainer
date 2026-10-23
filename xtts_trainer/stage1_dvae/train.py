"""Training loop for Stage 1: Discrete VAE (DVAE) Audio Tokenizer."""

import argparse
import logging
import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Optional

from xtts_trainer.config import DVAEConfig
from xtts_trainer.stage1_dvae.dataset import MelDataset
from TTS.tts.layers.xtts.dvae import DiscreteVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_dvae_model(
    channels: int = 80,
    num_tokens: int = 8192,
    codebook_dim: int = 512,
    hidden_dim: int = 512,
) -> DiscreteVAE:
    """Builds the native XTTS-v2 Discrete VAE architecture."""
    return DiscreteVAE(
        channels=channels,
        normalization=None,
        positional_dims=1,
        num_tokens=num_tokens,
        codebook_dim=codebook_dim,
        hidden_dim=hidden_dim,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )


def train_dvae(
    audio_dir: str,
    output_dir: str,
    config: Optional[DVAEConfig] = None,
    epochs: int = 100,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    continue_path: str = "",
) -> str:
    """Trains or fine-tunes the Discrete VAE audio tokenizer."""
    if config is None:
        config = DVAEConfig()

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting Stage 1 DVAE training on device: {device}")

    # 1. Dataset & Dataloader (native 22,050 Hz mel spectrograms)
    dataset = MelDataset(
        audio_dir=audio_dir,
        sample_rate=getattr(config, "sample_rate", 22050),
        n_mels=getattr(config, "n_mels", 80),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    logger.info(f"Loaded {len(dataset)} audio files for DVAE training.")

    # 2. Native DVAE Model
    dvae = build_dvae_model(
        channels=getattr(config, "n_mels", 80),
        num_tokens=config.num_tokens,
        codebook_dim=config.codebook_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(dvae.parameters(), lr=config.learning_rate)

    start_epoch = 0
    global_step = 0

    if continue_path:
        if not os.path.exists(continue_path):
            raise FileNotFoundError(f"DVAE checkpoint not found at: {continue_path}")

        logger.info(f"Loading pretrained DVAE weights from: {continue_path}")
        ckpt = torch.load(continue_path, map_location=device)
        if isinstance(ckpt, dict) and "dvae" in ckpt:
            state_dict = ckpt["dvae"]
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception as ex:
                    logger.warning(f"Could not load optimizer state ({ex}). Re-initializing optimizer.")
            start_epoch = ckpt.get("epoch", 0)
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        missing, unexpected = dvae.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded DVAE weights (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    dvae.train()
    last_ckpt = ""

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        total_recon = 0.0
        total_commit = 0.0

        for i, mels in enumerate(dataloader):
            global_step += 1
            mels = mels.to(device)

            optimizer.zero_grad()

            # Native DVAE forward computes reconstruction and commitment losses
            recon_loss, commitment_loss, recon_mels = dvae(mels)
            loss = recon_loss.mean() + commitment_loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dvae.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.mean().item()
            total_commit += commitment_loss.mean().item()

            if (i + 1) % max(1, len(dataloader) // 5) == 0 or (i + 1) == len(dataloader):
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Step [{i+1}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f} (Recon: {recon_loss.mean().item():.4f}, Commit: {commitment_loss.mean().item():.4f})"
                )

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_commit = total_commit / len(dataloader)
        elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch} finished in {elapsed:.2f}s | Avg Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f})")

        # Save checkpoint
        last_ckpt = os.path.join(output_dir, f"dvae_epoch_{epoch}.pth")
        torch.save(
            {
                "dvae": dvae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
            },
            last_ckpt,
        )

        # Maintain standard dvae.pth for downstream stage consumption
        torch.save(
            {
                "dvae": dvae.state_dict(),
                "epoch": epoch,
            },
            os.path.join(output_dir, "dvae.pth"),
        )

    logger.info(f"DVAE training finished. Final model saved to: {output_dir}")
    return last_ckpt


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Discrete VAE (DVAE) Audio Tokenizer.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio (.wav) files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/dvae", help="Directory to save checkpoints")
    parser.add_argument("--continue_path", type=str, default="", help="Path to base dvae.pth to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = DVAEConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    train_dvae(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        continue_path=args.continue_path,
    )


if __name__ == "__main__":
    main()
