"""Synthesize waveform audio from trained HiFi-GAN vocoder checkpoint and extracted GPT latents."""

import os
import argparse
import torch
import soundfile as sf
import numpy as np
from xtts_trainer.config import HiFiGANConfig


def synthesize_audio(
    checkpoint_path: str,
    latent_path: str,
    output_path: str = "synthesized.wav",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sample_rate: int = 24000,
):
    """Loads a trained HiFi-GAN vocoder checkpoint and synthesizes audio from latents."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Loading HiFi-GAN checkpoint: {checkpoint_path} on {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 1. Initialize Generator matching training config
    config = HiFiGANConfig()
    try:
        from TTS.vocoder.models.hifigan_generator import HifiganGenerator
        generator = HifiganGenerator(
            in_channels=1024,
            out_channels=1,
            resblock_type="1",
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_factors=config.upsample_rates,
        ).to(device)
    except Exception:
        from xtts_trainer.stage3_hifigan.models import HiFiGANGenerator
        generator = HiFiGANGenerator(in_channels=1024).to(device)

    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    # 2. Load latent
    print(f"Loading latent file: {latent_path}...")
    latent_data = torch.load(latent_path, map_location=device)

    if isinstance(latent_data, dict) and "gpt_latents" in latent_data:
        latents = latent_data["gpt_latents"].to(device)
        speaker_emb = latent_data.get("speaker_embedding", None)
        if speaker_emb is not None:
            speaker_emb = speaker_emb.to(device)
    elif isinstance(latent_data, torch.Tensor):
        latents = latent_data.to(device)
        speaker_emb = None
    else:
        raise ValueError("Unrecognized latent file structure.")

    # HiFi-GAN expects [B, Channels, Time]
    if latents.ndim == 2:
        latents = latents.unsqueeze(0)
    if latents.shape[1] != 1024 and latents.shape[2] == 1024:
        latents = latents.transpose(1, 2)

    # 3. Vocoder synthesis forward pass
    print("Synthesizing audio waveform...")
    with torch.no_grad():
        try:
            waveform = generator(latents, g=speaker_emb)
        except Exception:
            waveform = generator(latents)

    audio = waveform.squeeze().cpu().numpy()

    # Normalize audio
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.9 / peak)

    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    duration = len(audio) / sample_rate
    print(f"Successfully saved synthesized audio to: {output_path} ({duration:.2f}s at {sample_rate}Hz)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Synthesize audio from HiFi-GAN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to hifigan checkpoint (.pt)")
    parser.add_argument("--latent", type=str, required=True, help="Path to .pt latent file")
    parser.add_argument("--output", type=str, default="synthesized.wav", help="Output .wav path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    synthesize_audio(
        checkpoint_path=args.checkpoint,
        latent_path=args.latent,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
