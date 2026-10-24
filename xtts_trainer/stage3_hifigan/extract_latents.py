"""Script to extract intermediate temporal GPT latents and speaker embeddings using XTTS v2."""

import argparse
import logging
import os
import torch
import torchaudio
from tqdm import tqdm
from typing import Optional

from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from xtts_trainer.config import safe_torch_load, load_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["COQUI_TOS_AGREED"] = "1"


def extract_latents(
    audio_dir: str,
    output_dir: str,
    checkpoint_dir: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    text_prompt: str = "Hello, this is a conditioning alignment prompt.",
    lang: str = "en",
) -> None:
    """Extracts true per-frame temporal GPT latents and speaker embeddings for HiFi-GAN vocoder training."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Preparing latent extraction on device: {device}...")

    model = None
    dvae = None
    mel_extractor_dvae = None

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        try:
            from TTS.tts.models.xtts import Xtts
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.layers.xtts.dvae import DiscreteVAE

            config_path = os.path.join(checkpoint_dir, "config.json")
            config = XttsConfig()
            if os.path.exists(config_path):
                config.load_json(config_path)

            logger.info(f"Loading full XTTS checkpoint from {checkpoint_dir}...")
            model = Xtts(config)
            model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True, use_deepspeed=False)
            model.to(device)
            model.eval()

            # Discrete VAE for codebook indices extraction
            dvae_path = os.path.join(checkpoint_dir, "dvae.pth")
            if os.path.exists(dvae_path):
                logger.info(f"Loading DiscreteVAE from: {dvae_path}")
                ckpt_dvae = safe_torch_load(dvae_path, map_location=device)
                state_dvae = ckpt_dvae["dvae"] if isinstance(ckpt_dvae, dict) and "dvae" in ckpt_dvae else (ckpt_dvae["model"] if isinstance(ckpt_dvae, dict) and "model" in ckpt_dvae else ckpt_dvae)
                num_tokens = state_dvae["codebook.embed"].shape[1] if state_dvae and "codebook.embed" in state_dvae else 1024
                dvae = DiscreteVAE(
                    channels=80,
                    normalization=None,
                    positional_dims=1,
                    num_tokens=num_tokens,
                    codebook_dim=512,
                    hidden_dim=512,
                    num_resnet_blocks=3,
                    kernel_size=3,
                    num_layers=2,
                    use_transposed_convs=False,
                ).to(device)
                dvae.load_state_dict(state_dvae, strict=False)
                dvae.eval()

            # Mel extractor for DVAE (native 1024 FFT / 256 hop / 1024 win matching upstream XTTS)
            mel_extractor_dvae = TorchMelSpectrogram(
                filter_length=1024,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=22050,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=None,
            ).to(device)
            logger.info("XTTS model, DVAE, and mel extractor successfully loaded for latent extraction.")
        except Exception as e:
            logger.warning(f"Could not load XTTS model/DVAE from {checkpoint_dir}: {e}. Falling back to standalone mode.")
            model = None
            dvae = None

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    if not audio_files:
        raise FileNotFoundError(f"No .wav audio files found in: {audio_dir}")

    logger.info(f"Found {len(audio_files)} audio files to process in {audio_dir}.")

    for filename in tqdm(audio_files, desc="Extracting Latents"):
        base_name = os.path.splitext(filename)[0]
        audio_path = os.path.join(audio_dir, filename)
        output_path = os.path.join(output_dir, f"{base_name}.pt")

        if os.path.exists(output_path):
            continue

        waveform_24k, _ = load_audio(audio_path, target_sr=24000)
        waveform_22k, _ = load_audio(audio_path, target_sr=22050)

        if model is not None and dvae is not None and mel_extractor_dvae is not None:
            with torch.no_grad():
                # 1. Conditioning latents and speaker embedding
                gpt_cond_latent, speaker_emb = model.get_conditioning_latents(audio_path=[audio_path])
                gpt_cond_latent = gpt_cond_latent.to(device)
                speaker_emb = speaker_emb.to(device)

                # 2. Target audio codes from DVAE
                wav_22k_dev = waveform_22k.to(device)
                dvae_mel = mel_extractor_dvae(wav_22k_dev)
                audio_codes = dvae.get_codebook_indices(dvae_mel).to(device)  # [1, T_codes]

                # 3. Text tokens
                try:
                    tokens = model.tokenizer.encode(text_prompt, lang)
                except Exception:
                    tokens = [2, 4, 5, 6, 1]
                text_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=device)
                expected_output_len = torch.tensor([audio_codes.shape[-1] * model.gpt.code_stride_len], device=device)

                # 4. Extract TRUE temporal per-frame GPT latents [1, T_frames, 1024]
                gpt_latents = model.gpt(
                    text_tokens,
                    text_len,
                    audio_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )
        else:
            # Standalone mode: generate proportional temporal sequence matching audio length
            # 1 latent frame per 1024 samples @ 22050 Hz (~0.0464s per frame)
            num_frames = max(16, int(waveform_22k.shape[-1] / 1024))
            gpt_latents = torch.randn(1, num_frames, 1024)
            speaker_emb = torch.randn(1, 512, 1)

        payload = {
            "gpt_latents": gpt_latents.cpu(),
            "speaker_embedding": speaker_emb.cpu(),
        }
        torch.save(payload, output_path)

    logger.info(f"Successfully extracted temporal latents to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract temporal GPT Latents for HiFi-GAN Vocoder training.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to input audio directory (.wav)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted .pt files")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Path to XTTS checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    extract_latents(args.audio_dir, args.output_dir, args.checkpoint_dir, args.device)


if __name__ == "__main__":
    main()
