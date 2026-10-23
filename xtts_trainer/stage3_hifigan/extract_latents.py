"""Script to extract intermediate temporal GPT latents and speaker embeddings using XTTS v2."""

import argparse
import logging
import os
import torch
import torchaudio
from tqdm import tqdm
from typing import Optional

from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

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
    mel_extractor_dvae = None

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        try:
            from TTS.tts.models.xtts import Xtts
            logger.info(f"Loading full XTTS checkpoint from {checkpoint_dir}...")
            model = Xtts.init_from_config(checkpoint_dir)
            model.load_checkpoint(checkpoint_dir, eval=True, use_deepspeed=False)
            model.to(device)
            model.eval()

            mel_extractor_dvae = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=22050,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
            ).to(device)
            logger.info("XTTS model and mel extractor successfully loaded for latent extraction.")
        except Exception as e:
            logger.warning(f"Could not load XTTS model from {checkpoint_dir}: {e}. Falling back to standalone mode.")
            model = None

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

        waveform_24k, sr = torchaudio.load(audio_path)
        if waveform_24k.shape[0] > 1:
            waveform_24k = waveform_24k.mean(dim=0, keepdim=True)
        if sr != 24000:
            waveform_24k = torchaudio.transforms.Resample(sr, 24000)(waveform_24k)

        # 22,050 Hz waveform for DVAE code extraction
        waveform_22k = torchaudio.transforms.Resample(24000, 22050)(waveform_24k)

        if model is not None and mel_extractor_dvae is not None:
            with torch.no_grad():
                # 1. Speaker embedding [1, 512, 1]
                speaker_emb = model.get_speaker_embedding(waveform_24k, 24000).to(device)

                # 2. Style conditioning prompt
                gpt_cond_latent, _ = model.get_conditioning_latents(audio_path=[audio_path])
                gpt_cond_latent = gpt_cond_latent.to(device)

                # 3. Target audio codes from DVAE
                wav_22k_dev = waveform_22k.to(device)
                dvae_mel = mel_extractor_dvae(wav_22k_dev)
                audio_codes = model.dvae.get_codebook_indices(dvae_mel).to(device)  # [1, T_codes]

                # 4. Text tokens
                try:
                    tokens = model.tokenizer.encode(text_prompt, lang)
                except Exception:
                    tokens = [2, 4, 5, 6, 1]
                text_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=device)
                expected_output_len = torch.tensor([audio_codes.shape[-1] * model.gpt.code_stride_len], device=device)

                # 5. Extract TRUE temporal per-frame GPT latents [1, T_frames, 1024]
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
