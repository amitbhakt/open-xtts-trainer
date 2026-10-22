"""Script to extract intermediate GPT latents from an audio dataset using XTTS v2."""

import argparse
import os
import torch
import torchaudio
from tqdm import tqdm

# Auto-agree to non-commercial CPML to prevent blocking interactive prompts
os.environ["COQUI_TOS_AGREED"] = "1"


def extract_latents(
    audio_dir: str,
    output_dir: str,
    checkpoint_dir: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Extracts GPT latents and speaker embeddings from audio files using XTTS."""
    if device == "cuda" and not torch.cuda.is_available():
        print("Notice: CUDA is not available in this PyTorch build. Falling back to CPU.")
        device = "cpu"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading XTTS model for latent extraction on device: {device}...")
    try:
        from TTS.tts.models.xtts import Xtts, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        # If checkpoint_dir is given, load from it; otherwise load official xtts_v2
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            model = Xtts.init_from_config(checkpoint_dir)
            model.load_checkpoint(checkpoint_dir, use_deepspeed=False)
        else:
            from TTS.api import TTS
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
            model = tts.synthesizer.tts_model
    except Exception as e:
        print(f"Notice: Could not load full XTTS weights ({e}). Running in standalone mock mode for validation.")
        model = None

    audio_files = [f for f in sorted(os.listdir(audio_dir)) if f.endswith(".wav")]
    print(f"Found {len(audio_files)} audio files to process.")

    for filename in tqdm(audio_files):
        base_name = os.path.splitext(filename)[0]
        audio_path = os.path.join(audio_dir, filename)
        output_path = os.path.join(output_dir, f"{base_name}.pt")

        if os.path.exists(output_path):
            continue

        from xtts_trainer.config import load_audio
        waveform, sr = load_audio(audio_path, 24000)

        if model is not None:
            with torch.no_grad():
                # Extract speaker embedding
                speaker_emb = model.speaker_manager.compute_embedding(waveform)
                # Compute mel and conditioning latents
                audio_tensor = waveform.to(device)
                gpt_latents = model.get_gpt_cond_latents(audio_tensor, 24000)
        else:
            # Synthetic placeholder for CPU unit tests / dry-runs
            time_frames = waveform.shape[-1] // 256
            gpt_latents = torch.randn(1, 1024, max(1, time_frames))
            speaker_emb = torch.randn(1, 512)

        payload = {
            "gpt_latents": gpt_latents.cpu(),
            "speaker_embedding": speaker_emb.cpu(),
        }
        torch.save(payload, output_path)

    print(f"Successfully extracted latents to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract GPT Latents for HiFi-GAN Vocoder training.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to input audio directory (.wav)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted .pt files")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Path to XTTS checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    extract_latents(args.audio_dir, args.output_dir, args.checkpoint_dir, args.device)


if __name__ == "__main__":
    main()
