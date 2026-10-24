"""Script to extract intermediate temporal GPT latents and speaker embeddings using XTTS v2."""

import argparse
import logging
import os
import torch
from tqdm import tqdm
from typing import List, Optional

from xtts_trainer.config import load_audio
from xtts_trainer.data import load_validated_dataset
from xtts_trainer.data.latents import SYNTHETIC_CHECKPOINT, make_latent_payload, sha256_file
from xtts_trainer.frontend import dvae_mel_extractor
from xtts_trainer.models import load_dvae
from xtts_trainer.spec import SPEC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["COQUI_TOS_AGREED"] = "1"


def latent_filename(audio: str) -> str:
    """Latent file name for a dataset record's audio path (its stem)."""
    return os.path.splitext(os.path.basename(audio))[0] + ".pt"


def extract_latents(
    dataset_path: str,
    output_dir: str,
    checkpoint_dir: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    overwrite: bool = True,
    allow_synthetic_fallback: bool = False,
    extra_languages: Optional[List[str]] = None,
) -> None:
    """Extracts teacher-forced per-frame GPT latents and speaker embeddings for HiFi-GAN vocoder training.

    Every clip in the (validated) dataset is teacher-forced with its own transcript, in eval mode, exactly as
    Xtts.inference slices latents. Each output file records the audio, checkpoint and spec it came from.
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Preparing latent extraction on device: {device}...")

    model = None
    dvae = None
    mel_extractor_dvae = None

    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Specified checkpoint directory does not exist: {checkpoint_dir}")

        # Transcripts are teacher-forced, so the dataset must pass the full (text) validation
        records, dataset_dir, _ = load_validated_dataset(
            dataset_path, os.path.join(checkpoint_dir, "vocab.json"), extra_languages or (), logger=logger
        )

        from xtts_trainer.stage2_gpt.export import check_dvae_pairing, load_for_inference

        check_dvae_pairing(checkpoint_dir)

        # Strict load plus the project tokenizer, so languages registered in Stage 2 tokenize here too (F4)
        logger.info(f"Loading full XTTS checkpoint from {checkpoint_dir}...")
        model = load_for_inference(checkpoint_dir, device)
        model.eval()
        # Latents must come from the eval-mode GPT: no dropout, and eval-mode latent trimming (gpt.py "sub")
        assert not model.gpt.training
        checkpoint_sha = sha256_file(os.path.join(checkpoint_dir, "model.pth"))

        # Discrete VAE for codebook indices extraction
        dvae_path = os.path.join(checkpoint_dir, "dvae.pth")
        if not os.path.exists(dvae_path):
            raise FileNotFoundError(f"Required DVAE weights not found in checkpoint_dir: {dvae_path}")

        logger.info(f"Loading DiscreteVAE from: {dvae_path}")
        dvae = load_dvae(dvae_path, device=device)

        # Mel extractor for DVAE (xtts_trainer.frontend; matches upstream XTTS)
        mel_extractor_dvae = dvae_mel_extractor(device)
        logger.info("XTTS model, DVAE, and mel extractor successfully loaded for latent extraction.")
    else:
        if not allow_synthetic_fallback:
            raise ValueError(
                "Stage 3 latent extraction requires a valid XTTS checkpoint (--checkpoint_dir xtts_base). "
                "Standalone random-latent fallback has been removed to prevent training vocoders on noise."
            )
        logger.warning("Running in synthetic fallback mode for unit testing.")
        records, dataset_dir, _ = load_validated_dataset(dataset_path, None, require_text=False, logger=logger)
        checkpoint_sha = SYNTHETIC_CHECKPOINT

    names = [latent_filename(r.audio) for r in records]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Audio files share a stem, so their latent files would collide: {duplicates}")

    logger.info(f"Extracting latents for {len(records)} clips from {dataset_path}.")

    for record, name in tqdm(list(zip(records, names)), desc="Extracting Latents"):
        audio_path = record.audio_path(dataset_dir)
        output_path = os.path.join(output_dir, name)

        if os.path.exists(output_path) and not overwrite:
            logger.info(f"Skipping existing latent: {output_path}")
            continue

        waveform_gpt, _ = load_audio(audio_path, target_sr=SPEC.gpt_sample_rate)

        if model is not None:
            with torch.no_grad():
                # 1. Conditioning latents and speaker embedding
                gpt_cond_latent, speaker_emb = model.get_conditioning_latents(audio_path=[audio_path])
                gpt_cond_latent = gpt_cond_latent.to(device)
                speaker_emb = speaker_emb.to(device)

                # 2. Target audio codes from DVAE
                dvae_mel = mel_extractor_dvae(waveform_gpt.to(device))
                audio_codes = dvae.get_codebook_indices(dvae_mel).to(device)  # [1, T_codes]

                # 3. Ground-truth text tokens for this clip (validated: supported language, no [UNK])
                tokens = model.tokenizer.encode(record.text.strip(), record.lang.lower())
                text_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=device)
                expected_output_len = torch.tensor([audio_codes.shape[-1] * model.gpt.code_stride_len], device=device)

                # 4. Teacher-forced per-frame GPT latents [1, T_frames, model_dim] (eval-mode slicing)
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
            # Synthetic testing mode only
            num_frames = max(16, waveform_gpt.shape[-1] // SPEC.gpt.code_stride_len)
            gpt_latents = torch.randn(1, num_frames, SPEC.gpt.model_dim)
            speaker_emb = torch.randn(1, SPEC.d_vector_dim, 1)

        payload = make_latent_payload(
            gpt_latents, speaker_emb, audio_path, record.text, record.lang, checkpoint_sha
        )
        torch.save(payload, output_path)

    logger.info(f"Successfully extracted temporal latents to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract temporal GPT Latents for HiFi-GAN Vocoder training.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted .pt files")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to XTTS checkpoint directory (xtts_base)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_overwrite", action="store_true", help="Skip re-extracting existing latents")
    parser.add_argument("--languages", type=str, nargs="+", default=None, help="Registered extra language codes")
    args = parser.parse_args()

    extract_latents(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        overwrite=not args.no_overwrite,
        extra_languages=args.languages,
    )


if __name__ == "__main__":
    main()
