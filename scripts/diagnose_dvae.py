"""Diagnostic script to benchmark base DVAE vs fine-tuned DVAE on held-out audio.

Computes:
1. Mel Reconstruction Error (L1 & MSE)
2. Commitment Loss
3. Codebook Index Diversity (unique tokens used)
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torchaudio
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from xtts_trainer.config import safe_torch_load, load_audio


def load_dvae_model(checkpoint_path: str, device: str = "cpu") -> DiscreteVAE:
    """Instantiate and restore a DiscreteVAE checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = safe_torch_load(checkpoint_path, map_location=device)
    state = ckpt["dvae"] if isinstance(ckpt, dict) and "dvae" in ckpt else (
        ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    )

    num_tokens = state["codebook.embed"].shape[1] if state and "codebook.embed" in state else 1024
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

    missing, unexpected = dvae.load_state_dict(state, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, f"DVAE strict load failed: missing={missing}, unexpected={unexpected}"
    dvae.eval()
    return dvae


def evaluate_dvae(
    dvae: DiscreteVAE,
    mel_extractor: TorchMelSpectrogram,
    audio_paths: list[str],
    device: str = "cpu",
) -> dict:
    """Evaluate DVAE reconstruction metrics on a list of audio files."""
    total_l1 = 0.0
    total_mse = 0.0
    total_commitment = 0.0
    all_codes = []

    with torch.no_grad():
        for path in audio_paths:
            waveform, _ = load_audio(path, target_sr=22050)
            waveform = waveform.to(device)
            mels = mel_extractor(waveform)

            recon_loss, commit_loss, recon_mels = dvae(mels)
            l1_err = F.l1_loss(recon_mels, mels).item()
            mse_err = F.mse_loss(recon_mels, mels).item()
            codes = dvae.get_codebook_indices(mels)

            total_l1 += l1_err
            total_mse += mse_err
            total_commitment += commit_loss.mean().item()
            all_codes.append(codes.flatten().cpu())

    n = max(1, len(audio_paths))
    all_codes_cat = torch.cat(all_codes)
    unique_tokens = len(torch.unique(all_codes_cat))

    return {
        "mel_l1": total_l1 / n,
        "mel_mse": total_mse / n,
        "commitment_loss": total_commitment / n,
        "unique_codes": unique_tokens,
        "total_codes": len(all_codes_cat),
        "codebook_utilization_pct": (unique_tokens / dvae.num_tokens) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned DVAE reconstruction")
    parser.add_argument("--base_dvae", type=str, default="xtts_base/dvae.pth", help="Base pretrained DVAE path")
    parser.add_argument("--trained_dvae", type=str, default="checkpoints/dvae/dvae.pth", help="Fine-tuned DVAE path")
    parser.add_argument("--audio_dir", type=str, default="sample_audio", help="Directory of evaluation audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    audio_files = sorted([os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) if f.endswith(".wav")])
    if not audio_files:
        print(f"No WAV files found in: {args.audio_dir}")
        return

    mel_extractor = TorchMelSpectrogram(
        filter_length=4096,
        hop_length=1024,
        win_length=4096,
        normalize=False,
        sampling_rate=22050,
        mel_fmin=0,
        mel_fmax=8000,
        n_mel_channels=80,
    ).to(args.device)

    print(f"\n=======================================================")
    print(f"DVAE Benchmark on {len(audio_files)} Held-Out Audio Clips ({args.device})")
    print(f"=======================================================")

    results = {}
    if os.path.exists(args.base_dvae):
        print(f"Evaluating Base DVAE: {args.base_dvae}...")
        base_model = load_dvae_model(args.base_dvae, device=args.device)
        results["Base"] = evaluate_dvae(base_model, mel_extractor, audio_files, device=args.device)

    if os.path.exists(args.trained_dvae):
        print(f"Evaluating Fine-Tuned DVAE: {args.trained_dvae}...")
        trained_model = load_dvae_model(args.trained_dvae, device=args.device)
        results["Fine-Tuned"] = evaluate_dvae(trained_model, mel_extractor, audio_files, device=args.device)

    print("\n-------------------------------------------------------")
    print(f"{'Metric':<25} | {'Base DVAE':<15} | {'Fine-Tuned DVAE':<15}")
    print("-------------------------------------------------------")
    for metric in ["mel_l1", "mel_mse", "commitment_loss", "unique_codes", "codebook_utilization_pct"]:
        base_val = f"{results['Base'][metric]:.4f}" if "Base" in results else "N/A"
        trained_val = f"{results['Fine-Tuned'][metric]:.4f}" if "Fine-Tuned" in results else "N/A"
        suffix = "%" if "pct" in metric else ""
        print(f"{metric:<25} | {base_val + suffix:<15} | {trained_val + suffix:<15}")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()
