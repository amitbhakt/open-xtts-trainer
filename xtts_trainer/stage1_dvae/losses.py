"""Loss functions for Discrete VAE training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def dvae_loss(
    mel_target: torch.Tensor,
    mel_recon: torch.Tensor,
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates DVAE reconstruction loss and codebook entropy loss.

    Args:
        mel_target: Ground truth log-mel spectrogram [B, n_mels, T].
        mel_recon: Reconstructed log-mel spectrogram [B, n_mels, T].
        logits: Gumbel-Softmax codebook logits [B, num_tokens, T_reduced].

    Returns:
        total_loss, recon_loss, entropy_loss
    """
    # 1. Mel Reconstruction Loss (L1)
    recon_loss = F.l1_loss(mel_recon, mel_target)

    # 2. Codebook Entropy / Uniformity regularization
    # Prevents codebook collapse by encouraging uniform distribution across tokens
    probs = F.softmax(logits, dim=1)
    avg_probs = torch.mean(probs, dim=(0, 2))  # Mean probability per codebook token
    entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-7))

    # Minimize recon loss while maximizing codebook entropy
    total_loss = recon_loss - 0.01 * entropy_loss
    return total_loss, recon_loss, entropy_loss
