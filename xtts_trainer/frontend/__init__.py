"""Audio front ends (spectrogram extractors) shared by every stage."""

from xtts_trainer.frontend.mels import dvae_mel_extractor, cond_mel_extractor, vocoder_loss_mel, LogMel

__all__ = ["dvae_mel_extractor", "cond_mel_extractor", "vocoder_loss_mel", "LogMel"]
