"""Stage 2 output in the upstream inference format, and the matching loader (docs/recipe_design.md §6, phase 2, F4).

export_for_inference writes model.pth / config.json / vocab.json that stock Xtts.load_checkpoint loads with
strict=True: the trained GPT plus the HiFi-GAN decoder of the base model. load_for_inference loads it the
upstream way, then installs the project tokenizer, because stock VoiceBpeTokenizer raises for any language
registered after release. New languages are recorded in config.json's "languages" list, which upstream's
Xtts.synthesize also checks.
"""

import json
import os
import shutil
from typing import Dict, Optional

import torch

from xtts_trainer.frontend.text import ProjectTokenizer
from xtts_trainer.data.latents import sha256_file
from xtts_trainer.models import safe_torch_load, stamp
from xtts_trainer.spec import XTTS_BASE_LANGUAGES

# Everything stock Xtts.load_checkpoint(strict=True) expects; the DVAE and spectrogram modules stay behind
INFERENCE_PREFIXES = ("gpt.", "hifigan_decoder.", "mel_stats")
DVAE_SHA_KEY = "dvae_sha256"


def _xtts_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = safe_torch_load(path, map_location="cpu")
    weights = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return {k[len("xtts."):] if k.startswith("xtts.") else k: v for k, v in weights.items()}


def export_for_inference(trainer_model, output_dir: str, base_model_path: Optional[str] = None) -> str:
    """Write an inference checkpoint from a ProjectGPTTrainer.

    The decoder comes from base_model_path (Stage 2 doesn't train it); without one, the model's own decoder
    weights are written (only meaningful for tests or a separately trained Stage 3). The DVAE the GPT was
    trained against is copied alongside, so the directory is a complete --checkpoint_dir for Stage 3.
    """
    os.makedirs(output_dir, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in trainer_model.xtts.state_dict().items() if k.startswith(INFERENCE_PREFIXES)}
    if base_model_path:
        base = _xtts_state(base_model_path)
        decoder = {k: v for k, v in base.items() if k.startswith("hifigan_decoder.")}
        expected = {k for k in state if k.startswith("hifigan_decoder.")}
        if set(decoder) != expected:
            raise KeyError(f"{base_model_path} decoder keys differ from the model's: "
                           f"missing {sorted(expected - set(decoder))[:5]}, extra {sorted(set(decoder) - expected)[:5]}")
        state.update(decoder)
    path = os.path.join(output_dir, "model.pth")
    # The DVAE whose codes this GPT was trained on; Stage 3 refuses a retrained DVAE that doesn't match
    torch.save(stamp({"model": state, DVAE_SHA_KEY: sha256_file(trainer_model.config.model_args.dvae_checkpoint)}),
               path + ".tmp")
    os.replace(path + ".tmp", path)

    config = trainer_model.config
    dvae_dst = os.path.join(output_dir, "dvae.pth")
    if os.path.abspath(config.model_args.dvae_checkpoint) != os.path.abspath(dvae_dst):
        shutil.copy2(config.model_args.dvae_checkpoint, dvae_dst)
    vocab_dst = os.path.join(output_dir, "vocab.json")
    if os.path.abspath(config.model_args.tokenizer_file) != os.path.abspath(vocab_dst):
        shutil.copy2(config.model_args.tokenizer_file, vocab_dst)
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsArgs

    args = {f: getattr(config.model_args, f) for f in XttsArgs.__dataclass_fields__}
    args.update(tokenizer_file="vocab.json", gpt_number_text_tokens=trainer_model.xtts.gpt.text_embedding.weight.shape[0])
    inference_config = XttsConfig(model_args=XttsArgs(**args), audio=config.audio, languages=list(config.languages))
    inference_config.save_json(os.path.join(output_dir, "config.json"))
    return path


def export_upstream(base_dir: str, output_dir: str, vocab_path: Optional[str] = None,
                    extra_languages=(), arch_overrides: Optional[dict] = None) -> str:
    """Load a base model through the Stage 2 path and export it without training (docs/recipe_design.md §6,
    phase 2). With the base vocab this is the identity canary's subject: the export must reproduce the base
    model exactly. With an extended vocab it is a vocab-extended model (new rows at the mean) ready to train.
    """
    from xtts_trainer.stage2_gpt.gpt_trainer import ProjectGPTTrainer, build_gpt_trainer_config

    base_model = os.path.join(base_dir, "model.pth")
    config = build_gpt_trainer_config(vocab_path or os.path.join(base_dir, "vocab.json"),
                                      os.path.join(base_dir, "dvae.pth"), output_dir,
                                      extra_languages=extra_languages, arch_overrides=arch_overrides)
    model = ProjectGPTTrainer(config, extra_languages=extra_languages, base_model_path=base_model)
    return export_for_inference(model, output_dir, base_model_path=base_model)


def check_dvae_pairing(checkpoint_dir: str) -> None:
    """Refuse a Stage-1-trained dvae.pth next to a GPT that wasn't trained on its codes (F1).

    The released model.pth records no DVAE, so it pairs only with the released (unmarked) dvae.pth.
    """
    from xtts_trainer.models import dvae_requires_gpt_retrain

    dvae_path = os.path.join(checkpoint_dir, "dvae.pth")
    if not os.path.exists(dvae_path) or not dvae_requires_gpt_retrain(dvae_path):
        return
    recorded = safe_torch_load(os.path.join(checkpoint_dir, "model.pth"), map_location="cpu").get(DVAE_SHA_KEY)
    if recorded != sha256_file(dvae_path):
        raise ValueError(f"{dvae_path} was trained by Stage 1, but the GPT in {checkpoint_dir}/model.pth was not "
                         f"trained on its codes (recorded DVAE: {recorded}); retrain Stage 2 with it first")


def load_for_inference(checkpoint_dir: str, device: str = "cpu"):
    """Stock Xtts loaded strictly from an exported directory, with the project tokenizer installed."""
    from xtts_trainer import patch_xtts_prevent_checkpoint_mutation

    patch_xtts_prevent_checkpoint_mutation()
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))
    model = Xtts(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True, strict=True)
    base = {l.split("-")[0] for l in XTTS_BASE_LANGUAGES}
    extra = [l for l in config.languages if l.split("-")[0] not in base]
    model.tokenizer = ProjectTokenizer(os.path.join(checkpoint_dir, "vocab.json"), extra)
    return model.to(device)
