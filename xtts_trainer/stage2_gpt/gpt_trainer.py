"""Stage 2 as a thin wrapper over upstream GPTTrainer (docs/recipe_design.md §6, phase 7).

Upstream owns the loop, prompt sampling, DVAE coding and batching. The project changes only what the
evidence says is wrong, each guarded by a test:
- F2: upstream drops the voice-prompt region from the mel loss for the last sample of each batch only, and
  with the perceiver resampler indexes it in conditioning-mel frames (1/256 of the samples) instead of codes
  (1/1024). The wrapper recomputes the mel loss from upstream's own logits with every sample's prompt masked
  in code units.
- F3: the DVAE is reloaded strictly (exact allowlist) over upstream's non-strict load.
- F4: the project tokenizer (registered languages) replaces VoiceBpeTokenizer.
- F6 / A5: base weights are loaded strictly by the project, with new text rows at the mean of existing rows;
  upstream's checkpoint arguments (non-strict loading, random rows, start-row copy) are never used.
- Data loading failures raise. Upstream's __getitem__ catches every exception and silently trains on sample 1
  instead; the validator has already passed every record, so a failure here is a bug.
"""

import dataclasses
import json
import logging
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig

from xtts_trainer.data import DatasetRecord
from xtts_trainer.frontend.text import ProjectTokenizer
from xtts_trainer.models import (
    dvae_requires_gpt_retrain, load_dvae_state, load_state_strict, resolve_dvae_state, safe_torch_load,
)
from xtts_trainer.spec import DEFAULT_MEL_STATS_PATH, SPEC
from xtts_trainer.spec.xtts_spec import VENDORED_CONFIG_PATH
from xtts_trainer.stage2_gpt.vocab_extender import extend_text_rows

logger = logging.getLogger(__name__)

IGNORE_INDEX = -1


# --------------------------------------------------------------------------- F2: prompt masking

def prompt_code_span(start_sample: int, end_sample: int, code_stride_len: int) -> Tuple[int, int]:
    """Target positions [lo, hi) of the codes overlapping prompt samples [start, end).

    mel_targets[j] is code j (inputs are shifted right by the start token), and code j covers samples
    [j * stride, (j + 1) * stride).
    """
    return start_sample // code_stride_len, math.ceil(end_sample / code_stride_len)


def mel_targets_like_upstream(gpt, audio_codes: torch.Tensor, wav_lengths: torch.Tensor) -> torch.Tensor:
    """The mel targets GPT.forward builds (tts/layers/xtts/gpt.py:322-375, 453-458), from its own helpers."""
    code_lengths = torch.ceil(wav_lengths / gpt.code_stride_len).long() + 3
    max_mel_len = int(code_lengths.max())
    codes = audio_codes.clone()
    if max_mel_len > codes.shape[-1]:
        codes = F.pad(codes, (0, max_mel_len - codes.shape[-1]))
    codes = F.pad(codes[:, :max_mel_len], (0, 1), value=gpt.stop_audio_token)
    codes = gpt.set_mel_padding(codes, code_lengths - 3)
    _, targets = gpt.set_inputs_and_targets(codes, gpt.start_audio_token, gpt.stop_audio_token)
    for idx, length in enumerate(code_lengths):
        targets[idx, length + 1:] = IGNORE_INDEX
    return targets


def mask_prompt_regions(mel_targets: torch.Tensor, cond_idxs: torch.Tensor, code_stride_len: int) -> torch.Tensor:
    """Ignore, for every sample, the targets its own voice prompt was cut from (cond_idxs in samples)."""
    for idx in range(mel_targets.shape[0]):
        lo, hi = prompt_code_span(int(cond_idxs[idx, 0]), int(cond_idxs[idx, 1]), code_stride_len)
        mel_targets[idx, lo:hi] = IGNORE_INDEX
    return mel_targets


def gpt_losses(gpt, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens):
    """(loss_text, loss_mel, mel_logits) with the F2 fix. cond_idxs=None gives upstream's loss exactly."""
    loss_text, loss_mel, mel_logits = gpt(
        text_inputs, text_lengths, audio_codes, wav_lengths,
        cond_mels=cond_mels, cond_idxs=None, cond_lens=cond_lens,
    )
    if cond_idxs is None:
        return loss_text, loss_mel, mel_logits
    targets = mask_prompt_regions(mel_targets_like_upstream(gpt, audio_codes, wav_lengths), cond_idxs,
                                  gpt.code_stride_len)
    loss_mel = F.cross_entropy(mel_logits, targets.long(), ignore_index=IGNORE_INDEX,
                               label_smoothing=gpt.label_smoothing)
    return loss_text, loss_mel, mel_logits


# --------------------------------------------------------------------------- data

class StrictXTTSDataset(XTTSDataset):
    """Upstream's dataset without its silent failure handling.

    Upstream XTTSDataset.__getitem__ wraps load_item in a bare except and returns self[1] instead (recursing
    forever if that fails too), and check_eval_samples drops eval samples that fail to load. Sampling here is
    upstream's: in training, a uniformly random language, then a random clip of it.
    """

    def check_eval_samples(self):
        for sample in self.samples:
            self._checked(sample)

    def _checked(self, sample):
        tseq, audiopath, wav, cond, cond_len, cond_idxs = self.load_item(sample)
        if wav.shape[-1] > self.max_wav_len or tseq.shape[0] > self.max_text_len:
            raise ValueError(f"{sample['audio_file']} exceeds max_wav_length/max_text_length; the validator should have caught it")
        return tseq, audiopath, wav, cond, cond_len, cond_idxs

    def __getitem__(self, index):
        if self.is_eval:
            sample = self.samples[index]
        else:
            lang = random.choice(list(self.samples.keys()))
            sample = self.samples[lang][random.randint(0, len(self.samples[lang]) - 1)]
        try:
            tseq, audiopath, wav, cond, cond_len, cond_idxs = self._checked(sample)
        except Exception as e:
            raise RuntimeError(f"failed to load {sample['audio_file']}; upstream would silently substitute another "
                               f"sample, but the dataset passed validation, so this is a bug to fix") from e
        return {
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long) if cond_len is not torch.nan else torch.tensor([cond_len]),
            "cond_idxs": torch.tensor(cond_idxs) if cond_idxs is not torch.nan else torch.tensor([cond_idxs]),
        }


def records_to_samples(records: Sequence[DatasetRecord], dataset_dir: str) -> List[Dict[str, Any]]:
    """dataset.jsonl records in upstream's sample format."""
    return [{"text": r.text, "audio_file": r.audio_path(dataset_dir), "speaker_name": r.speaker,
             "language": r.lang.lower(), "root_path": dataset_dir} for r in records]


def language_shares(samples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Each language's share of training draws. Upstream picks a language uniformly for every item
    (XTTSDataset.__getitem__), so replay share is set by the number of languages, not by clip counts."""
    langs = sorted({s["language"] for s in samples})
    return {lang: round(1 / len(langs), 4) for lang in langs}


# Parts of the GPT that can be held at their released values during fine-tuning (M1 candidates). The speaker
# conditioning path turns a reference clip into the 32 latents every voice is cloned from; if fine-tuning on one
# speaker moves it, zero-shot cloning of other speakers can suffer.
FREEZABLE_PARTS = {"conditioning": ("conditioning_encoder.", "conditioning_perceiver.")}


def freeze_gpt_parts(gpt: torch.nn.Module, parts: Sequence[str]) -> int:
    """Stop gradients for the named FREEZABLE_PARTS of gpt. Returns the number of frozen parameter tensors."""
    unknown = sorted(set(parts) - set(FREEZABLE_PARTS))
    if unknown:
        raise ValueError(f"unknown parts to freeze {unknown}; choose from {sorted(FREEZABLE_PARTS)}")
    frozen = 0
    for part in dict.fromkeys(parts):  # a part named twice is frozen, and counted, once
        matched = [p for name, p in gpt.named_parameters() if name.startswith(FREEZABLE_PARTS[part])]
        if not matched:
            raise ValueError(f"no parameters match {part!r} ({FREEZABLE_PARTS[part]}); the architecture lacks it")
        for p in matched:
            p.requires_grad_(False)
        frozen += len(matched)
    return frozen


# --------------------------------------------------------------------------- config and model

def build_gpt_trainer_config(
    vocab_path: str,
    dvae_path: str,
    output_path: str,
    *,
    extra_languages: Sequence[str] = (),
    batch_size: int = 4,
    lr: float = 5e-6,
    epochs: int = 10,
    seed: int = 1,
    arch_overrides: Optional[Dict[str, Any]] = None,
    run_name: str = "stage2_gpt",
    num_workers: int = 0,
) -> GPTTrainerConfig:
    """GPTTrainerConfig whose architecture comes from the vendored XTTS-v2 config, never GPTArgs defaults
    (those default to an 8194-token codebook and no perceiver, which don't match the released model).

    Optimizer settings are upstream's recipe (docs/recipe_design.md §3, [E4]); tune against validation loss.
    """
    with open(VENDORED_CONFIG_PATH, "r", encoding="utf-8") as f:
        vendored = json.load(f)
    fields = {f.name for f in dataclasses.fields(GPTArgs)}
    model_args = {k: v for k, v in vendored["model_args"].items() if k in fields}
    model_args.update(arch_overrides or {})
    model_args.update(
        tokenizer_file=vocab_path, dvae_checkpoint=dvae_path, mel_norm_file=DEFAULT_MEL_STATS_PATH,
        xtts_checkpoint="", gpt_checkpoint="",  # base weights are loaded strictly by ProjectGPTTrainer
        gpt_use_masking_gt_prompt_approach=True,
    )
    audio = XttsAudioConfig(sample_rate=SPEC.gpt_sample_rate, dvae_sample_rate=SPEC.gpt_sample_rate,
                            output_sample_rate=SPEC.output_sample_rate)
    languages = list(vendored["languages"]) + [l for l in extra_languages if l not in vendored["languages"]]
    return GPTTrainerConfig(
        output_path=output_path, run_name=run_name, project_name="open-xtts-trainer",
        model_args=GPTArgs(**model_args), audio=audio, languages=languages,
        epochs=epochs, batch_size=batch_size, eval_batch_size=batch_size, num_loader_workers=num_workers,
        num_eval_loader_workers=num_workers, eval_split_max_size=256, print_step=50, plot_step=100, log_model_step=1000,
        save_step=1000, save_n_checkpoints=1, save_checkpoints=True, save_best_after=0,
        dashboard_logger="tensorboard", training_seed=seed, mixed_precision=False,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=lr, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        grad_clip=1.0, test_sentences=[],
    )


def base_gpt_state(model_path: str) -> Dict[str, torch.Tensor]:
    """The GPT tensors of an XTTS model.pth ("gpt." keys, with or without a leading "xtts.")."""
    ckpt = safe_torch_load(model_path, map_location="cpu")
    weights = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {}
    for key, value in weights.items():
        key = key[len("xtts."):] if key.startswith("xtts.") else key
        if key.startswith("gpt."):
            state[key[len("gpt."):]] = value
    if not state:
        raise KeyError(f"{model_path} has no gpt.* tensors")
    return state


class ProjectGPTTrainer(GPTTrainer):
    def __init__(self, config: GPTTrainerConfig, extra_languages: Sequence[str] = (),
                 base_model_path: Optional[str] = None, init_seed: int = 42, retrain_for_new_dvae: bool = False,
                 masking_mode: str = "per_sample"):
        """retrain_for_new_dvae: acknowledge that the DVAE was trained by Stage 1, so its codes no longer mean
        what the pretrained GPT learned (F1) and this run must retrain the GPT on them. Without it, pairing
        such a DVAE with pretrained GPT weights is refused.

        masking_mode: 'per_sample' (default, project fix) masks each sample's voice-prompt region individually in code
        units. 'upstream' reproduces Coqui's buggy masking (cond_idxs passed directly to GPT.forward, masking only the
        last sample's prompt in mel units) for ablation study A3.
        """
        if masking_mode not in ("per_sample", "upstream"):
            raise ValueError(f"Invalid masking_mode '{masking_mode}', must be 'per_sample' or 'upstream'")
        self.masking_mode = masking_mode
        args = config.model_args
        if base_model_path and dvae_requires_gpt_retrain(args.dvae_checkpoint) and not retrain_for_new_dvae:
            raise ValueError(
                f"{args.dvae_checkpoint} was trained by Stage 1, so its codes differ from the ones the pretrained GPT "
                f"in {base_model_path} learned (docs/recipe_design.md F1). Use the released dvae.pth, or pass "
                f"retrain_for_new_dvae / --retrain_for_new_dvae for a full Stage 2 retrain on the new codes."
            )
        self.retrain_for_new_dvae = retrain_for_new_dvae
        if args.xtts_checkpoint or args.gpt_checkpoint:
            raise ValueError("pass base weights as base_model_path; upstream's checkpoint arguments load non-strictly")
        if config.audio.sample_rate != config.audio.dvae_sample_rate:
            raise ValueError("prompt masking assumes cond_idxs and DVAE codes share one sample rate")
        from xtts_trainer import patch_xtts_prevent_checkpoint_mutation

        patch_xtts_prevent_checkpoint_mutation()
        super().__init__(config)
        self.xtts.tokenizer = ProjectTokenizer(args.tokenizer_file, extra_languages)
        # Upstream loaded the DVAE non-strictly; reload with the exact allowlist (F3)
        load_dvae_state(self.dvae, resolve_dvae_state(safe_torch_load(args.dvae_checkpoint, map_location="cpu")),
                        f"DVAE ({args.dvae_checkpoint})")
        if base_model_path:
            state = base_gpt_state(base_model_path)
            state = extend_text_rows(state, self.xtts.gpt.text_embedding.weight.shape[0], seed=init_seed)
            load_state_strict(self.xtts.gpt, state, f"GPT ({base_model_path})")
            logger.info("GPT weights loaded strictly from %s", base_model_path)

    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens):
        if self.masking_mode == "upstream":
            return self.xtts.gpt(
                text_inputs, text_lengths, audio_codes, wav_lengths,
                cond_mels=cond_mels, cond_idxs=cond_idxs, cond_lens=cond_lens,
            )
        return gpt_losses(self.xtts.gpt, text_inputs, text_lengths, audio_codes, wav_lengths,
                          cond_mels, cond_idxs, cond_lens)

    def get_data_loader(self, config, *, is_eval=False, samples=None, verbose=True):
        """Upstream's loader (GPTTrainer.get_data_loader) over StrictXTTSDataset."""
        dataset = StrictXTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            sampler=None if is_eval else self.get_sampler(config, dataset),
            shuffle=False, drop_last=False, collate_fn=dataset.collate_fn, pin_memory=False,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
        )
