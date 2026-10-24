"""Training script for Stage 2: XTTS v2 Autoregressive GPT Backbone.

Fine-tunes the autoregressive GPT backbone on paired text and speech with:
- Native XTTS-v2 GPT architecture and discrete DVAE token target construction.
- Dual cross-entropy objectives (text CE + audio mel code CE).
- Extended vocabulary support for new alphabets/scripts.
- Loud validation (zero silent mock fallbacks).
- Checkpoint persistence compatible with the upstream XTTS-v2 inference engine.
"""

import argparse
import json
import logging
import os
import shutil
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any

from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from xtts_trainer.config import safe_torch_load, DEFAULT_MEL_STATS_PATH, ensure_mel_stats, is_language_code
from xtts_trainer.stage2_gpt.vocab_extender import resize_gpt_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["COQUI_TOS_AGREED"] = "1"


class GPTTextAudioDataset(Dataset):
    """Dataset pairing text transcriptions with audio files, encoding target audio into DVAE discrete tokens."""

    def __init__(
        self,
        audio_dir: str,
        metadata_path: str,
        tokenizer: VoiceBpeTokenizer,
        dvae: DiscreteVAE,
        mel_spectrogram_dvae: TorchMelSpectrogram,
        mel_spectrogram_style: TorchMelSpectrogram,
        default_lang: str = "en",
        sample_rate: int = 22050,
        max_text_tokens: int = 400,
        max_audio_tokens: int = 600,
        device: str = "cpu",
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.dvae = dvae
        self.mel_spectrogram_dvae = mel_spectrogram_dvae
        self.mel_spectrogram_style = mel_spectrogram_style
        self.default_lang = default_lang
        self.sample_rate = sample_rate
        self.max_text_tokens = max_text_tokens
        self.max_audio_tokens = max_audio_tokens
        self.device = device
        self.items: List[Tuple[str, str, str, str]] = []  # (audio_path, text, lang, speaker)
        self.speaker_to_indices: Dict[str, List[int]] = {}

        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

        resolved_metadata_path = metadata_path
        if not resolved_metadata_path or not os.path.exists(resolved_metadata_path):
            # Check for standard metadata.csv or metadata.txt in audio_dir
            candidate_csv = os.path.join(audio_dir, "metadata.csv")
            candidate_txt = os.path.join(audio_dir, "metadata.txt")
            if os.path.exists(candidate_csv):
                resolved_metadata_path = candidate_csv
                logger.info(f"Auto-detected metadata file: {resolved_metadata_path}")
            elif os.path.exists(candidate_txt):
                resolved_metadata_path = candidate_txt
                logger.info(f"Auto-detected metadata file: {resolved_metadata_path}")

        raw_items: List[Tuple[str, str, str, str]] = []
        if resolved_metadata_path and os.path.exists(resolved_metadata_path):
            with open(resolved_metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 4:
                        wav_name = parts[0]
                        text = parts[1]
                        speaker = parts[2]
                        lang = parts[3]
                    elif len(parts) == 3:
                        wav_name = parts[0]
                        text = parts[1]
                        p2 = parts[2].strip()
                        if is_language_code(p2):
                            lang = p2.lower()
                            speaker = "default"
                        else:
                            speaker = p2
                            lang = default_lang
                    elif len(parts) >= 2:
                        wav_name = parts[0]
                        text = parts[1]
                        speaker = "default"
                        lang = default_lang
                    else:
                        continue

                    # If speaker is default, infer from filename prefix if available
                    base_stem = os.path.splitext(os.path.basename(wav_name))[0]
                    if speaker == "default":
                        if base_stem.lower().startswith("ljspeech") or base_stem.lower().startswith("lj"):
                            speaker = "ljspeech"
                        elif base_stem.lower().startswith("jfk"):
                            speaker = "jfk"
                        elif "_" in base_stem:
                            speaker = base_stem.split("_")[0]

                    wav_path = os.path.join(audio_dir, wav_name if wav_name.endswith(".wav") else f"{wav_name}.wav")
                    if os.path.exists(wav_path):
                        raw_items.append((wav_path, text, lang, speaker))
                    else:
                        logger.warning(f"Audio file referenced in metadata not found: {wav_path}")
        else:
            raise FileNotFoundError(
                f"No metadata file found in '{audio_dir}' and --metadata_path was not provided. "
                f"Stage 2 GPT training requires ground-truth transcripts paired with each audio clip "
                f"(e.g. metadata.csv with format: audio_file|transcript|lang). Training without true transcripts "
                f"causes the GPT backbone to ignore text representations."
            )

        if not raw_items:
            raise ValueError(f"No valid text-audio pairs found for training in {audio_dir} and {resolved_metadata_path}")

        # Filter clips by duration rather than truncating mid-sentence (prevents premature stop tokens)
        max_duration_sec = (self.max_audio_tokens * 1024) / self.sample_rate
        for item in raw_items:
            wav_path = item[0]
            try:
                info = torchaudio.info(wav_path)
                dur = info.num_frames / info.sample_rate
                if dur > max_duration_sec:
                    logger.warning(
                        f"Skipping {wav_path}: duration ({dur:.2f}s) exceeds max limit ({max_duration_sec:.2f}s). "
                        f"Filtered to prevent mid-sentence truncation."
                    )
                    continue
                self.items.append(item)
            except Exception:
                self.items.append(item)

        if not self.items:
            raise ValueError(
                f"All {len(raw_items)} audio clips exceeded maximum duration limit ({max_duration_sec:.2f}s). "
                f"Please provide shorter speech clips (<= {max_duration_sec:.1f}s) or increase max_audio_tokens."
            )

        # Build speaker index mapping for same-speaker conditioning
        for idx, item in enumerate(self.items):
            spk = item[3]
            if spk not in self.speaker_to_indices:
                self.speaker_to_indices[spk] = []
            self.speaker_to_indices[spk].append(idx)

        logger.info(
            f"Loaded {len(self.items)} text-audio training pairs across {len(self.speaker_to_indices)} speaker(s)."
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        import random
        wav_path, text, lang, speaker = self.items[index]

        # 1. Load and resample target audio to 22050 Hz
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Ensure minimum duration (~0.5s)
        min_samples = int(0.5 * self.sample_rate)
        if waveform.shape[-1] < min_samples:
            waveform = nn.functional.pad(waveform, (0, min_samples - waveform.shape[-1]))

        # 2. Extract DVAE discrete codes from target audio
        with torch.no_grad():
            wav_dvae = waveform.to(self.device)
            dvae_mel = self.mel_spectrogram_dvae(wav_dvae)
            audio_codes = self.dvae.get_codebook_indices(dvae_mel).squeeze(0).cpu()  # [T_codes]
            if audio_codes.shape[-1] > self.max_audio_tokens:
                raise ValueError(
                    f"Audio clip {wav_path} yielded {audio_codes.shape[-1]} codes, exceeding max {self.max_audio_tokens}. "
                    f"Mid-sentence truncation is prohibited; ensure audio duration matches max_audio_tokens."
                )

            # 3. Extract speaker conditioning mel from a SEPARATE clip of the SAME speaker
            same_spk_indices = self.speaker_to_indices.get(speaker, [index])
            candidate_indices = [i for i in same_spk_indices if i != index]
            ref_index = random.choice(candidate_indices) if candidate_indices else index
            ref_path = self.items[ref_index][0]

            ref_waveform, ref_sr = torchaudio.load(ref_path)
            if ref_waveform.shape[0] > 1:
                ref_waveform = ref_waveform.mean(dim=0, keepdim=True)
            if ref_sr != self.sample_rate:
                ref_resampler = torchaudio.transforms.Resample(ref_sr, self.sample_rate)
                ref_waveform = ref_resampler(ref_waveform)

            max_ref_samples = int(6.0 * self.sample_rate)
            ref_slice = ref_waveform[:, :max_ref_samples].to(self.device)
            cond_mel = self.mel_spectrogram_style(ref_slice).squeeze(0).cpu()  # [80, T_cond_mel]

        # 4. Tokenize text using XTTS VoiceBpeTokenizer
        try:
            tokens = self.tokenizer.encode(text, lang)
        except Exception:
            # Language token prefix handling for extended scripts
            lang_token = f"[{lang}]"
            vocab = self.tokenizer.tokenizer.get_vocab()
            if lang_token in vocab:
                txt = f"{lang_token}{text}".replace(" ", "[SPACE]")
                tokens = self.tokenizer.tokenizer.encode(txt).ids
            else:
                txt = f"[{self.default_lang}]{text}".replace(" ", "[SPACE]")
                tokens = self.tokenizer.tokenizer.encode(txt).ids

        text_tokens = torch.tensor(tokens, dtype=torch.long)
        if len(text_tokens) > self.max_text_tokens:
            text_tokens = text_tokens[: self.max_text_tokens]

        return {
            "text_tokens": text_tokens,
            "audio_codes": audio_codes,
            "cond_mel": cond_mel,
        }


def collate_gpt_batch(batch: List[Dict[str, torch.Tensor]], stop_audio_token: int = 8193) -> Dict[str, torch.Tensor]:
    """Pads text and discrete audio codes into a batch, cropping conditioning mels to uniform batch length."""
    batch_size = len(batch)

    # Text padding
    text_lengths = torch.tensor([len(item["text_tokens"]) for item in batch], dtype=torch.long)
    max_text_len = max(1, text_lengths.max().item())
    text_inputs = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    for i, item in enumerate(batch):
        text_inputs[i, : text_lengths[i]] = item["text_tokens"]

    # Audio codes padding (padded with stop_audio_token)
    code_lengths = torch.tensor([item["audio_codes"].shape[-1] for item in batch], dtype=torch.long)
    max_code_len = max(1, code_lengths.max().item())
    audio_codes = torch.full((batch_size, max_code_len), fill_value=stop_audio_token, dtype=torch.long)
    for i, item in enumerate(batch):
        audio_codes[i, : code_lengths[i]] = item["audio_codes"]

    # Audio wav length proxy (expected output audio length = code_length * 1024)
    wav_lengths = code_lengths * 1024

    # Conditioning mel: crop all items in batch to the minimum length in the batch.
    # This guarantees ZERO zero-padding in cond_mels, so the perceiver resampler never attends to loud log-mel zeros.
    min_cond_len = min(item["cond_mel"].shape[-1] for item in batch)
    cond_mels = torch.stack([item["cond_mel"][:, :min_cond_len].unsqueeze(0) for item in batch], dim=0)

    return {
        "text_inputs": text_inputs,
        "text_lengths": text_lengths,
        "audio_codes": audio_codes,
        "wav_lengths": wav_lengths,
        "cond_mels": cond_mels,
    }


def build_gpt_model(
    vocab_size: int,
    start_text_token: int = 261,
    stop_text_token: int = 0,
    layers: int = 30,
    model_dim: int = 1024,
    heads: int = 16,
    num_audio_tokens: int = 1026,
    start_audio_token: int = 1024,
    stop_audio_token: int = 1025,
    use_perceiver_resampler: bool = True,
) -> GPT:
    """Builds native XTTS-v2 autoregressive GPT module (with Perceiver Resampler for XTTS-v2)."""
    return GPT(
        layers=layers,
        model_dim=model_dim,
        start_text_token=start_text_token,
        stop_text_token=stop_text_token,
        heads=heads,
        max_text_tokens=402,
        max_mel_tokens=605,
        max_prompt_tokens=70,
        number_text_tokens=vocab_size,
        num_audio_tokens=num_audio_tokens,
        start_audio_token=start_audio_token,
        stop_audio_token=stop_audio_token,
        use_perceiver_resampler=use_perceiver_resampler,
        code_stride_len=1024,
    )


def train_gpt(
    audio_dir: str,
    output_dir: str,
    metadata_path: str = "",
    vocab_path: str = "",
    checkpoint_dir: str = "",
    dvae_checkpoint: str = "",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 5e-6,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    gpt_layers: int = 30,
    gpt_model_dim: int = 1024,
    gpt_heads: int = 16,
) -> str:
    """Executes Stage 2 GPT fine-tuning."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False. "
            "Please switch your Colab runtime to GPU (Runtime -> Change runtime type -> T4 GPU)."
        )

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting Stage 2 GPT fine-tuning on device: {device}")

    # 1. Resolve & Load Tokenizer
    if not vocab_path:
        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "vocab.json")):
            vocab_path = os.path.join(checkpoint_dir, "vocab.json")
        else:
            raise FileNotFoundError("Missing vocab_path: specify path to vocab.json or extended vocabulary.")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    vocab_size = tokenizer.get_number_tokens()

    start_token_id = tokenizer.tokenizer.token_to_id("[START]")
    start_text_token = 261 if start_token_id is None else start_token_id

    stop_token_id = tokenizer.tokenizer.token_to_id("[STOP]")
    stop_text_token = 0 if stop_token_id is None else stop_token_id
    logger.info(f"Loaded tokenizer with {vocab_size} tokens (start: {start_text_token}, stop: {stop_text_token}).")

    # 2. Initialize Mel Spectrogram Extractors (22,050 Hz matching upstream XTTS)
    # DVAE extractor: 1024 FFT / 256 hop / 1024 win normalized with mel_stats
    mel_stats_verified_path = ensure_mel_stats(DEFAULT_MEL_STATS_PATH)
    mel_extractor_dvae = TorchMelSpectrogram(
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        normalize=False,
        sampling_rate=22050,
        mel_fmin=0,
        mel_fmax=8000,
        n_mel_channels=80,
        mel_norm_file=mel_stats_verified_path,
    ).to(device)

    # Style/Perceiver extractor: 2048 FFT / 256 hop / 1024 win normalized with mel_stats (parity with inference)
    mel_extractor_style = TorchMelSpectrogram(
        filter_length=2048,
        hop_length=256,
        win_length=1024,
        normalize=False,
        sampling_rate=22050,
        mel_fmin=0,
        mel_fmax=8000,
        n_mel_channels=80,
        mel_norm_file=mel_stats_verified_path,
    ).to(device)

    # 3. Initialize DVAE Tokenizer (for target audio code extraction)
    dvae_path = dvae_checkpoint or (os.path.join(checkpoint_dir, "dvae.pth") if checkpoint_dir else "")
    dvae_tokens = 1024
    state_dvae = None
    if dvae_path and os.path.exists(dvae_path):
        logger.info(f"Loading pretrained DVAE from: {dvae_path}")
        ckpt_dvae = safe_torch_load(dvae_path, map_location=device)
        state_dvae = ckpt_dvae["dvae"] if isinstance(ckpt_dvae, dict) and "dvae" in ckpt_dvae else (ckpt_dvae["model"] if isinstance(ckpt_dvae, dict) and "model" in ckpt_dvae else ckpt_dvae)
        if state_dvae and "codebook.embed" in state_dvae:
            dvae_tokens = state_dvae["codebook.embed"].shape[1]
            logger.info(f"Inferred DVAE codebook tokens: {dvae_tokens}")

    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=dvae_tokens,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    ).to(device)

    if state_dvae is not None:
        missing, unexpected = dvae.load_state_dict(state_dvae, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"DVAE strict load failed: missing={missing}, unexpected={unexpected}")
        logger.info(f"DVAE restored with strict=True (100% parameter match)")
    else:
        logger.info(f"Using initialized DVAE with {dvae_tokens} tokens.")

    if dvae_checkpoint and "xtts_base" not in dvae_checkpoint and checkpoint_dir and "xtts_base" in checkpoint_dir:
        logger.warning(
            "CRITICAL WARNING: Fine-tuned DVAE loaded with stock pretrained GPT checkpoint! "
            "Because DVAE fine-tuning changes codebook token semantics, ensure Stage 2 is trained "
            "for sufficient epochs so the GPT backbone re-learns discrete token mappings."
        )

    dvae.eval()

    num_audio_tokens = dvae_tokens + 2
    start_audio_token = dvae_tokens
    stop_audio_token = dvae_tokens + 1

    # 4. Inspect Pretrained GPT Checkpoint to Determine Initial Architecture Shapes
    base_model_weights = {}
    gpt_state = None
    init_vocab_size = vocab_size

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        model_path = os.path.join(checkpoint_dir, "model.pth")
        if os.path.exists(model_path):
            logger.info(f"Loading pretrained XTTS GPT weights from {model_path}...")
            ckpt = safe_torch_load(model_path, map_location="cpu")
            raw_weights = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            base_model_weights = raw_weights

            # Filter non-GPT keys and known extraneous buffers
            ignore_prefixes = (
                "torch_mel_spectrogram_style_encoder",
                "torch_mel_spectrogram_dvae",
                "dvae",
                "hifigan_decoder",
                "mel_stats",
            )
            allowed_missing_patterns = ("attn.bias",)

            gpt_state = {}
            for k, v in raw_weights.items():
                clean_k = k[5:] if k.startswith("xtts.") else k
                if any(clean_k.startswith(p) for p in ignore_prefixes):
                    continue
                if clean_k.startswith("gpt."):
                    gpt_state[clean_k[4:]] = v
                else:
                    gpt_state[clean_k] = v

            # If checkpoint has text_embedding, initialize GPT with checkpoint's base vocab size
            if "text_embedding.weight" in gpt_state:
                orig_vocab_size = gpt_state["text_embedding.weight"].shape[0]
                if orig_vocab_size != vocab_size:
                    logger.info(f"Pretrained checkpoint vocab size is {orig_vocab_size}. Building base model at {orig_vocab_size} before expansion.")
                    init_vocab_size = orig_vocab_size

    # Check if config.json specifies use_perceiver_resampler
    use_perceiver_resampler = True
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        try:
            with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                use_perceiver_resampler = cfg_data.get("model_args", {}).get("gpt_use_perceiver_resampler", True)
        except Exception:
            pass

    # 5. Initialize GPT Model
    gpt = build_gpt_model(
        vocab_size=init_vocab_size,
        start_text_token=start_text_token,
        stop_text_token=stop_text_token,
        layers=gpt_layers,
        model_dim=gpt_model_dim,
        heads=gpt_heads,
        num_audio_tokens=num_audio_tokens,
        start_audio_token=start_audio_token,
        stop_audio_token=stop_audio_token,
        use_perceiver_resampler=use_perceiver_resampler,
    ).to(device)

    # 6. Load Pretrained Weights with Strict Validation and Expand Vocabulary
    if gpt_state is not None:
        missing, unexpected = gpt.load_state_dict(gpt_state, strict=False)

        # Fail loudly on any unexpected keys
        if unexpected:
            raise RuntimeError(f"Unexpected non-allowlisted keys in GPT checkpoint: {unexpected}")

        # Fail loudly on any missing keys that are not runtime causal mask buffers
        unresolved_missing = [k for k in missing if not any(pat in k for pat in allowed_missing_patterns)]
        if unresolved_missing:
            raise RuntimeError(f"Required GPT parameter tensors missing from checkpoint: {unresolved_missing}")

        logger.info(
            f"GPT base weights verified and loaded successfully ({len(gpt_state)} tensors, missing runtime buffers: {len(missing)})."
        )

        if vocab_size > init_vocab_size:
            logger.info(f"Expanding GPT text embeddings: {init_vocab_size} -> {vocab_size}")
            resize_gpt_embeddings(gpt, vocab_size)

    # 6. Prepare Dataset & DataLoader
    dataset = GPTTextAudioDataset(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        dvae=dvae,
        mel_spectrogram_dvae=mel_extractor_dvae,
        mel_spectrogram_style=mel_extractor_style,
        device=device,
    )
    effective_batch_size = max(1, min(batch_size, len(dataset)))
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_gpt_batch,
    )

    optimizer = torch.optim.AdamW(
        gpt.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.96),
        weight_decay=0.01,
    )

    # Copy vocab.json to output directory for XTTS inference compatibility
    dest_vocab = os.path.join(output_dir, "vocab.json")
    if os.path.abspath(vocab_path) != os.path.abspath(dest_vocab):
        shutil.copy2(vocab_path, dest_vocab)

    logger.info(f"Beginning GPT training for {epochs} epochs (Batch size: {batch_size}, Steps per epoch: {len(dataloader)})...")
    last_ckpt_path = ""

    for epoch in range(1, epochs + 1):
        gpt.train()
        total_loss = 0.0
        total_text_ce = 0.0
        total_mel_ce = 0.0

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            text_inputs = batch["text_inputs"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            audio_codes = batch["audio_codes"].to(device)
            wav_lengths = batch["wav_lengths"].to(device)
            cond_mels = batch["cond_mels"].to(device)

            # Native autoregressive forward pass (cond_mels is uniformly cropped with zero padding)
            loss_text, loss_mel, _ = gpt(
                text_inputs=text_inputs,
                text_lengths=text_lengths,
                audio_codes=audio_codes,
                wav_lengths=wav_lengths,
                cond_mels=cond_mels,
            )

            loss = loss_text * 0.01 + loss_mel * 1.0

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered at step {step}: total={loss.item()}, mel_ce={loss_mel.item()}, text_ce={loss_text.item()}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_text_ce += loss_text.item()
            total_mel_ce += loss_mel.item()

            if (step + 1) % max(1, len(dataloader) // 5) == 0 or (step + 1) == len(dataloader):
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f} (Mel CE: {loss_mel.item():.4f}, Text CE: {loss_text.item():.4f})"
                )

        avg_loss = total_loss / len(dataloader)
        avg_mel_ce = total_mel_ce / len(dataloader)
        avg_text_ce = total_text_ce / len(dataloader)
        logger.info(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} (Mel: {avg_mel_ce:.4f}, Text: {avg_text_ce:.4f})")

        # 7. Save Checkpoint in XTTS Compatible Format using Atomic Writes
        last_ckpt_path = os.path.join(output_dir, f"gpt_epoch_{epoch}.pth")
        gpt_state_dict = gpt.state_dict()

        # Build full state dict if base decoder weights exist
        combined_model_dict = {}
        if base_model_weights:
            for k, v in base_model_weights.items():
                if not k.startswith("gpt."):
                    combined_model_dict[k] = v
        for k, v in gpt_state_dict.items():
            combined_model_dict[f"gpt.{k}"] = v.cpu()

        epoch_payload = {
            "model": combined_model_dict,
            "gpt": {k: v.cpu() for k, v in gpt_state_dict.items()},
            "epoch": epoch,
            "vocab_size": vocab_size,
            "optimizer": optimizer.state_dict(),
        }

        # Atomic write for epoch checkpoint
        tmp_epoch_path = f"{last_ckpt_path}.tmp"
        torch.save(epoch_payload, tmp_epoch_path)
        os.replace(tmp_epoch_path, last_ckpt_path)

        # Also maintain a latest model.pth for seamless XTTS inference loading
        latest_model_path = os.path.join(output_dir, "model.pth")
        latest_payload = {
            "model": combined_model_dict,
            "epoch": epoch,
            "vocab_size": vocab_size,
        }
        tmp_latest_path = f"{latest_model_path}.tmp"
        torch.save(latest_payload, tmp_latest_path)
        os.replace(tmp_latest_path, latest_model_path)

        # 8. Save updated config.json with updated gpt_number_text_tokens
        config_out_path = os.path.join(output_dir, "config.json")
        base_config_path = os.path.join(checkpoint_dir, "config.json") if checkpoint_dir else ""
        cfg_dict = {}
        if base_config_path and os.path.exists(base_config_path):
            try:
                with open(base_config_path, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read base config from {base_config_path}: {e}")
        if "model_args" not in cfg_dict:
            cfg_dict["model_args"] = {}
        cfg_dict["model_args"]["gpt_number_text_tokens"] = vocab_size
        cfg_dict["model_args"]["gpt_layers"] = gpt_layers
        cfg_dict["model_args"]["gpt_n_model_channels"] = gpt_model_dim
        cfg_dict["model_args"]["gpt_n_heads"] = gpt_heads
        cfg_dict["model_args"]["gpt_use_perceiver_resampler"] = use_perceiver_resampler
        cfg_dict["model_args"]["gpt_num_audio_tokens"] = num_audio_tokens
        cfg_dict["model_args"]["gpt_start_audio_token"] = start_audio_token
        cfg_dict["model_args"]["gpt_stop_audio_token"] = stop_audio_token
        cfg_dict["model_args"]["gpt_code_stride_len"] = 1024
        tmp_cfg_path = f"{config_out_path}.tmp"
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
        os.replace(tmp_cfg_path, config_out_path)

    logger.info(f"Stage 2 GPT training finished. Final model saved to: {output_dir}")
    return last_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stage 2 GPT Backbone for XTTS v2.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing .wav audio files")
    parser.add_argument("--metadata_path", type=str, default="", help="Path to metadata.csv (file.wav|transcript)")
    parser.add_argument("--vocab_path", type=str, default="", help="Path to vocab.json (or extended vocab)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/gpt")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Path to base XTTS checkpoint directory")
    parser.add_argument("--dvae_checkpoint", type=str, default="", help="Path to dvae.pth checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_gpt(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        vocab_path=args.vocab_path,
        checkpoint_dir=args.checkpoint_dir,
        dvae_checkpoint=args.dvae_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
