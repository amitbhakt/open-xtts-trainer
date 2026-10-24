"""Single source of truth for every XTTS-v2 constant used by this project.

Model-level values are read from the vendored upstream config (xtts_v2_config.json, hash-locked in
artifacts.lock.json). Values that upstream hardcodes in code rather than in its config (spectrogram
settings, DVAE architecture, speaker-encoder sample rate) are declared here exactly once, and
tests/test_spec_parity.py checks each of them against the pinned upstream source.

No other module may define STFT parameters, sample rates, stride ratios or model constructor
arguments; tests/test_spec_parity.py enforces that with a source scan.
"""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Tuple

SPEC_DIR = os.path.dirname(os.path.abspath(__file__))
VENDORED_CONFIG_PATH = os.path.join(SPEC_DIR, "xtts_v2_config.json")


@dataclass(frozen=True)
class MelSpec:
    """Parameters of one log-mel front end (see docs/upstream_reference.md, section 0)."""
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    fmin: float
    fmax: float
    divide_by_mel_stats: bool  # True for upstream TorchMelSpectrogram/wav_to_mel_cloning paths


@dataclass(frozen=True)
class DVAEArch:
    """DiscreteVAE constructor arguments used by upstream GPTTrainer."""
    channels: int
    num_tokens: int
    codebook_dim: int
    hidden_dim: int
    num_resnet_blocks: int
    kernel_size: int
    num_layers: int
    positional_dims: int
    use_transposed_convs: bool

    @property
    def downsample_factor(self) -> int:
        """Mel frames per DVAE code (one stride-2 conv per layer)."""
        return 2 ** self.num_layers


@dataclass(frozen=True)
class GPTArch:
    """GPT constructor arguments, as Xtts.init_models builds them from model_args."""
    layers: int
    model_dim: int
    heads: int
    max_text_tokens: int
    max_audio_tokens: int
    max_prompt_tokens: int
    number_text_tokens: int
    num_audio_tokens: int
    start_audio_token: int
    stop_audio_token: int
    code_stride_len: int
    use_perceiver_resampler: bool


@dataclass(frozen=True)
class XttsSpec:
    gpt_sample_rate: int        # model_args.input_sample_rate: GPT, DVAE and conditioning audio
    output_sample_rate: int     # model_args.output_sample_rate: vocoder output
    output_hop_length: int      # model_args.output_hop_length: vocoder samples per upsampled frame
    decoder_input_dim: int
    d_vector_dim: int
    cond_d_vector_in_each_upsampling_layer: bool
    speaker_encoder_sample_rate: int
    languages: Tuple[str, ...]
    dvae_mel: MelSpec
    cond_mel: MelSpec
    vocoder_loss_mel: MelSpec   # owned by this project (Stage 3 training loss), not an upstream contract
    dvae: DVAEArch
    gpt: GPTArch

    def codes_to_samples(self, n_codes: int) -> int:
        """Audio samples at gpt_sample_rate covered by n DVAE codes / GPT latents."""
        return n_codes * self.gpt.code_stride_len

    def max_audio_seconds(self, max_audio_tokens: int) -> float:
        """Longest clip (seconds) whose DVAE codes fit in max_audio_tokens."""
        return self.codes_to_samples(max_audio_tokens) / self.gpt_sample_rate

    @property
    def samples_per_latent(self) -> float:
        """Average vocoder output samples per GPT latent frame (≈1114.56 for XTTS-v2)."""
        return self.gpt.code_stride_len * self.output_sample_rate / self.gpt_sample_rate

    def vocoder_num_samples(self, n_latents: int) -> int:
        """Exact output length of HifiDecoder.forward for n latents (mirrors its two F.interpolate calls)."""
        frames = math.floor(n_latents * (self.gpt.code_stride_len / self.output_hop_length))
        if self.output_sample_rate != self.gpt_sample_rate:
            frames = math.floor(frames * (self.output_sample_rate / self.gpt_sample_rate))
        return frames * self.output_hop_length

    def latent_to_output_sample(self, latent_index: int) -> int:
        """Output-sample offset of a latent frame, for slicing audio aligned with a latent window."""
        return int(round(latent_index * self.samples_per_latent))


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_spec(config_path: str = VENDORED_CONFIG_PATH) -> XttsSpec:
    """Build the spec from an XTTS-v2 config.json (defaults to the vendored, hash-locked copy)."""
    cfg = _load_config(config_path)
    args = cfg["model_args"]
    gpt_sr = args["input_sample_rate"]
    if cfg["audio"]["sample_rate"] != gpt_sr:
        raise ValueError(
            f"config audio.sample_rate ({cfg['audio']['sample_rate']}) != model_args.input_sample_rate ({gpt_sr})"
        )

    # Upstream-hardcoded values; each is checked against the pinned source in tests/test_spec_parity.py.
    dvae_mel = MelSpec(
        sample_rate=gpt_sr, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, fmin=0.0, fmax=8000.0, divide_by_mel_stats=True,
    )
    cond_mel = MelSpec(
        sample_rate=gpt_sr, n_fft=2048, hop_length=256, win_length=1024,
        n_mels=80, fmin=0.0, fmax=8000.0, divide_by_mel_stats=True,
    )
    vocoder_loss_mel = MelSpec(
        sample_rate=args["output_sample_rate"], n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, fmin=0.0, fmax=8000.0, divide_by_mel_stats=False,
    )
    dvae = DVAEArch(
        channels=80,
        num_tokens=args["gpt_num_audio_tokens"] - 2,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        positional_dims=1,
        use_transposed_convs=False,
    )
    gpt = GPTArch(
        layers=args["gpt_layers"],
        model_dim=args["gpt_n_model_channels"],
        heads=args["gpt_n_heads"],
        max_text_tokens=args["gpt_max_text_tokens"],
        max_audio_tokens=args["gpt_max_audio_tokens"],
        max_prompt_tokens=args["gpt_max_prompt_tokens"],
        number_text_tokens=args["gpt_number_text_tokens"],
        num_audio_tokens=args["gpt_num_audio_tokens"],
        start_audio_token=args["gpt_start_audio_token"],
        stop_audio_token=args["gpt_stop_audio_token"],
        code_stride_len=args["gpt_code_stride_len"],
        use_perceiver_resampler=args["gpt_use_perceiver_resampler"],
    )
    spec = XttsSpec(
        gpt_sample_rate=gpt_sr,
        output_sample_rate=args["output_sample_rate"],
        output_hop_length=args["output_hop_length"],
        decoder_input_dim=args["decoder_input_dim"],
        d_vector_dim=args["d_vector_dim"],
        cond_d_vector_in_each_upsampling_layer=args["cond_d_vector_in_each_upsampling_layer"],
        speaker_encoder_sample_rate=16000,
        languages=tuple(cfg["languages"]),
        dvae_mel=dvae_mel,
        cond_mel=cond_mel,
        vocoder_loss_mel=vocoder_loss_mel,
        dvae=dvae,
        gpt=gpt,
    )

    stride_from_mel = dvae_mel.hop_length * dvae.downsample_factor
    if stride_from_mel != gpt.code_stride_len:
        raise ValueError(
            f"DVAE hop ({dvae_mel.hop_length}) x downsample ({dvae.downsample_factor}) = {stride_from_mel} "
            f"does not match gpt_code_stride_len ({gpt.code_stride_len})"
        )
    return spec


def spec_hash(spec: "XttsSpec" = None) -> str:
    """Stable hash of the spec, embedded in every checkpoint and latent file this project writes."""
    spec = spec or load_spec()
    payload = json.dumps(asdict(spec), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


SPEC = load_spec()
