"""Phase 2 guards: every spec value matches the pinned upstream source, and nothing redefines it elsewhere.

These tests read upstream code (installed from the pinned coqui-tts commit) rather than trusting values
typed into this repo. If upstream changes under a new pin, they fail and point at the value to re-check.
"""

import ast
import inspect
import json
import os
import re
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torchaudio

from xtts_trainer.frontend import cond_mel_extractor, dvae_mel_extractor
from xtts_trainer.models import build_dvae, build_gpt, build_hifigan
from xtts_trainer.spec import SPEC, DEFAULT_MEL_STATS_PATH, verify_vendored_config
from xtts_trainer.spec import artifacts
from xtts_trainer.spec.xtts_spec import VENDORED_CONFIG_PATH

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _call_keywords(func, callee_name):
    """Constant keyword arguments of every `callee_name(...)` call inside `func`'s source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == callee_name:
            kwargs = {}
            for kw in node.keywords:
                kwargs[kw.arg] = kw.value.value if isinstance(kw.value, ast.Constant) else ast.unparse(kw.value)
            calls.append(kwargs)
    return calls


# ---------------------------------------------------------------------------
# Vendored artifacts
# ---------------------------------------------------------------------------

def test_vendored_artifacts_match_lockfile():
    with open(os.path.join(REPO_ROOT, "artifacts.lock.json"), "r", encoding="utf-8") as f:
        lock = json.load(f)
    assert artifacts.XTTS_CONFIG_SHA256 == lock["config.json"]["sha256"]
    assert artifacts.XTTS_MEL_STATS_SHA256 == lock["mel_stats.pth"]["sha256"]
    assert artifacts.XTTS_MEL_STATS_URL == lock["mel_stats.pth"]["url"]
    verify_vendored_config(VENDORED_CONFIG_PATH)
    assert artifacts.sha256_of_file(DEFAULT_MEL_STATS_PATH) == lock["mel_stats.pth"]["sha256"]


# ---------------------------------------------------------------------------
# Spectrogram front ends
# ---------------------------------------------------------------------------

def test_dvae_mel_matches_upstream_trainer():
    """GPTTrainer builds the DVAE extractor from TorchMelSpectrogram defaults plus mel_norm_file and sample rate."""
    from TTS.tts.configs.xtts_config import XttsAudioConfig
    from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

    dvae_calls = [c for c in _call_keywords(GPTTrainer.__init__, "TorchMelSpectrogram") if "filter_length" not in c]
    assert len(dvae_calls) == 1, dvae_calls
    assert set(dvae_calls[0]) == {"mel_norm_file", "sampling_rate"}

    defaults = {k: p.default for k, p in inspect.signature(TorchMelSpectrogram.__init__).parameters.items()}
    spec = SPEC.dvae_mel
    assert (defaults["filter_length"], defaults["hop_length"], defaults["win_length"]) == (spec.n_fft, spec.hop_length, spec.win_length)
    assert (defaults["n_mel_channels"], defaults["mel_fmin"], defaults["mel_fmax"]) == (spec.n_mels, spec.fmin, spec.fmax)
    assert defaults["normalize"] is False
    assert XttsAudioConfig().dvae_sample_rate == spec.sample_rate


def test_cond_mel_matches_upstream_inference(monkeypatch):
    """Capture the exact wav_to_mel_cloning arguments Xtts inference uses, then compare numerically."""
    import TTS.tts.models.xtts as xtts_module

    captured = {}
    real_fn = xtts_module.wav_to_mel_cloning

    def recorder(wav, **kwargs):
        captured.update(kwargs)
        return real_fn(wav, **kwargs)

    monkeypatch.setattr(xtts_module, "wav_to_mel_cloning", recorder)
    mel_stats = torch.load(DEFAULT_MEL_STATS_PATH, weights_only=True)
    fake_self = SimpleNamespace(
        args=SimpleNamespace(gpt_use_perceiver_resampler=True),
        mel_stats=mel_stats,
        gpt=SimpleNamespace(get_style_emb=lambda mel, *a: torch.zeros(1, SPEC.gpt.model_dim, 32)),
        device="cpu",
    )
    torch.manual_seed(0)
    wav = torch.randn(1, SPEC.gpt_sample_rate) * 0.1
    xtts_module.Xtts.get_gpt_cond_latents(fake_self, wav, SPEC.gpt_sample_rate)

    spec = SPEC.cond_mel
    assert captured["n_fft"] == spec.n_fft
    assert captured["hop_length"] == spec.hop_length
    assert captured["win_length"] == spec.win_length
    assert captured["sample_rate"] == spec.sample_rate
    assert captured["n_mels"] == spec.n_mels
    assert (captured["f_min"], captured["f_max"]) == (spec.fmin, spec.fmax)
    assert captured["mel_norms"] is not None and spec.divide_by_mel_stats

    upstream = real_fn(wav, **captured)
    ours = cond_mel_extractor()(wav)
    assert ours.shape == upstream.shape
    assert torch.allclose(ours, upstream, atol=1e-5)


def test_dvae_mel_applies_mel_stats():
    """The DVAE extractor divides log-mel by mel_stats (upstream recipe), not raw log-mel."""
    from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

    torch.manual_seed(0)
    wav = torch.randn(1, SPEC.gpt_sample_rate) * 0.1
    raw = TorchMelSpectrogram(mel_norm_file=None, sampling_rate=SPEC.dvae_mel.sample_rate)(wav)
    mel_stats = torch.load(DEFAULT_MEL_STATS_PATH, weights_only=True)
    expected = raw / mel_stats.unsqueeze(0).unsqueeze(-1)
    assert torch.allclose(dvae_mel_extractor()(wav), expected, atol=1e-5)


def test_speaker_encoder_sample_rate_matches_upstream(monkeypatch):
    import TTS.tts.models.xtts as xtts_module

    captured = {}

    def fake_resample(audio, orig_freq, new_freq):
        captured["new_freq"] = new_freq
        return audio

    monkeypatch.setattr(torchaudio.functional, "resample", fake_resample)
    encoder = SimpleNamespace(forward=lambda audio, l2_norm: torch.zeros(1, SPEC.d_vector_dim))
    fake_self = SimpleNamespace(hifigan_decoder=SimpleNamespace(speaker_encoder=encoder), device="cpu")
    xtts_module.Xtts.get_speaker_embedding(fake_self, torch.zeros(1, 100), SPEC.gpt_sample_rate)
    assert captured["new_freq"] == SPEC.speaker_encoder_sample_rate


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def test_dvae_arch_matches_upstream_trainer():
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

    calls = _call_keywords(GPTTrainer.__init__, "DiscreteVAE")
    assert len(calls) == 1, calls
    upstream = calls[0]
    arch = SPEC.dvae
    for field in ("channels", "codebook_dim", "hidden_dim", "num_resnet_blocks", "kernel_size",
                  "num_layers", "positional_dims", "use_transposed_convs"):
        assert upstream[field] == getattr(arch, field), field
    assert upstream["normalization"] is None
    assert upstream["num_tokens"] == "self.args.gpt_num_audio_tokens - 2"
    assert arch.num_tokens == SPEC.gpt.num_audio_tokens - 2
    assert build_dvae().num_tokens == arch.num_tokens


def _record_constructor(monkeypatch, module, name):
    calls = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, name, Recorder)
    return calls


def test_gpt_and_hifigan_factory_match_xtts_init_models(monkeypatch):
    """Our factory passes the same constructor arguments as Xtts.init_models with the vendored config."""
    import TTS.tts.layers.xtts.gpt as gpt_module
    import TTS.tts.layers.xtts.hifigan_decoder as hifi_module
    import TTS.tts.models.xtts as xtts_module
    from TTS.tts.configs.xtts_config import XttsConfig

    config = XttsConfig()
    config.load_json(VENDORED_CONFIG_PATH)

    upstream_gpt = _record_constructor(monkeypatch, xtts_module, "GPT")
    upstream_hifi = _record_constructor(monkeypatch, xtts_module, "HifiDecoder")
    fake_self = SimpleNamespace(args=config.model_args, tokenizer=SimpleNamespace(tokenizer=None))
    xtts_module.Xtts.init_models(fake_self)

    ours_gpt = _record_constructor(monkeypatch, gpt_module, "GPT")
    ours_hifi = _record_constructor(monkeypatch, hifi_module, "HifiDecoder")
    build_gpt(number_text_tokens=config.model_args.gpt_number_text_tokens)
    build_hifigan()

    # Text start/stop ids come from the tokenizer at runtime (None in the config), so compare everything else
    skip = {"start_text_token", "stop_text_token"}
    up_gpt = {k: v for k, v in upstream_gpt[0].items() if k not in skip}
    our_gpt = {k: v for k, v in ours_gpt[0].items() if k not in skip}
    assert our_gpt == up_gpt
    assert ours_hifi[0] == upstream_hifi[0]


def test_vocoder_num_samples_matches_hifidecoder():
    decoder = build_hifigan().eval()
    for n in (1, 3, 16, 33):
        with torch.no_grad():
            out = decoder(torch.randn(1, n, SPEC.gpt.model_dim), g=torch.randn(1, SPEC.d_vector_dim, 1))
        assert out.shape[-1] == SPEC.vocoder_num_samples(n), n


def test_code_stride_consistent_with_dvae_front_end():
    assert SPEC.dvae_mel.hop_length * SPEC.dvae.downsample_factor == SPEC.gpt.code_stride_len
    torch.manual_seed(0)
    n_codes = 20
    wav = torch.randn(1, SPEC.codes_to_samples(n_codes)) * 0.1
    codes = build_dvae().get_codebook_indices(dvae_mel_extractor()(wav))
    assert abs(codes.shape[-1] - n_codes) <= 1


# ---------------------------------------------------------------------------
# Single source of truth: nothing outside the spec modules redefines these constants
# ---------------------------------------------------------------------------

SCANNED_DIRS = ("xtts_trainer", "scripts", "eval")
ALLOWED_PATHS = (
    "xtts_trainer/spec/",
    "xtts_trainer/frontend/",
    "xtts_trainer/models/factory.py",
    "xtts_trainer/asr.py",
)
FORBIDDEN_PATTERNS = {
    "spectrogram constructor": re.compile(r"\b(Torch)?MelSpectrogram\("),
    "model constructor": re.compile(r"\b(DiscreteVAE|HifiDecoder|GPT)\("),
    "ASR pipeline": re.compile(r"pipeline\(\s*[\"']automatic-speech-recognition"),
    "sample-rate literal": re.compile(r"\b(22050|24000|16000)\b"),
    "STFT literal": re.compile(r"\b(filter_length|n_fft|hop_length|win_length)\s*=\s*\d"),
    "stride literal": re.compile(r"code_stride_len\s*=\s*\d|\*\s*1024\b|\b1024\s*\*"),
    "mel_stats bypass": re.compile(r"mel_norm_file\s*=\s*None"),
}
# path -> pattern names allowed there, with the reason
EXCEPTIONS = {
    "eval/metrics.py": {"sample-rate literal"},  # ECAPA2/UTMOS fixed 16 kHz input rate, not an XTTS constant
}


def _code_lines(path):
    """Source lines with comments and docstrings blanked out (strings that are code are kept)."""
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                for i in range(first.lineno - 1, first.end_lineno):
                    lines[i] = ""
    return [re.sub(r"(^|\s)#.*$", "", line) for line in lines]


def test_no_spec_literals_outside_spec_modules():
    violations = []
    for top in SCANNED_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(REPO_ROOT, top)):
            if "__pycache__" in dirpath:
                continue
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
                if rel.startswith(ALLOWED_PATHS):
                    continue
                code = "\n".join(_code_lines(path))
                for label, pattern in FORBIDDEN_PATTERNS.items():
                    if label in EXCEPTIONS.get(rel, set()):
                        continue
                    for match in pattern.finditer(code):
                        line_no = code.count("\n", 0, match.start()) + 1
                        violations.append(f"{rel}:{line_no}: {label}: {match.group(0)!r}")
    assert not violations, "Spec values must come from xtts_trainer.spec / frontend / models:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# F3: strict DVAE loading accepts upstream dvae.pth and nothing looser
# ---------------------------------------------------------------------------

def test_dvae_strict_load_allows_only_unsaved_buffers():
    from xtts_trainer.models import DVAE_UNSAVED_BUFFERS, load_dvae_state

    full = build_dvae().state_dict()
    assert DVAE_UNSAVED_BUFFERS <= set(full)

    upstream_like = {k: v for k, v in full.items() if k not in DVAE_UNSAVED_BUFFERS}
    load_dvae_state(build_dvae(), upstream_like, "upstream-like DVAE")

    missing_real_param = dict(upstream_like)
    missing_real_param.pop("codebook.embed")
    with pytest.raises(RuntimeError, match="codebook.embed"):
        load_dvae_state(build_dvae(), missing_real_param, "DVAE missing a parameter")

    with pytest.raises(RuntimeError, match="unexpected"):
        load_dvae_state(build_dvae(), {**full, "extra.weight": torch.zeros(1)}, "DVAE with an extra key")


UPSTREAM_DVAE = os.path.join(REPO_ROOT, "xtts_base", "dvae.pth")


@pytest.mark.skipif(not os.path.exists(UPSTREAM_DVAE), reason="needs xtts_base/dvae.pth (scripts/fetch_base_artifacts.py)")
def test_upstream_dvae_loads():
    from xtts_trainer.models import load_dvae

    dvae = load_dvae(UPSTREAM_DVAE)
    assert dvae.num_tokens == SPEC.dvae.num_tokens
