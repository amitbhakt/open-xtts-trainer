"""Phase 5: static invariants, the alignment invariant, overfit canaries, and model-gated canaries.

Static scans fail the build on patterns that caused past audit findings (silent fallbacks, random data in
data paths, loose loading, placeholder transcripts). Each scanner is itself tested on planted bad code, so a
scanner that can't fail is caught too.
"""

import ast
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import soundfile as sf
import torch

from xtts_trainer.spec import SPEC

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNED_DIRS = ("xtts_trainer", "scripts", "eval")
XTTS_BASE = os.path.join(REPO_ROOT, "xtts_base")
HAS_BASE_MODEL = all(os.path.exists(os.path.join(XTTS_BASE, f)) for f in ("model.pth", "dvae.pth", "config.json", "vocab.json"))
needs_base_model = pytest.mark.skipif(not HAS_BASE_MODEL, reason="needs xtts_base/ (scripts/fetch_base_artifacts.py)")


# ---------------------------------------------------------------------------
# Static invariants
# ---------------------------------------------------------------------------

def _python_files(roots):
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for name in filenames:
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)
                    yield path, os.path.relpath(path, REPO_ROOT if root.startswith(REPO_ROOT) else os.path.dirname(root))


def find_swallowed_broad_excepts(roots):
    """`except:` / `except Exception` / `except BaseException` whose body never re-raises."""
    hits = []
    for path, rel in _python_files(roots):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            types = [node.type] if not isinstance(node.type, ast.Tuple) else list(node.type.elts)
            broad = node.type is None or any(isinstance(t, ast.Name) and t.id in ("Exception", "BaseException") for t in types)
            reraises = any(isinstance(n, ast.Raise) for stmt in node.body for n in ast.walk(stmt))
            if broad and not reraises:
                hits.append(f"{rel}:{node.lineno}")
    return hits


def find_pattern(roots, pattern, allowed=()):
    hits = []
    for path, rel in _python_files(roots):
        for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(keepends=True), start=1):
            code = re.sub(r"(^|\s)#.*$", "", line)
            if re.search(pattern, code) and rel.replace(os.sep, "/") not in allowed:
                hits.append(f"{rel}:{i}: {line.strip()}")
    return hits


RANDOM_DATA = r"\btorch\.(randn|rand|randint)\(|\bnp\.random\.(randn|rand|normal|uniform|randint)\("
# path -> why random tensors are legitimate there
RANDOM_DATA_ALLOWED = {
    "xtts_trainer/stage2_gpt/vocab_extender.py": "seeded noise for new embedding rows",
    "xtts_trainer/stage3_hifigan/extract_latents.py": "synthetic mode, unit tests only; latents are marked synthetic and refused by training",
    "xtts_trainer/stage1_dvae/dataset.py": "random crop offset (an index into real audio, not data)",
    "xtts_trainer/stage3_hifigan/dataset.py": "random segment offsets (indices into real audio, not data)",
}
STRICT_FALSE_ALLOWED = {
    "xtts_trainer/models/checkpoint_io.py": "load_state_strict itself, which checks missing/unexpected keys",
}
# Fake transcripts and fabricated-baseline wording that earlier versions shipped
PLACEHOLDER_TEXT = r"(?i)clip number|speech prompt|sample speech|conditioning alignment prompt|calibrated baseline|clean human speech clip"


def _roots():
    return [os.path.join(REPO_ROOT, d) for d in SCANNED_DIRS]


def test_no_swallowed_broad_excepts():
    hits = find_swallowed_broad_excepts(_roots())
    assert not hits, "Broad except blocks that swallow errors (catch specific exceptions or re-raise):\n" + "\n".join(hits)


def test_no_random_data_in_data_paths():
    hits = find_pattern(_roots(), RANDOM_DATA, RANDOM_DATA_ALLOWED)
    assert not hits, "Random tensors outside the allowlist:\n" + "\n".join(hits)


def test_no_loose_state_dict_loading():
    hits = find_pattern(_roots(), r"strict\s*=\s*False", STRICT_FALSE_ALLOWED)
    assert not hits, "strict=False outside checkpoint_io.load_state_strict:\n" + "\n".join(hits)


def test_no_placeholder_transcripts():
    hits = find_pattern(_roots(), PLACEHOLDER_TEXT)
    assert not hits, "Placeholder transcript text found:\n" + "\n".join(hits)


def test_scanners_catch_planted_violations(tmp_path):
    """Mutation check: each scanner must flag code written to violate it."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "bad.py").write_text(
        "import torch, numpy as np\n"
        "try:\n    x = 1\nexcept Exception:\n    pass\n"
        "try:\n    y = 1\nexcept:\n    y = 2\n"
        "try:\n    z = 1\nexcept Exception:\n    raise\n"
        "fake = torch.randn(1, 10)\n"
        "noise = np.random.normal(0, 1)\n"
        "model.load_state_dict(sd, strict=False)\n"
        "text = 'Training speech prompt 1.'\n",
        encoding="utf-8",
    )
    roots = [str(pkg)]
    assert len(find_swallowed_broad_excepts(roots)) == 2  # the re-raising one is fine
    assert len(find_pattern(roots, RANDOM_DATA)) == 2
    assert len(find_pattern(roots, r"strict\s*=\s*False")) == 1
    assert len(find_pattern(roots, PLACEHOLDER_TEXT)) == 1


# ---------------------------------------------------------------------------
# Architecture parity against the upstream checkpoints' recorded index
# ---------------------------------------------------------------------------

INDEX_PATH = os.path.join(REPO_ROOT, "tests", "fixtures", "upstream_checkpoint_index.json")


def _upstream_tensors(name, prefix=""):
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        tensors = json.load(f)["pinned_files"][name]["tensors"]
    return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}


def _shapes(module):
    return {k: list(v.shape) for k, v in module.state_dict().items()}


def test_index_fixture_matches_lockfile():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)["pinned_files"]
    with open(os.path.join(REPO_ROOT, "artifacts.lock.json"), "r", encoding="utf-8") as f:
        lock = json.load(f)
    for name, entry in index.items():
        assert entry["locked_sha256"] == lock[name]["sha256"], f"{name} index recorded from a different file"


def test_gpt_factory_matches_upstream_checkpoint():
    from xtts_trainer.models import build_gpt

    with torch.device("meta"):
        gpt = build_gpt(number_text_tokens=SPEC.gpt.number_text_tokens)
    assert _shapes(gpt) == _upstream_tensors("model.pth", "gpt.")


def test_hifigan_factory_matches_upstream_checkpoint():
    from xtts_trainer.models import build_hifigan

    assert _shapes(build_hifigan()) == _upstream_tensors("model.pth", "hifigan_decoder.")


def test_dvae_factory_matches_upstream_checkpoint():
    from xtts_trainer.models import DVAE_UNSAVED_BUFFERS, build_dvae

    ours = {k: v for k, v in _shapes(build_dvae()).items() if k not in DVAE_UNSAVED_BUFFERS}
    assert ours == _upstream_tensors("dvae.pth")


# ---------------------------------------------------------------------------
# Alignment invariant: the Stage 3 dataset slices audio exactly where HifiDecoder puts each latent
# ---------------------------------------------------------------------------

class _RepeatFrames(torch.nn.Module):
    """Stands in for the HiFi-GAN generator: each upsampled frame becomes output_hop_length samples."""

    def forward(self, z, g=None):
        if z.dim() == 2:  # upstream HifiDecoder squeezes the batch dim when B == 1
            z = z.unsqueeze(0)
        return z[:, :1, :].repeat_interleave(SPEC.output_hop_length, dim=-1)


@pytest.mark.parametrize("window_start", [0, 7, 21, 40])
def test_stage3_segments_align_with_decoder(tmp_path, monkeypatch, window_start):
    """Each latent frame carries its own index as its value, and the clip's audio is the decoder's output for
    the whole clip. Decoding the dataset's frame window must reproduce the dataset's audio window exactly:
    the decoder input is interpolated over the whole clip before slicing, and each frame is exactly
    output_hop_length samples."""
    from dataset_helpers import write_test_dataset
    from xtts_trainer.data import load_dataset
    from xtts_trainer.data.latents import SYNTHETIC_CHECKPOINT, make_latent_payload
    from xtts_trainer.models import build_hifigan
    from xtts_trainer.stage3_hifigan.dataset import LatentHiFiGANDataset

    n_latents, window = 64, 32
    decoder = build_hifigan().eval()
    decoder.waveform_decoder = _RepeatFrames()
    ramp = (torch.arange(n_latents, dtype=torch.float32) / n_latents + 0.05)[None, :, None].repeat(1, 1, SPEC.gpt.model_dim)
    with torch.no_grad():
        full_audio = decoder(ramp).reshape(-1)
    assert full_audio.shape[-1] == SPEC.vocoder_num_samples(n_latents)

    wav_path = str(tmp_path / "ramp.wav")
    sf.write(wav_path, full_audio.numpy(), SPEC.output_sample_rate, subtype="FLOAT")
    payload = make_latent_payload(ramp, torch.zeros(1, SPEC.d_vector_dim, 1), wav_path, "ramp", "en", SYNTHETIC_CHECKPOINT)
    (tmp_path / "latents").mkdir()
    torch.save(payload, str(tmp_path / "latents" / "ramp.pt"))
    records = load_dataset(write_test_dataset(str(tmp_path), [wav_path]))

    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.full(size, window_start))
    dataset = LatentHiFiGANDataset(records, str(tmp_path), str(tmp_path / "latents"), decoder,
                                   segment_frames=window, allow_synthetic_latents=True)
    audio, frames, _ = dataset[0]
    assert frames.shape == (SPEC.decoder_input_dim, window) and audio.shape[-1] == window * SPEC.output_hop_length
    with torch.no_grad():
        decoded = decoder.waveform_decoder(frames.unsqueeze(0)).reshape(-1)
    assert torch.equal(decoded, audio.reshape(-1))
    start = window_start * SPEC.output_hop_length
    assert torch.equal(audio.reshape(-1), full_audio[start:start + window * SPEC.output_hop_length])


# ---------------------------------------------------------------------------
# Overfit canaries: each stage's real loss function must drive one fixed batch down
# ---------------------------------------------------------------------------

def _tones(seconds=1.0):
    sr = SPEC.gpt_sample_rate
    t = torch.linspace(0, seconds, int(seconds * sr))
    return [(torch.sin(2 * torch.pi * f * t) * 0.3).unsqueeze(0) for f in (220.0, 330.0)]


def test_stage1_overfits_one_batch():
    from xtts_trainer.frontend import dvae_mel_extractor
    from xtts_trainer.models import build_dvae
    from xtts_trainer.stage1_dvae.train import dvae_training_loss

    torch.manual_seed(0)
    dvae = build_dvae(num_tokens=64, codebook_dim=32, hidden_dim=32)
    mels = torch.cat([dvae_mel_extractor()(w) for w in _tones()])
    mels = mels[..., : mels.shape[-1] // 4 * 4]
    opt = torch.optim.Adam(dvae.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        loss, _, _ = dvae_training_loss(dvae, mels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"DVAE did not overfit: {losses[0]:.3f} -> {losses[-1]:.3f}"


# Stage 2's overfit canary lives in tests/test_stage2_gpt_trainer.py (test_stage2_overfits_one_batch_with_masking),
# next to the loss it exercises.


def test_stage3_overfits_one_segment():
    from xtts_trainer.frontend import vocoder_loss_mel
    from xtts_trainer.models import build_hifigan
    from xtts_trainer.stage3_hifigan.train import generator_mel_loss

    torch.manual_seed(0)
    generator = build_hifigan()
    mel = vocoder_loss_mel()
    latents = torch.randn(1, 16, SPEC.gpt.model_dim)
    speaker = torch.randn(1, SPEC.d_vector_dim, 1)
    target = (torch.sin(torch.linspace(0, 2000, SPEC.vocoder_num_samples(16))) * 0.3).reshape(1, 1, -1)
    opt = torch.optim.AdamW(generator.waveform_decoder.parameters(), lr=2e-4, betas=(0.8, 0.99))
    losses = []
    for _ in range(30):
        loss = generator_mel_loss(mel, generator(latents, g=speaker), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"vocoder did not overfit: {losses[0]:.1f} -> {losses[-1]:.1f}"


# ---------------------------------------------------------------------------
# Model-gated canaries (run where xtts_base/ exists, e.g. Colab)
# ---------------------------------------------------------------------------

def _reference_voice():
    """(lock entry, wav path) of the first locked reference voice, or skip.

    The lock is committed but the audio is built on demand (scripts/fetch_reference_voices.py), so a fresh
    checkout has the lock without the audio: checking only the lock made these tests fail instead of skip
    (Colab run 4 step 5, before step 6 built the voices).
    """
    lock_path = os.path.join(REPO_ROOT, "eval", "dataset", "reference_voices.lock.json")
    if not os.path.exists(lock_path):
        pytest.skip("needs the reference-voice lock (scripts/fetch_reference_voices.py)")
    with open(lock_path, "r", encoding="utf-8") as f:
        speaker_id, entry = next(iter(sorted(json.load(f)["voices"].items())))
    if "transcript" not in entry:
        pytest.skip("reference lock predates transcripts; rerun scripts/fetch_reference_voices.py --update_lock")
    wav = os.path.join(REPO_ROOT, "eval", "reference_voices", f"{speaker_id}.wav")
    if not os.path.exists(wav):
        pytest.skip(f"needs the reference voice audio {wav} (scripts/fetch_reference_voices.py)")
    return entry, wav


def test_reference_voice_helper_skips_without_audio(tmp_path, monkeypatch):
    """A checkout with the committed lock but no built audio must skip the real-voice tests, not fail them."""
    import shutil

    (tmp_path / "eval" / "dataset").mkdir(parents=True)
    shutil.copy(os.path.join(REPO_ROOT, "eval", "dataset", "reference_voices.lock.json"), tmp_path / "eval" / "dataset")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="reference voice audio"):
        _reference_voice()
    (tmp_path / "eval" / "reference_voices").mkdir()
    speaker = sorted(json.loads(Path(tmp_path / "eval" / "dataset" / "reference_voices.lock.json").read_text(encoding="utf-8"))["voices"])[0]
    (tmp_path / "eval" / "reference_voices" / f"{speaker}.wav").write_bytes(b"")
    assert _reference_voice()[1].endswith(f"{speaker}.wav")


@needs_base_model
def test_vocoder_round_trip_prefers_true_alignment(tmp_path):
    """Real clip -> teacher-forced latents (our extractor) -> stock HifiDecoder weights -> audio.

    The resynthesis must match the original best at the true alignment: comparing it against the original
    shifted by one latent frame either way must be worse. Threshold-free: the pipeline is compared against
    perturbed versions of itself, so it catches latent/audio offset bugs without a tuned bound.
    """
    from dataset_helpers import write_test_dataset
    from xtts_trainer.config import load_audio
    from xtts_trainer.frontend import vocoder_loss_mel
    from xtts_trainer.models import build_hifigan, load_state_strict, safe_torch_load
    from xtts_trainer.stage3_hifigan.extract_latents import extract_latents, latent_filename

    entry, reference_wav = _reference_voice()

    # Re-encode as float WAV: same audio, different bytes, so the eval-leakage check doesn't (rightly) refuse it
    original, sr = sf.read(reference_wav, dtype="float32")
    wav = str(tmp_path / "clip.wav")
    sf.write(wav, original, sr, subtype="FLOAT")
    dataset = write_test_dataset(str(tmp_path), [wav], texts=[entry["transcript"]])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_latents(dataset, str(tmp_path / "latents"), checkpoint_dir=XTTS_BASE, device=device)
    payload = torch.load(str(tmp_path / "latents" / latent_filename("clip.wav")), weights_only=False)

    ckpt = safe_torch_load(os.path.join(XTTS_BASE, "model.pth"))["model"]
    decoder = build_hifigan()
    load_state_strict(decoder, {k[16:]: v for k, v in ckpt.items() if k.startswith("hifigan_decoder.")}, "upstream HiFi-GAN")
    decoder.eval()
    with torch.no_grad():
        resynth = decoder(payload["gpt_latents"], g=payload["speaker_embedding"]).reshape(-1)
    target, _ = load_audio(wav, target_sr=SPEC.output_sample_rate)
    target = target.reshape(-1)

    mel = vocoder_loss_mel()
    shift = int(round(SPEC.samples_per_latent))

    def mel_error(offset):
        a, b = (resynth[offset:], target) if offset >= 0 else (resynth, target[-offset:])
        n = min(len(a), len(b))
        return float((mel(a[:n].unsqueeze(0)) - mel(b[:n].unsqueeze(0))).abs().mean())

    aligned, early, late = mel_error(0), mel_error(shift), mel_error(-shift)
    assert aligned < early and aligned < late, f"aligned {aligned:.3f} vs shifted {early:.3f} / {late:.3f}"


@needs_base_model
def test_upstream_checkpoint_loads_into_factory_models():
    """Identity canary, load half: the real upstream weights load strictly into our factory-built models."""
    from xtts_trainer.models import build_gpt, build_hifigan, load_dvae, load_state_strict, safe_torch_load

    ckpt = safe_torch_load(os.path.join(XTTS_BASE, "model.pth"))["model"]
    gpt = build_gpt(number_text_tokens=SPEC.gpt.number_text_tokens)
    load_state_strict(gpt, {k[4:]: v for k, v in ckpt.items() if k.startswith("gpt.")}, "upstream GPT")
    hifi = build_hifigan()
    load_state_strict(hifi, {k[16:]: v for k, v in ckpt.items() if k.startswith("hifigan_decoder.")}, "upstream HiFi-GAN")
    load_dvae(os.path.join(XTTS_BASE, "dvae.pth"))


# ---------------------------------------------------------------------------
# Identity canary: the export path is lossless
# ---------------------------------------------------------------------------

def _stock_xtts(checkpoint_dir):
    """The base model as stock upstream loads it (Xtts.load_checkpoint, strict, stock tokenizer)."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))
    model = Xtts(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True, strict=True)
    return model


def _assert_identical(base_dir, export_dir, reference_wav):
    from xtts_trainer.stage2_gpt.export import load_for_inference

    stock, ours = _stock_xtts(base_dir), load_for_inference(export_dir)
    a, b = stock.state_dict(), ours.state_dict()
    assert a.keys() == b.keys()
    different = [k for k in a if not torch.equal(a[k], b[k])]
    assert not different, f"{len(different)} tensors changed on the way through export, e.g. {different[:3]}"
    outputs = []
    for model in (stock, ours):
        lat, spk = model.get_conditioning_latents(audio_path=[reference_wav])
        torch.manual_seed(0)
        outputs.append(np.asarray(model.inference("The quick brown fox jumps over the lazy dog.", "en", lat, spk,
                                                  temperature=0.7)["wav"]))
    assert outputs[0].shape == outputs[1].shape and np.array_equal(outputs[0], outputs[1])


@pytest.mark.skipif(not os.path.exists(os.path.join(XTTS_BASE, "vocab.json")), reason="needs xtts_base/vocab.json")
def test_identity_canary_tiny(tmp_path):
    """export_upstream on an unchanged model reproduces it exactly: every tensor, and the audio for a seed."""
    from xtts_trainer.models import build_dvae
    from xtts_trainer.stage2_gpt.export import export_for_inference, export_upstream
    from xtts_trainer.stage2_gpt.gpt_trainer import ProjectGPTTrainer, build_gpt_trainer_config

    tiny = dict(gpt_layers=1, gpt_n_model_channels=64, gpt_n_heads=2, decoder_input_dim=64)
    base = tmp_path / "base"
    base.mkdir()
    torch.save(build_dvae().state_dict(), base / "dvae.pth")
    torch.manual_seed(3)  # a random "released" model, written in the upstream checkpoint format
    model = ProjectGPTTrainer(build_gpt_trainer_config(os.path.join(XTTS_BASE, "vocab.json"), str(base / "dvae.pth"),
                                                       str(tmp_path), arch_overrides=tiny))
    export_for_inference(model, str(base))
    export_upstream(str(base), str(tmp_path / "export"), arch_overrides=tiny)
    wav = str(tmp_path / "ref.wav")
    sf.write(wav, (0.1 * np.random.RandomState(0).randn(22050 * 3)).astype(np.float32), 22050)
    _assert_identical(str(base), str(tmp_path / "export"), wav)


@needs_base_model
def test_identity_canary_released_model(tmp_path):
    """The released XTTS-v2 through export_upstream equals the released XTTS-v2 (Phase 5 canary 2)."""
    from xtts_trainer.stage2_gpt.export import export_upstream

    _, reference_wav = _reference_voice()
    export_upstream(XTTS_BASE, str(tmp_path / "export"))
    _assert_identical(XTTS_BASE, str(tmp_path / "export"), reference_wav)
