"""Phase 1 evaluation plumbing: ECAPA2/UTMOS wrappers, harness wiring, reference voices, run comparison.

External models are replaced by fakes so these run offline; the real models are exercised in the
Phase 0/1 Colab notebook.
"""

import io
import json
import math
import os
import sys
import tarfile
import zlib
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import soundfile as sf
import torch

import eval.harness as harness_module
from eval.dataset import load_eval_set
from eval.harness import EvaluationHarness, aggregate_metric
from eval.metrics import ECAPA2_SAMPLE_RATE, UTMOS_SAMPLE_RATE, SpeakerSimilarity, UTMOSPredictor
from eval.reference_voices import MAX_CLIP_S, MIN_CLIP_S, build_reference_voices, verify_reference_voices
from scripts.compare_baselines import compare
from xtts_trainer.spec import SPEC


class FakeEncoder:
    """Records inputs; embedding = [mean, std] of the waveform."""

    def __init__(self):
        self.calls = []

    def __call__(self, wav):
        self.calls.append(tuple(wav.shape))
        return torch.stack([wav.mean(), wav.std()]).unsqueeze(0)


class FakeMOS:
    def __init__(self):
        self.calls = []

    def __call__(self, wav, sr):
        self.calls.append((tuple(wav.shape), sr))
        return torch.tensor([3.5])


def test_speaker_similarity_resamples_to_16k():
    enc = FakeEncoder()
    sim = SpeakerSimilarity(model=enc)
    audio = np.random.RandomState(0).randn(SPEC.output_sample_rate).astype(np.float32)
    emb = sim.embed(audio, SPEC.output_sample_rate)
    assert enc.calls[-1] == (1, ECAPA2_SAMPLE_RATE)
    assert sim.similarity(emb, audio, SPEC.output_sample_rate) == pytest.approx(1.0, abs=1e-5)


def test_utmos_predictor_passes_16k():
    mos = FakeMOS()
    score = UTMOSPredictor(model=mos).score(np.zeros(SPEC.output_sample_rate, dtype=np.float32), SPEC.output_sample_rate)
    assert score == pytest.approx(3.5)
    assert mos.calls[-1] == ((1, UTMOS_SAMPLE_RATE), UTMOS_SAMPLE_RATE)


def test_aggregate_metric_excludes_nan():
    agg = aggregate_metric([0.1, 0.3, float("nan")])
    assert agg["n"] == 2 and agg["n_nan"] == 1
    assert agg["mean"] == pytest.approx(0.2)
    assert agg["ci95"] > 0


def test_harness_uses_ecapa2_not_xtts_encoder(monkeypatch, tmp_path):
    """SECS must come from the independent encoder; the model exposes no get_speaker_embedding to call."""
    ref_path = str(tmp_path / "ref.wav")
    sf.write(ref_path, np.random.RandomState(1).randn(16000 * 7).astype(np.float32) * 0.1, 16000)
    wav = (np.sin(np.linspace(0, 2000, SPEC.output_sample_rate * 2)) * 0.3).astype(np.float32)

    model = SimpleNamespace(
        get_conditioning_latents=lambda audio_path: (torch.zeros(1, 32, 1024), torch.zeros(1, 512, 1)),
        inference=lambda **kw: {"wav": wav},
    )
    monkeypatch.setattr(harness_module, "asr_transcribe", lambda audio, sr, lang, model_name, device: "hello world")
    enc, mos = FakeEncoder(), FakeMOS()
    harness = EvaluationHarness(device="cpu", speaker_similarity=SpeakerSimilarity(model=enc),
                                mos_predictor=UTMOSPredictor(model=mos))
    manifest = harness.run_benchmark(model, ref_path, seeds=[1, 2], subset_size=2, language="en")

    summary = manifest["summary"]
    assert summary["total_evaluations"] == 4
    assert summary["utmos"]["mean"] == pytest.approx(3.5)
    assert summary["secs"]["n"] == 4 and summary["secs_all"]["n"] == 4  # 2.0 s clips count toward headline SECS
    assert set(summary["by_category"]) == {"harvard", "sibilant"}  # stratified subset of 2
    assert manifest["metrics_version"] == harness_module.METRICS_VERSION
    assert enc.calls[0] == (1, 16000 * 7)  # reference embedded once at its native 16 kHz
    assert manifest["asr_model"] == "openai/whisper-large-v3"


def test_harness_flags_babbling_with_the_absolute_excess_rule(monkeypatch):
    """A 1.15 s "Yes." (ratio ~2.3, run 2) is a normal take; 3 s of it is babbling."""
    monkeypatch.setattr(harness_module, "asr_transcribe", lambda audio, sr, lang, model_name, device: "yes")
    harness = EvaluationHarness(device="cpu", speaker_similarity=SpeakerSimilarity(model=FakeEncoder()),
                                mos_predictor=UTMOSPredictor(model=FakeMOS()))
    ref = harness.speaker_similarity.embed(np.random.RandomState(0).randn(16000).astype(np.float32), 16000)

    def take(seconds):
        wav = (np.sin(np.linspace(0, 500, int(SPEC.output_sample_rate * seconds))) * 0.3).astype(np.float32)
        model = SimpleNamespace(inference=lambda **kw: {"wav": wav})
        return harness.evaluate_utterance(model, "Yes.", "en", None, None, ref)

    normal, babble = take(1.15), take(3.0)
    assert normal["duration_ratio"] > 2.2 and not normal["failures"]["is_too_long"]
    assert babble["failures"]["is_too_long"]


def test_headline_secs_excludes_short_clips():
    from eval.harness import HEADLINE_SECS_MIN_S, summarize_results

    def result(duration, secs):
        return {"wer": 0.0, "cer": 0.0, "secs": secs, "utmos": 4.0, "duration_ratio": 1.0, "duration_s": duration,
                "failures": {"is_empty": False, "is_too_long": False, "is_nan": False}}

    s = summarize_results([result(0.8, 0.2), result(HEADLINE_SECS_MIN_S, 0.6), result(5.0, 0.7)])
    assert s["secs"]["n"] == 2 and s["secs"]["mean"] == pytest.approx(0.65)
    assert s["secs_all"]["n"] == 3 and s["secs_all"]["mean"] == pytest.approx(0.5)


def test_rescore_colab_run2_pilot():
    """Run 2's measured utterances under the current definitions: the digit sentence is no longer 'slow' and
    the one-word 'Yes.' no longer counts as babbling; headline SECS drops the sub-2 s clips."""
    import scripts.rescore_baseline as rescore_module

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "docs", "evidence", "colab_run2", "pilot_en.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(root, "eval", "dataset", "eval_set.json"), encoding="utf-8") as f:
        eval_set = json.load(f)
    assert manifest["failure_counts"]["too_long"] == 1
    out = rescore_module.rescore(manifest, eval_set)
    assert out["failure_counts"]["too_long"] == 0
    cats = out["summary_by_category"]["en"]
    assert 0.7 < cats["numbers"]["duration_ratio"]["mean"] < 1.3  # was 1.61 counting written digits
    assert cats["short"]["secs"]["n"] == 0 and cats["short"]["secs_all"]["n"] == 20
    en = out["summary_by_language"]["en"]
    assert en["secs_all"]["n"] == 120 and en["secs"]["n"] < 120 and en["secs"]["mean"] > en["secs_all"]["mean"]
    assert en["wer"] == manifest["summary_by_language"]["en"]["wer"]  # measured values untouched
    assert out["metrics_version"] == 2 and out["rescored_from_metrics_version"] == 1


def test_harness_rejects_unknown_language(monkeypatch, tmp_path):
    ref_path = str(tmp_path / "ref.wav")
    sf.write(ref_path, np.zeros(16000 * 7, dtype=np.float32), 16000)
    model = SimpleNamespace(get_conditioning_latents=lambda audio_path: (None, None))
    harness = EvaluationHarness(device="cpu", speaker_similarity=SpeakerSimilarity(model=FakeEncoder()),
                                mos_predictor=UTMOSPredictor(model=FakeMOS()))
    with pytest.raises(ValueError, match="no sentences"):
        harness.run_benchmark(model, ref_path, seeds=[1], language="xx")


# ---------------------------------------------------------------------------
# Reference voices
# ---------------------------------------------------------------------------

def _make_arctic_tarballs(src_dir, speakers, utterances, sr=16000):
    for spk in speakers:
        tar_path = os.path.join(src_dir, f"cmu_us_{spk}_arctic.tar.bz2")
        with tarfile.open(tar_path, "w:bz2") as tar:
            def add(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            add(f"cmu_us_{spk}_arctic/etc/txt.done.data",
                "\n".join(f'( {u} "Sentence {u[-2:]} for {spk}." )' for u in utterances).encode())
            rng = np.random.RandomState(zlib.crc32(spk.encode()) % 1000)
            for utt in utterances:
                speech = (rng.randn(int(2.0 * sr)) * 0.2).astype(np.float32)
                wav = np.concatenate([np.zeros(int(0.3 * sr), np.float32), speech, np.zeros(int(0.3 * sr), np.float32)])
                buf = io.BytesIO()
                sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
                add(f"cmu_us_{spk}_arctic/wav/{utt}.wav", buf.getvalue())
        # Like festvox: the packed tarball has no license; it is published per speaker alongside
        with open(os.path.join(src_dir, f"COPYING_{spk}"), "w", encoding="utf-8") as f:
            f.write(f"test license for {spk}\n")


def _voice_kwargs(tmp_path, src):
    return dict(
        out_dir=str(tmp_path / "voices"),
        lock_path=str(tmp_path / "lock.json"),
        cache_dir=str(tmp_path / "cache"),
        url_template=f"file://{src}/cmu_us_{{speaker}}_arctic.tar.bz2",
        license_url_template=f"file://{src}/COPYING_{{speaker}}",
    )


def test_reference_voices_build_lock_and_verify(tmp_path):
    eval_data = load_eval_set()
    speakers = [v["corpus_speaker"] for v in eval_data["reference_voices"]]
    utterances = eval_data["reference_voices"][0]["utterances"]
    src = tmp_path / "src"
    src.mkdir()
    _make_arctic_tarballs(str(src), speakers, utterances)

    kwargs = _voice_kwargs(tmp_path, src)
    lock = build_reference_voices(**kwargs)
    assert set(lock["voices"]) == {v["speaker_id"] for v in eval_data["reference_voices"]}
    for speaker_id, entry in lock["voices"].items():
        assert MIN_CLIP_S <= entry["duration_s"] <= MAX_CLIP_S
        license_path = os.path.join(kwargs["out_dir"], entry["license_file"])
        assert Path(license_path).read_text(encoding="utf-8") == f"test license for {speaker_id.split('_', 1)[1]}\n"
        # 2 s utterances + 0.25 s gaps: three are needed to pass 6 s, and the transcript is exactly theirs
        assert entry["utterances_used"] == utterances[:3]
        spk = speaker_id.split("_", 1)[1]
        assert entry["transcript"] == " ".join(f"Sentence {u[-2:]} for {spk}." for u in utterances[:3])

    # The license requires marking modifications: the notice names every clip's source utterances
    notice = Path(kwargs["out_dir"], "MODIFICATIONS.txt").read_text(encoding="utf-8")
    assert "MODIFIED" in notice and all(f"{sid}.wav" in notice for sid in lock["voices"])

    # Second run verifies against the lock and is deterministic
    assert build_reference_voices(**kwargs) == lock
    assert len(verify_reference_voices(kwargs["out_dir"], kwargs["lock_path"])) == len(speakers)

    # A changed license fails verification
    license_path = os.path.join(kwargs["out_dir"], lock["voices"][next(iter(lock["voices"]))]["license_file"])
    original_license = Path(license_path).read_bytes()
    with open(license_path, "ab") as f:
        f.write(b"edited")
    with pytest.raises(RuntimeError, match="License"):
        verify_reference_voices(kwargs["out_dir"], kwargs["lock_path"])
    with open(license_path, "wb") as f:
        f.write(original_license)

    # A tampered clip fails verification
    first = next(iter(lock["voices"]))
    with open(os.path.join(kwargs["out_dir"], f"{first}.wav"), "ab") as f:
        f.write(b"\0")
    with pytest.raises(RuntimeError, match="locked SHA-256"):
        verify_reference_voices(kwargs["out_dir"], kwargs["lock_path"])

    # A changed source tarball fails the build
    cached = os.path.join(kwargs["cache_dir"], f"cmu_us_{speakers[0]}_arctic.tar.bz2")
    with open(cached, "ab") as f:
        f.write(b"\0")
    with pytest.raises(RuntimeError, match="!= locked"):
        build_reference_voices(**kwargs)


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def _manifest(shift=0.0, seeds=(42,)):
    stats = lambda mean: {"mean": mean, "std": 0.05, "n": 100}
    return {
        "eval_set_sha256": "abc", "checkpoint_sha256": {"model.pth": "x"}, "reference_voices": [],
        "seeds": list(seeds), "subset_size": None, "languages": ["en"], "asr_model": "openai/whisper-large-v3",
        "metrics_version": 2,
        "summary_by_language": {"en": {m: stats(0.5 + shift) for m in ("wer", "cer", "secs", "secs_all", "utmos", "duration_ratio")}},
    }


def test_compare_baselines():
    assert compare(_manifest(), _manifest())["reproducible"]
    assert compare(_manifest(), _manifest(shift=0.005))["reproducible"]  # inside 1.96 * sqrt(2) * 0.005
    assert not compare(_manifest(), _manifest(shift=0.05))["reproducible"]
    result = compare(_manifest(), _manifest(seeds=(1,)))
    assert not result["reproducible"] and result["input_mismatches"] == ["seeds"]
    v1 = {**_manifest(), "metrics_version": 1}  # scored under different metric definitions
    assert compare(_manifest(), v1)["input_mismatches"] == ["metrics_version"]


def test_generate_baseline_dry_run(monkeypatch, tmp_path):
    """Full baseline wiring with a fake XTTS model and fake metric models (no downloads)."""
    import shutil
    import TTS.tts.models.xtts as xtts_module
    import scripts.run_eval_baseline as baseline_module
    from xtts_trainer.spec.xtts_spec import VENDORED_CONFIG_PATH

    # Reference voices built from synthetic tarballs, then locked
    eval_data = load_eval_set()
    src = tmp_path / "src"
    src.mkdir()
    _make_arctic_tarballs(str(src), [v["corpus_speaker"] for v in eval_data["reference_voices"]],
                          eval_data["reference_voices"][0]["utterances"])
    voices_dir, lock_path = str(tmp_path / "voices"), str(tmp_path / "lock.json")
    build_reference_voices(out_dir=voices_dir, lock_path=lock_path, cache_dir=str(tmp_path / "cache"),
                           url_template=f"file://{src}/cmu_us_{{speaker}}_arctic.tar.bz2",
                           license_url_template=f"file://{src}/COPYING_{{speaker}}")

    ckpt = tmp_path / "xtts_base"
    ckpt.mkdir()
    shutil.copy(VENDORED_CONFIG_PATH, ckpt / "config.json")
    (ckpt / "model.pth").write_bytes(b"fake")
    (ckpt / "vocab.json").write_bytes(b"{}")

    wav = (np.sin(np.linspace(0, 3000, SPEC.output_sample_rate * 2)) * 0.3).astype(np.float32)

    class FakeXtts:
        @staticmethod
        def init_from_config(config):
            return FakeXtts()

        def load_checkpoint(self, config, checkpoint_path, vocab_path, use_deepspeed):
            assert os.path.basename(checkpoint_path) == "model.pth"

        def to(self, device):
            return self

        def eval(self):
            return self

        def get_conditioning_latents(self, audio_path):
            return torch.zeros(1, 32, 1024), torch.zeros(1, 512, 1)

        def inference(self, **kw):
            return {"wav": wav}

    monkeypatch.setattr(xtts_module, "Xtts", FakeXtts)
    monkeypatch.setattr(harness_module, "SpeakerSimilarity", lambda device: SpeakerSimilarity(model=FakeEncoder()))
    monkeypatch.setattr(harness_module, "UTMOSPredictor", lambda device: UTMOSPredictor(model=FakeMOS()))
    monkeypatch.setattr(harness_module, "asr_transcribe", lambda audio, sr, lang, model_name, device: "text")

    out = str(tmp_path / "baseline.json")
    manifest = baseline_module.generate_baseline(
        checkpoint_dir=str(ckpt), reference_audio_dir=voices_dir, lock_path=lock_path,
        output_path=out, seeds=[1, 2], languages=["en", "hi"], subset_size=2,
    )
    with open(out, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["summary_by_language"].keys() == {"en", "hi"}
    for lang in ("en", "hi"):
        stats = on_disk["summary_by_language"][lang]
        assert set(stats) == {"wer", "cer", "secs", "secs_all", "utmos", "duration_ratio"}
        assert on_disk["summary_by_category"][lang]
        assert stats["wer"]["n"] == 2 * 2 * len(eval_data["reference_voices"])  # sentences x seeds x voices
    assert len(on_disk["reference_voices"]) == len(eval_data["reference_voices"])
    assert on_disk["subset_size"] == 2 and on_disk["asr_model"] == "openai/whisper-large-v3"
    assert compare(manifest, on_disk)["reproducible"]


def test_assemble_clip_skips_rather_than_truncates():
    """An utterance that would push the clip past MAX_CLIP_S is skipped whole, never cut mid-word."""
    from eval.reference_voices import assemble_clip

    sr = 16000
    utt = lambda seconds, seed: (np.random.RandomState(seed).randn(int(seconds * sr)) * 0.2).astype(np.float32)
    clip, used = assemble_clip([utt(4.0, 0), utt(7.0, 1), utt(3.0, 2)], sr)
    assert used == [0, 2]  # the 7 s utterance would exceed 10 s
    assert abs(len(clip) / sr - (4.0 + 0.25 + 3.0)) < 0.02
    with pytest.raises(ValueError, match="only"):
        assemble_clip([utt(2.0, 0), utt(2.0, 1)], sr)


def test_old_lock_without_license_is_refused(tmp_path):
    """A lock from before license capture (like Colab run 1's) must be rebuilt, not silently used."""
    eval_data = load_eval_set()
    src = tmp_path / "src"
    src.mkdir()
    _make_arctic_tarballs(str(src), [v["corpus_speaker"] for v in eval_data["reference_voices"]],
                          eval_data["reference_voices"][0]["utterances"])
    kwargs = _voice_kwargs(tmp_path, src)
    lock = build_reference_voices(**kwargs)
    for entry in lock["voices"].values():
        for key in ("license_url", "license_file", "license_sha256"):
            del entry[key]
    with open(kwargs["lock_path"], "w", encoding="utf-8") as f:
        json.dump(lock, f)
    with pytest.raises(RuntimeError, match="--update_lock"):
        build_reference_voices(**kwargs)
    with pytest.raises(RuntimeError, match="--update_lock"):
        verify_reference_voices(kwargs["out_dir"], kwargs["lock_path"])
    assert build_reference_voices(**kwargs, update_lock=True)["voices"]  # rebuild works


def test_code_fingerprint_tracks_source_changes(tmp_path):
    from eval.harness import get_code_fingerprint

    root = tmp_path / "repo"
    (root / "xtts_trainer").mkdir(parents=True)
    (root / "eval" / "baselines").mkdir(parents=True)
    (root / "xtts_trainer" / "a.py").write_text("x = 1\n")
    (root / "pyproject.toml").write_text("[project]\n")
    first = get_code_fingerprint(str(root))
    assert first == get_code_fingerprint(str(root)) and first["n_files"] == 2
    (root / "eval" / "baselines" / "run.json").write_text("{}")  # outputs don't change the fingerprint
    (root / "eval" / "dataset").mkdir()
    (root / "eval" / "dataset" / "reference_voices.lock.json").write_text("{}")  # nor data built during a run
    assert get_code_fingerprint(str(root)) == first
    (root / "xtts_trainer" / "spec.json").write_text("{}")  # vendored config inside the package is code
    second = get_code_fingerprint(str(root))
    assert second["n_files"] == 3 and second != first
    (root / "xtts_trainer" / "a.py").write_text("x = 2\n")
    assert get_code_fingerprint(str(root))["sha256"] != second["sha256"]
