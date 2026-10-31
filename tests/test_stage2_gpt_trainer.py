"""Phase 7 step 1: Stage 2 through upstream GPTTrainer, with the F2, F4 and F6 fixes.

Evidence tests run upstream's own code to show each defect; guard tests show the wrapper fixes it.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest
import soundfile as sf
import torch

from dataset_helpers import write_test_dataset
from xtts_trainer.models import build_dvae, build_gpt, stamp
from xtts_trainer.spec import SPEC
from xtts_trainer.stage2_gpt.gpt_trainer import (
    ProjectGPTTrainer, StrictXTTSDataset, build_gpt_trainer_config, gpt_losses, language_shares,
    mel_targets_like_upstream, prompt_code_span, records_to_samples,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_VOCAB = os.path.join(REPO_ROOT, "xtts_base", "vocab.json")
needs_vocab = pytest.mark.skipif(not os.path.exists(REAL_VOCAB), reason="needs xtts_base/vocab.json")
TINY = dict(gpt_layers=1, gpt_n_model_channels=64, gpt_n_heads=2, decoder_input_dim=64)
STRIDE = SPEC.gpt.code_stride_len


# --------------------------------------------------------------------------- F2: prompt masking

def _tiny_gpt():
    torch.manual_seed(0)
    return build_gpt(number_text_tokens=30, start_text_token=28, stop_text_token=0, layers=1, model_dim=64, heads=2)


def _batch():
    n_codes = torch.tensor([40, 30, 36])
    torch.manual_seed(1)
    return dict(
        text_inputs=torch.randint(3, 25, (3, 6)), text_lengths=torch.tensor([6, 5, 4]),
        audio_codes=torch.randint(0, SPEC.dvae.num_tokens, (3, 40)), wav_lengths=n_codes * STRIDE,
        cond_mels=torch.randn(3, 1, 80, 100), cond_lens=None,
        # prompt spans in samples, as XTTSDataset produces them
        cond_idxs=torch.tensor([[2 * STRIDE, 8 * STRIDE], [STRIDE, 6 * STRIDE + 100], [3 * STRIDE, 11 * STRIDE]]),
    )


class CrossEntropySpy:
    def __init__(self, monkeypatch):
        self.targets = []
        original = torch.nn.functional.cross_entropy

        def spy(logits, targets, *args, **kwargs):
            self.targets.append(targets.clone())
            return original(logits, targets, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "cross_entropy", spy)


def test_upstream_f2_masks_only_the_last_sample_in_the_wrong_units(monkeypatch):
    """Evidence for F2 (tts/layers/xtts/gpt.py:461-465): only the last sample's prompt is masked, and with the
    perceiver resampler at cond_idxs // 256 (conditioning-mel frames) instead of // 1024 (codes)."""
    gpt, b = _tiny_gpt().train(), _batch()
    assert gpt.use_perceiver_resampler and gpt.perceiver_cond_length_compression == 256
    unmasked = mel_targets_like_upstream(gpt, b["audio_codes"], b["wav_lengths"])
    spy = CrossEntropySpy(monkeypatch)
    gpt(b["text_inputs"], b["text_lengths"], b["audio_codes"], b["wav_lengths"], cond_mels=b["cond_mels"],
        cond_idxs=b["cond_idxs"].clone(), cond_lens=None)
    upstream = spy.targets[-1]
    assert torch.equal(upstream[:2], unmasked[:2])  # samples 0 and 1: prompt not masked at all
    changed = (upstream[2] != unmasked[2]).nonzero().flatten().tolist()
    s, e = b["cond_idxs"][2].tolist()
    assert changed == [j for j in range(s // 256, e // 256) if unmasked[2, j] != -1]
    lo, hi = prompt_code_span(s, e, STRIDE)
    assert not set(changed) & set(range(lo, hi))  # the real prompt codes are still trained on


def test_prompt_masking_per_sample(monkeypatch):
    """The wrapper masks every sample's own prompt codes, and nothing else."""
    gpt, b = _tiny_gpt().train(), _batch()
    unmasked = mel_targets_like_upstream(gpt, b["audio_codes"], b["wav_lengths"])
    spy = CrossEntropySpy(monkeypatch)
    gpt_losses(gpt, **b)
    ours = spy.targets[-1]
    for i, (s, e) in enumerate(b["cond_idxs"].tolist()):
        lo, hi = prompt_code_span(s, e, STRIDE)
        assert (ours[i, lo:hi] == -1).all()
        keep = torch.ones_like(ours[i], dtype=torch.bool)
        keep[lo:hi] = False
        assert torch.equal(ours[i][keep], unmasked[i][keep])


def test_prompt_code_span():
    assert prompt_code_span(0, STRIDE, STRIDE) == (0, 1)
    assert prompt_code_span(STRIDE - 1, STRIDE + 1, STRIDE) == (0, 2)  # partial codes at both edges
    assert prompt_code_span(3 * STRIDE, 5 * STRIDE, STRIDE) == (3, 5)


def test_gpt_loss_parity_without_prompt():
    """With no prompt to mask, the wrapper's loss is upstream's, bit for bit (target reconstruction guard)."""
    gpt, b = _tiny_gpt().eval(), _batch()
    b["cond_idxs"] = None
    torch.manual_seed(0)
    upstream = gpt(b["text_inputs"], b["text_lengths"], b["audio_codes"], b["wav_lengths"],
                   cond_mels=b["cond_mels"], cond_idxs=None, cond_lens=None)
    _, _, logits = upstream
    targets = mel_targets_like_upstream(gpt, b["audio_codes"], b["wav_lengths"])
    ours = torch.nn.functional.cross_entropy(logits, targets.long(), ignore_index=-1)
    assert torch.equal(ours, upstream[1])
    torch.manual_seed(0)
    assert torch.equal(gpt_losses(gpt, **b)[1], upstream[1])


def test_stage2_overfits_one_batch_with_masking():
    gpt, b = _tiny_gpt().train(), _batch()
    opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    mel_ce = []
    for _ in range(120):
        loss_text, loss_mel, _ = gpt_losses(gpt, **{**b, "cond_idxs": b["cond_idxs"].clone()})
        opt.zero_grad()
        (0.01 * loss_text + loss_mel).backward()
        opt.step()
        mel_ce.append(loss_mel.item())
    assert mel_ce[-1] < 0.25 * mel_ce[0], f"GPT did not overfit: mel CE {mel_ce[0]:.3f} -> {mel_ce[-1]:.3f}"


# --------------------------------------------------------------------------- F4: project tokenizer

@pytest.fixture
def sw_vocab(tmp_path):
    """The real vocab extended with a [sw] language token (Swahili is not an XTTS base language)."""
    from xtts_trainer.stage2_gpt.vocab_extender import extend_vocab

    corpus = tmp_path / "sw.txt"
    corpus.write_text("habari ya asubuhi rafiki yangu.\n", encoding="utf-8")
    out = str(tmp_path / "vocab_sw.json")
    extend_vocab(REAL_VOCAB, str(corpus), out, new_languages=["sw"])
    return out


@needs_vocab
def test_project_tokenizer_registers_languages(sw_vocab):
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

    from xtts_trainer.frontend.text import DEFAULT_CHAR_LIMIT, ProjectTokenizer

    with pytest.raises(NotImplementedError):
        VoiceBpeTokenizer(sw_vocab).encode("habari yako", "sw")  # F4: stock tokenizer refuses
    tok = ProjectTokenizer(sw_vocab, ["sw"])
    ids = tok.encode('  Habari   "Yako"  ', "sw")
    assert ids[0] == tok.tokenizer.token_to_id("[sw]") and 1 not in ids
    assert tok.decode(ids[1:]) == "habari yako"
    assert tok.char_limits["sw"] == DEFAULT_CHAR_LIMIT
    assert tok.encode("Hello there.", "en") == VoiceBpeTokenizer(sw_vocab).encode("Hello there.", "en")
    with pytest.raises(NotImplementedError):
        tok.encode("bonjour", "xx")  # unregistered languages still raise
    with pytest.raises(ValueError, match="base language"):
        ProjectTokenizer(sw_vocab, ["en"])
    with pytest.raises(ValueError, match=r"no \[dv\] token"):
        ProjectTokenizer(sw_vocab, ["dv"])


@needs_vocab
def test_validator_accepts_registered_language(tmp_path, sw_vocab):
    from xtts_trainer.data import validate_dataset

    paths = []
    for i in range(2):
        p = str(tmp_path / f"c{i}.wav")
        sf.write(p, (0.1 * np.random.RandomState(i).randn(44100)).astype("float32"), 22050)
        paths.append(p)
    ds = write_test_dataset(str(tmp_path), paths, texts=["habari ya asubuhi", "rafiki yangu"], lang="sw")
    kw = dict(eval_set_path=None, reference_lock_path=None)
    assert {e.code for e in validate_dataset(ds, sw_vocab, **kw).errors} == {"unknown_language"}
    assert validate_dataset(ds, sw_vocab, extra_languages=["sw"], **kw).ok


# --------------------------------------------------------------------------- model construction

def _dvae_file(tmp_path):
    path = str(tmp_path / "dvae.pth")
    torch.save(build_dvae().state_dict(), path)
    return path


@needs_vocab
def test_config_architecture_comes_from_the_released_model(tmp_path):
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs

    cfg = build_gpt_trainer_config(REAL_VOCAB, "dvae.pth", str(tmp_path))
    args = cfg.model_args
    assert GPTArgs().gpt_num_audio_tokens == 8194 and not GPTArgs().gpt_use_perceiver_resampler  # upstream defaults
    assert args.gpt_num_audio_tokens == SPEC.gpt.num_audio_tokens and args.gpt_use_perceiver_resampler
    assert (args.gpt_layers, args.gpt_n_model_channels) == (SPEC.gpt.layers, SPEC.gpt.model_dim)
    assert args.xtts_checkpoint == "" and args.gpt_checkpoint == ""
    assert cfg.audio.sample_rate == cfg.audio.dvae_sample_rate == SPEC.gpt_sample_rate


@needs_vocab
def test_upstream_checkpoint_arguments_are_refused(tmp_path):
    cfg = build_gpt_trainer_config(REAL_VOCAB, _dvae_file(tmp_path), str(tmp_path), arch_overrides=TINY)
    cfg.model_args.xtts_checkpoint = "xtts_base/model.pth"
    with pytest.raises(ValueError, match="non-strictly"):
        ProjectGPTTrainer(cfg)


@needs_vocab
def test_dvae_is_reloaded_strictly(tmp_path):
    state = build_dvae().state_dict()
    state["unexpected.weight"] = torch.zeros(1)
    path = str(tmp_path / "bad_dvae.pth")
    torch.save(state, path)
    cfg = build_gpt_trainer_config(REAL_VOCAB, path, str(tmp_path), arch_overrides=TINY)
    with pytest.raises(RuntimeError, match="unexpected"):
        ProjectGPTTrainer(cfg)  # upstream's own strict=False load accepted it


@needs_vocab
def test_new_text_rows_mean_init_and_upstream_f6(tmp_path, sw_vocab):
    """Base rows load unchanged; new rows start at the mean. Upstream (F6) copies the old last row, [hi]."""
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

    dvae = _dvae_file(tmp_path)
    base = ProjectGPTTrainer(build_gpt_trainer_config(REAL_VOCAB, dvae, str(tmp_path), arch_overrides=TINY))
    base_rows = base.xtts.gpt.text_embedding.weight.detach().clone()
    gpt_state = {k: v.clone() for k, v in base.xtts.gpt.state_dict().items()}
    model_pth = str(tmp_path / "model.pth")
    torch.save(stamp({"model": {f"gpt.{k}": v for k, v in gpt_state.items()}}), model_pth)
    n_base = base_rows.shape[0]

    cfg = build_gpt_trainer_config(sw_vocab, dvae, str(tmp_path), extra_languages=["sw"], arch_overrides=TINY)
    ours = ProjectGPTTrainer(cfg, ["sw"], base_model_path=model_pth).xtts.gpt.text_embedding.weight.detach()
    assert ours.shape[0] == n_base + 1
    assert torch.equal(ours[:n_base], base_rows)
    assert torch.allclose(ours[n_base], base_rows.mean(0), atol=0.05)

    gpt_ckpt = str(tmp_path / "gpt.pth")
    torch.save(gpt_state, gpt_ckpt)
    cfg = build_gpt_trainer_config(sw_vocab, dvae, str(tmp_path), arch_overrides=TINY)
    cfg.model_args.gpt_checkpoint = gpt_ckpt
    upstream = GPTTrainer(cfg).xtts.gpt.text_embedding.weight.detach()
    hi = json.loads(Path(REAL_VOCAB).read_text(encoding="utf-8"))["model"]["vocab"].get("[hi]")
    hi = hi if hi is not None else n_base - 1
    assert hi == n_base - 1
    assert torch.equal(upstream[n_base], base_rows[hi])  # the new [sw] row is a copy of [hi]


# --------------------------------------------------------------------------- DVAE pairing (F1)

def _stage1_dvae(path):
    """A full-size DVAE in the Stage 1 output format (marked requires_gpt_retrain)."""
    from xtts_trainer.models import REQUIRES_GPT_RETRAIN_KEY

    torch.save(stamp({"dvae": build_dvae().state_dict(), REQUIRES_GPT_RETRAIN_KEY: True}), path)
    return path


@needs_vocab
def test_stage1_dvae_with_pretrained_gpt_is_refused(tmp_path):
    dvae = _stage1_dvae(str(tmp_path / "dvae_stage1.pth"))
    base = ProjectGPTTrainer(build_gpt_trainer_config(REAL_VOCAB, _dvae_file(tmp_path), str(tmp_path), arch_overrides=TINY))
    model_pth = str(tmp_path / "model.pth")
    torch.save(stamp({"model": {f"gpt.{k}": v for k, v in base.xtts.gpt.state_dict().items()}}), model_pth)
    cfg = build_gpt_trainer_config(REAL_VOCAB, dvae, str(tmp_path), arch_overrides=TINY)
    with pytest.raises(ValueError, match="retrain_for_new_dvae"):
        ProjectGPTTrainer(cfg, base_model_path=model_pth)
    assert ProjectGPTTrainer(cfg, base_model_path=model_pth, retrain_for_new_dvae=True).retrain_for_new_dvae
    ProjectGPTTrainer(cfg)  # no pretrained GPT to mismatch


@needs_vocab
def test_dvae_pairing_check(tmp_path):
    from xtts_trainer.stage2_gpt.export import DVAE_SHA_KEY, check_dvae_pairing
    from xtts_trainer.data.latents import sha256_file

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save(build_dvae().state_dict(), ckpt_dir / "dvae.pth")  # released format
    torch.save(stamp({"model": {}}), ckpt_dir / "model.pth")
    check_dvae_pairing(str(ckpt_dir))
    _stage1_dvae(str(ckpt_dir / "dvae.pth"))
    with pytest.raises(ValueError, match="not trained on its codes"):
        check_dvae_pairing(str(ckpt_dir))
    torch.save(stamp({"model": {}, DVAE_SHA_KEY: sha256_file(str(ckpt_dir / "dvae.pth"))}), ckpt_dir / "model.pth")
    check_dvae_pairing(str(ckpt_dir))  # a GPT trained on exactly this DVAE


@needs_vocab
def test_extract_latents_refuses_mismatched_dvae(tmp_path):
    import shutil

    from xtts_trainer.stage3_hifigan.extract_latents import extract_latents

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    shutil.copy(REAL_VOCAB, ckpt_dir / "vocab.json")
    _stage1_dvae(str(ckpt_dir / "dvae.pth"))
    torch.save(stamp({"model": {}}), ckpt_dir / "model.pth")  # stock-style GPT, no DVAE recorded
    wav = str(tmp_path / "a.wav")
    sf.write(wav, (0.1 * np.random.RandomState(0).randn(44100)).astype("float32"), 22050)
    wav2 = str(tmp_path / "b.wav")
    sf.write(wav2, (0.1 * np.random.RandomState(1).randn(44100)).astype("float32"), 22050)
    ds = write_test_dataset(str(tmp_path), [wav, wav2], texts=["hello world", "nice weather today"])
    with pytest.raises(ValueError, match="not trained on its codes"):
        extract_latents(ds, str(tmp_path / "latents"), checkpoint_dir=str(ckpt_dir), device="cpu")


# --------------------------------------------------------------------------- data

@needs_vocab
def test_dataset_failures_raise_instead_of_substituting(tmp_path):
    cfg = build_gpt_trainer_config(REAL_VOCAB, _dvae_file(tmp_path), str(tmp_path), arch_overrides=TINY)
    model = ProjectGPTTrainer(cfg)
    good = str(tmp_path / "good.wav")
    sf.write(good, (0.1 * np.random.RandomState(0).randn(66150)).astype("float32"), 22050)
    samples = [{"text": "hello", "audio_file": str(tmp_path / "missing.wav"), "speaker_name": "s", "language": "en",
                "root_path": str(tmp_path)}]
    dataset = StrictXTTSDataset(cfg, samples, model.xtts.tokenizer, 22050, is_eval=False)
    with pytest.raises(RuntimeError, match="silently substitute"):
        dataset[0]
    with pytest.raises(Exception):
        StrictXTTSDataset(cfg, samples, model.xtts.tokenizer, 22050, is_eval=True)


@needs_vocab
@pytest.mark.parametrize("is_eval", [False, True])
def test_strict_dataset_items_match_upstream(tmp_path, is_eval):
    """For loadable samples, StrictXTTSDataset returns exactly what upstream XTTSDataset returns."""
    import random

    from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset

    cfg = build_gpt_trainer_config(REAL_VOCAB, _dvae_file(tmp_path), str(tmp_path), arch_overrides=TINY)
    tok = ProjectGPTTrainer(cfg).xtts.tokenizer
    samples = []
    for i, lang in enumerate(["en", "en", "de"]):
        p = str(tmp_path / f"c{i}.wav")
        sf.write(p, (0.1 * np.random.RandomState(i).randn(22050 * (4 + i))).astype("float32"), 22050)
        samples.append({"text": "hallo welt" if lang == "de" else "hello world", "audio_file": p,
                        "speaker_name": "s", "language": lang, "root_path": str(tmp_path)})
    ours = StrictXTTSDataset(cfg, list(samples), tok, 22050, is_eval)
    upstream = XTTSDataset(cfg, list(samples), tok, 22050, is_eval)
    for index in range(3):
        random.seed(index)
        a = ours[index]
        random.seed(index)
        b = upstream[index]
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], torch.Tensor):
                assert torch.equal(a[key], b[key]) or (a[key].isnan().all() and b[key].isnan().all()), key
            else:
                assert a[key] == b[key], key


def test_language_shares_are_uniform_by_language():
    samples = [{"language": "sw"}] * 90 + [{"language": "en"}] * 10
    assert language_shares(samples) == {"en": 0.5, "sw": 0.5}


def test_split_and_replay_rules():
    from xtts_trainer.stage2_gpt.train import split_samples

    samples = [{"language": "sw", "id": i} for i in range(10)] + [{"language": "en", "id": i} for i in range(4)]
    train, evals = split_samples(samples, 0.2, seed=3)
    assert sorted(s["language"] for s in evals) == ["en", "sw", "sw"]
    assert not {id(s) for s in train} & {id(s) for s in evals}
    assert split_samples(samples, 0.2, seed=3) == (train, evals)
    with pytest.raises(ValueError, match="eval"):
        split_samples([{"language": "sw"}], 0.1, seed=0)


def _noise_dataset(root, name, speaker, n=2, texts=None):
    d = root / name
    d.mkdir()
    paths = []
    for i in range(n):
        p = str(d / f"c{i}.wav")
        sf.write(p, (0.3 * np.sin(np.linspace(0, 400 * (i + 1 + len(name)), 55125))).astype("float32"), 22050)
        paths.append(p)
    texts = texts or ["hello world", "nice weather today", "the river runs slowly", "a quiet green field"][:n]
    return write_test_dataset(str(d), paths, texts=texts, speaker=speaker)


@needs_vocab
def test_same_language_replay_must_come_from_other_speakers(tmp_path):
    from xtts_trainer.stage2_gpt.train import train_gpt

    with pytest.raises(ValueError, match="other speakers"):
        train_gpt(_noise_dataset(tmp_path, "target", "spk"), str(tmp_path / "run"), str(tmp_path),
                  vocab_path=REAL_VOCAB, replay_paths=[_noise_dataset(tmp_path, "replay", "spk")])


@needs_vocab
def test_same_language_replay_trains_only_and_freeze_is_recorded(tmp_path):
    """M1: other speakers' clips in the target's language join training only, so the eval split (and the best
    checkpoint's choice) is the target's own; frozen parts are recorded."""
    from xtts_trainer.stage2_gpt.train import split_samples, load_samples, train_gpt

    base = tmp_path / "base"
    base.mkdir()
    torch.save(build_dvae().state_dict(), base / "dvae.pth")
    target = _noise_dataset(tmp_path, "target", "target_spk", n=4)
    replay = _noise_dataset(tmp_path, "replay", "other_spk", n=2)
    # Training seed 2 with split seed 1: other training randomness, the same eval clips as seed 1 (M1 confirmation)
    export = train_gpt(target, str(tmp_path / "run"), str(base), vocab_path=REAL_VOCAB, replay_paths=[replay],
                       epochs=1, batch_size=2, grad_accum_steps=1, lr=1e-3, eval_fraction=0.25,
                       arch_overrides=TINY, load_base_weights=False, freeze=["conditioning", "conditioning"],
                       seed=2, split_seed=1)
    summary = json.loads(Path(export, "training_summary.json").read_text(encoding="utf-8"))
    _, target_eval = split_samples(load_samples([target], REAL_VOCAB, []), 0.25, seed=1)
    expected_eval = sorted(os.path.relpath(s["audio_file"], s["root_path"]) for s in target_eval)
    assert summary["eval_clips"] == 1 and summary["eval_audio"] == expected_eval  # the very clips, not a count
    assert summary["eval_speakers"] == ["target_spk"]  # never a replay speaker
    assert summary["seed"] == 2 and summary["split_seed"] == 1
    assert summary["train_clips"] == 3 + 2 and summary["same_language_replay_clips"] == 2
    assert summary["same_language_replay_share"] == 0.4 and summary["replay_speakers"] == ["other_spk"]
    # Named twice, frozen and counted once
    assert summary["frozen_parts"] == ["conditioning"] and summary["frozen_parameter_tensors"] > 0
    assert (summary["batch_size"], summary["grad_accum_steps"], summary["eval_fraction"]) == (2, 1, 0.25)


def test_freezing_conditioning_keeps_it_at_its_starting_values():
    from xtts_trainer.stage2_gpt.gpt_trainer import FREEZABLE_PARTS, freeze_gpt_parts

    gpt = build_gpt(number_text_tokens=30, start_text_token=28, stop_text_token=0, layers=1, model_dim=64, heads=2)
    prefixes = FREEZABLE_PARTS["conditioning"]
    before = {n: p.detach().clone() for n, p in gpt.named_parameters()}
    once = freeze_gpt_parts(gpt, ["conditioning"])
    assert once > 0 and freeze_gpt_parts(gpt, ["conditioning", "conditioning"]) == once
    opt = torch.optim.AdamW([p for p in gpt.parameters() if p.requires_grad], lr=1e-2)
    torch.manual_seed(0)
    loss = sum((p.float() ** 2).sum() for p in gpt.parameters())  # touches every parameter
    loss.backward()
    opt.step()
    after = dict(gpt.named_parameters())
    frozen = [n for n in before if n.startswith(prefixes)]
    trained = [n for n in before if not n.startswith(prefixes)]
    assert frozen and trained
    assert all(torch.equal(before[n], after[n]) for n in frozen)
    assert any(not torch.equal(before[n], after[n]) for n in trained)
    with pytest.raises(ValueError, match="unknown parts"):
        freeze_gpt_parts(gpt, ["decoder"])


# --------------------------------------------------------------------------- end to end (F4 inference)

@needs_vocab
def test_train_export_and_new_language_inference(tmp_path, sw_vocab):
    """Tiny run through the coqui Trainer, best checkpoint exported, loaded by stock Xtts (strict), and
    synthesizing a language registered after release."""
    from xtts_trainer.stage2_gpt.export import load_for_inference
    from xtts_trainer.stage2_gpt.train import train_gpt

    base = tmp_path / "base"
    base.mkdir()
    torch.save(build_dvae().state_dict(), base / "dvae.pth")
    data = tmp_path / "data"
    data.mkdir()
    paths = []
    for i in range(4):
        p = str(data / f"c{i}.wav")
        sf.write(p, (0.3 * np.sin(np.linspace(0, 500 * (i + 1), 55125))).astype("float32"), 22050)
        paths.append(p)
    texts = ["habari ya asubuhi", "rafiki yangu", "asubuhi njema", "habari rafiki"]
    ds = write_test_dataset(str(data), paths, texts=texts, lang="sw")

    export = train_gpt(ds, str(tmp_path / "run"), str(base), vocab_path=sw_vocab, languages=["sw"], epochs=1,
                       batch_size=2, grad_accum_steps=1, lr=1e-3, eval_fraction=0.25, arch_overrides=TINY,
                       load_base_weights=False)
    summary = json.loads(Path(export, "training_summary.json").read_text(encoding="utf-8"))
    assert summary["train_clips"] == 3 and summary["eval_clips"] == 1 and "sw" in summary["languages"]
    assert summary["eval_loss_best"] < summary["eval_loss_before"] and summary["improved"]  # lr 1e-3 from random init

    assert sorted(os.listdir(export)) == ["config.json", "dvae.pth", "model.pth", "training_summary.json", "vocab.json"]
    model = load_for_inference(export)
    assert "sw" in model.tokenizer.extra_languages
    lat, spk = model.get_conditioning_latents(audio_path=[paths[0]])
    torch.manual_seed(0)
    wav = model.inference("habari ya asubuhi", "sw", lat, spk, temperature=0.7)["wav"]
    assert len(wav) > 0 and np.isfinite(np.asarray(wav)).all()

    # Phase 2 audit requirement: intermediate Stage 2 checkpoints carry spec_hash
    from xtts_trainer.models.checkpoint_io import SPEC_HASH_KEY, spec_hash, check_spec_hash, safe_torch_load
    run_dir = tmp_path / "run"
    checkpoints = list(run_dir.glob("*.pth")) + list(run_dir.glob("*/*.pth"))
    assert any(ckpt.name.startswith("checkpoint_") or ckpt.name.startswith("best_model") for ckpt in checkpoints)
    for ckpt in checkpoints:
        data = safe_torch_load(str(ckpt))
        if isinstance(data, dict) and "model" in data:
            assert data.get(SPEC_HASH_KEY) == spec_hash()
            check_spec_hash(data, str(ckpt))
