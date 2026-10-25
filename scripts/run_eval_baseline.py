"""Measure a checkpoint on the frozen eval set; saves eval/baselines/<label>_xtts_<pinned-sha12>_<voices>.json.

Phase 1 deliverable:
- Loads frozen eval_set.json.
- Synthesizes every eval sentence with the stock XTTS-v2 checkpoint, for every reference voice,
  every language in the eval set, and every seed.
- Computes WER/CER (Whisper large-v3), SECS (ECAPA2), UTMOS, duration ratio and failure counts per utterance.
- Records a manifest with git SHA, eval set hash, checkpoint and reference-voice hashes, and per-sentence values.

There is no synthetic or "calibrated" mode: a baseline is only ever produced by running the model.
If the checkpoint or any reference voice audio is missing, the script fails.
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
from typing import Optional, Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.dataset import DEFAULT_EVAL_SET_PATH, load_eval_set
from eval.harness import (
    METRICS_VERSION, EvaluationHarness, get_code_fingerprint, get_git_info, summarize_by_category, summarize_results,
)
from eval.reference_voices import DEFAULT_LOCK_PATH, DEFAULT_OUT_DIR, verify_reference_voices
from eval.target_voices import DEFAULT_TARGET_DIR, verify_target_voices
from xtts_trainer.spec import XTTS_BASE_LANGUAGES


PINNED_UPSTREAM_COMMIT = "03339e5b790b5e16ea8867a142e8bb26ec646d94"
REQUIRED_CHECKPOINT_FILES = ("model.pth", "config.json", "vocab.json")


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_reference_voices(eval_data: Dict[str, Any], reference_audio_dir: str) -> List[Dict[str, str]]:
    """Map every reference voice in the eval set to <reference_audio_dir>/<speaker_id>.wav, failing if any is missing."""
    voices = eval_data["reference_voices"]
    resolved, missing = [], []
    for voice in voices:
        path = os.path.join(reference_audio_dir, f"{voice['speaker_id']}.wav")
        if os.path.exists(path):
            resolved.append({"speaker_id": voice["speaker_id"], "language": voice.get("language", "en"), "path": path})
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            f"Missing reference voice audio for {len(missing)} of {len(voices)} eval-set voices: {missing}. "
            f"Each must be a real, held-out 6-10 s recording never used in training."
        )
    return resolved


def summaries_from_results(results_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """summary_by_language, summary_by_category (per language) and total failure counts."""
    by_language = {lang: summarize_results(rs) for lang, rs in results_by_language.items()}
    failure_counts = {k: sum(s["failure_counts"][k] for s in by_language.values()) for k in ("empty", "too_long", "nan")}
    for s in by_language.values():
        del s["failure_counts"]
    by_category = {lang: summarize_by_category(rs) for lang, rs in results_by_language.items()}
    return {"summary_by_language": by_language, "summary_by_category": by_category, "failure_counts": failure_counts}


def generate_baseline(
    checkpoint_dir: str,
    reference_audio_dir: str = DEFAULT_OUT_DIR,
    eval_set_path: str = DEFAULT_EVAL_SET_PATH,
    output_path: Optional[str] = None,
    seeds: List[int] = (42, 123, 999),
    languages: Optional[List[str]] = None,
    asr_model_name: str = "openai/whisper-large-v3",
    subset_size: Optional[int] = None,
    lock_path: str = DEFAULT_LOCK_PATH,
    label: str = "base",
    voices: str = "arctic",
    target_dir: str = DEFAULT_TARGET_DIR,
    allow_unchecked_location: bool = False,
) -> Dict[str, Any]:
    """Measure a checkpoint (stock XTTS-v2 by default) on the frozen eval set. Raises if any input is missing.

    label names the model in the manifest ("base", "tuned", ...); scripts/compare_ab.py pairs two manifests.
    voices: "arctic" (the eval set's CMU ARCTIC voices: forgetting check on unseen speakers), "target" (the
    target speaker's held-out clips from scripts/build_target_voices.py: did the fine-tune learn the voice?).
    The two are separate runs: pooled, every ungated gate in eval/gates.json would mix the forgetting check on
    unseen voices with the target speaker. Target clips are read only from DEFAULT_TARGET_DIR, the one place
    training refuses them (allow_unchecked_location: tests only).
    """
    if voices not in ("arctic", "target"):
        raise ValueError(f"voices must be arctic or target, not {voices!r}")
    if (voices == "target" and os.path.abspath(target_dir) != os.path.abspath(DEFAULT_TARGET_DIR)
            and not allow_unchecked_location):
        raise ValueError(f"target voices are read only from {DEFAULT_TARGET_DIR}: training refuses held-out clips "
                         f"listed there, and nowhere else")
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", label):
        raise ValueError(f"label {label!r} must be a plain name (letters, digits, '_', '-', '.'); it becomes a file name")
    missing_ckpt = [f for f in REQUIRED_CHECKPOINT_FILES if not os.path.exists(os.path.join(checkpoint_dir, f))]
    if missing_ckpt:
        raise FileNotFoundError(
            f"Checkpoint directory '{checkpoint_dir}' is missing {missing_ckpt}. "
            f"Fetch it with: python scripts/fetch_base_artifacts.py --dest {checkpoint_dir}"
        )

    if output_path is None:
        # One file per label and voice set, so a target run never overwrites the eval-voice run (or vice versa)
        output_path = f"eval/baselines/{label}_xtts_{PINNED_UPSTREAM_COMMIT[:12]}_{voices}.json"

    eval_data = load_eval_set(eval_set_path)
    selected = voices
    voices = []
    if selected == "arctic":
        voices = resolve_reference_voices(eval_data, reference_audio_dir)
        locked = verify_reference_voices(reference_audio_dir, lock_path)
        unlocked = sorted({v["speaker_id"] for v in voices} - set(locked))
        if unlocked:
            raise RuntimeError(f"Reference voices {unlocked} are not in the lock {lock_path}; rebuild the lock.")
    if selected == "target":
        voices += [{"speaker_id": vid, "language": None, "path": path}
                   for vid, path in sorted(verify_target_voices(target_dir).items())]
    if languages is None:
        languages = ["en"] + sorted(eval_data["target_languages"].keys())

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=os.path.join(checkpoint_dir, "model.pth"),
        vocab_path=os.path.join(checkpoint_dir, "vocab.json"),
        use_deepspeed=False,
    )
    extra = [l for l in config.languages if l.split("-")[0] not in {b.split("-")[0] for b in XTTS_BASE_LANGUAGES}]
    if extra:  # a fine-tune that registered new languages: stock VoiceBpeTokenizer would refuse them (F4)
        from xtts_trainer.frontend.text import ProjectTokenizer

        model.tokenizer = ProjectTokenizer(os.path.join(checkpoint_dir, "vocab.json"), extra)
    harness = EvaluationHarness(eval_set_path=eval_set_path, asr_model_name=asr_model_name)
    model.to(harness.device)
    model.eval()

    runs = []
    results_by_language: Dict[str, List[Dict[str, Any]]] = {}
    for lang in languages:
        for voice in voices:
            run = harness.run_benchmark(
                model=model,
                reference_audio_path=voice["path"],
                seeds=seeds,
                language=lang,
                subset_size=subset_size,
            )
            run["speaker_id"] = voice["speaker_id"]
            runs.append(run)
            results_by_language.setdefault(lang, []).extend(r for rs in run["per_seed_results"].values() for r in rs)
    manifest_summaries = summaries_from_results(results_by_language)

    manifest = {
        "label": label,
        "description": f"XTTS-v2 checkpoint '{label}' ({checkpoint_dir}), measured by synthesizing the frozen eval set",
        "pinned_upstream_commit": PINNED_UPSTREAM_COMMIT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git": get_git_info(),
        "code_fingerprint": get_code_fingerprint(),
        "eval_set_sha256": eval_data.get("sha256", ""),
        "checkpoint_sha256": {
            f: _sha256_of_file(os.path.join(checkpoint_dir, f)) for f in REQUIRED_CHECKPOINT_FILES
        },
        "voices": selected,
        "reference_voices": [
            {"speaker_id": v["speaker_id"], "sha256": _sha256_of_file(v["path"])} for v in voices
        ],
        "asr_model": asr_model_name,
        "seeds": list(seeds),
        "subset_size": subset_size,
        "languages": languages,
        "metrics_version": METRICS_VERSION,
        **manifest_summaries,
        "runs": runs,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the stock base XTTS-v2 baseline on the frozen eval set.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with model.pth, config.json, vocab.json")
    parser.add_argument("--reference_audio_dir", type=str, default=DEFAULT_OUT_DIR, help="Directory with <speaker_id>.wav for every eval-set reference voice")
    parser.add_argument("--reference_lock", type=str, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--subset_size", type=int, default=None, help="First N sentences per language (pilot runs)")
    parser.add_argument("--eval_set", type=str, default=DEFAULT_EVAL_SET_PATH)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--languages", type=str, nargs="+", default=None, help="Languages to evaluate (default: all in eval set)")
    parser.add_argument("--asr_model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--label", type=str, default="base", help="Name of this model in the manifest (base, tuned, ...)")
    parser.add_argument("--voices", choices=["arctic", "target"], default="arctic",
                        help="arctic: eval-set CMU voices; target: held-out target-speaker clips (build_target_voices.py)")
    args = parser.parse_args()

    res = generate_baseline(
        checkpoint_dir=args.checkpoint_dir,
        reference_audio_dir=args.reference_audio_dir,
        eval_set_path=args.eval_set,
        output_path=args.output,
        languages=args.languages,
        asr_model_name=args.asr_model,
        seeds=args.seeds,
        subset_size=args.subset_size,
        lock_path=args.reference_lock,
        label=args.label,
        voices=args.voices,
    )
    print("Baseline measured successfully.")
    print(json.dumps(res["summary_by_language"], indent=2))
