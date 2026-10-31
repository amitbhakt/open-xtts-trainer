"""Run the Target Voice Adaptation, Evaluation, and A/B Gating Pipeline headlessly.

Implements Phase 7 exit 2 and Phase 8 exit 1 from REMEDIATION_PLAN.md:
1. Prepares consenting target speaker audio (LibriVox Art of War, Moira Fogarty).
2. Holds out 6-10s reference clips via scripts/build_target_voices.py.
3. Fine-tunes Stage 2 GPT on dataset_train.jsonl with prompt masking (F2) & mean embedding init.
4. Evaluates base XTTS-v2 on target held-out clips.
5. Evaluates fine-tuned XTTS-v2 on target held-out clips.
6. Evaluates fine-tuned XTTS-v2 on ARCTIC voices (preservation check on unseen speakers).
7. Executes paired A/B hypothesis tests via scripts/compare_ab.py against pre-registered gates.
8. Runs and archives the Phase 5 released model identity canary.

Usage:
    python scripts/run_target_finetune_ab.py [--results-dir runs/results_target_speaker]
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"

EVAL_BATCH_SIZE = 16
# M1 candidate C2's replay set. The download cache sits in runs/, a persistent Modal volume, so the 1.36 GB archive
# is fetched once rather than on every run.
REPLAY_DIR = "runs/replay_librittsr"
REPLAY_CACHE = "runs/_cache/librittsr"
REPLAY_SELECTION = "docs/evidence/m1_stage2_preservation/replay_selection.json"
# Run 6's train/eval split. Runs at other training seeds keep it, so every run chooses its checkpoint on the same
# target clips (M1 confirmation).
SPLIT_SEED = 1


def train_command(args, train_dataset_path: str, run_dir: str) -> List[str]:
    """The Stage 2 training command for one run; options left at their defaults are not passed (Run 6's command)."""
    cmd = [
        sys.executable, "-m", "xtts_trainer.stage2_gpt.train",
        "--dataset", train_dataset_path,
        "--base_dir", "xtts_base",
        "--output_dir", run_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--lr", str(args.lr),
        "--eval_fraction", "0.1",
    ]
    if args.train_seed != 1:
        cmd += ["--seed", str(args.train_seed), "--split_seed", str(SPLIT_SEED)]
    if args.freeze:
        cmd += ["--freeze", *args.freeze]
    if args.replay:
        cmd += ["--replay", *args.replay]
    return cmd


def tree_digest(paths: List[str], root: str) -> str:
    """One SHA-256 over files' paths (relative to root) and contents: equal digests mean identical files."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda q: os.path.relpath(q, root)):
        with open(path, "rb") as f:
            digest.update(os.path.relpath(path, root).encode() + b"\0" + hashlib.sha256(f.read()).hexdigest().encode() + b"\n")
    return digest.hexdigest()


def data_identity(train_dataset_path: str, target_dir: str) -> dict:
    """Hashes that show which target data a run trained and was judged on, so candidates can be shown to share
    them. Only hashes: the held-out clips and their lock stay private."""
    from xtts_trainer.data import load_dataset

    data_dir = os.path.dirname(os.path.abspath(train_dataset_path))
    clips = [r.audio_path(data_dir) for r in load_dataset(train_dataset_path)]
    held_out = [str(q) for q in Path(target_dir).rglob("*") if q.is_file()]
    return {"train_dataset_sha256": tree_digest([train_dataset_path], data_dir),
            "train_clips": len(clips), "train_clips_sha256": tree_digest(clips, data_dir),
            "target_voices_files": len(held_out), "target_voices_sha256": tree_digest(held_out, target_dir)}


def git_tree() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def resolve_results_dir(results_dir: Optional[str], run_tag: str) -> str:
    """The results directory: given explicitly, or derived from the run tag so tagged runs never share one."""
    if results_dir:
        return results_dir
    return "runs/results_target_speaker" + (f"_{run_tag}" if run_tag else "")


def check_replay_args(replay: List[str], replay_librittsr: bool) -> None:
    """C2 trains on the pre-registered set only: no other replay files beside it."""
    if replay_librittsr and replay:
        raise SystemExit("--replay-librittsr trains on the pre-registered replay set only; drop --replay")


def replay_build_command(evidence_dir: str) -> List[str]:
    return [sys.executable, "scripts/build_replay_librittsr.py", "--output_dir", REPLAY_DIR,
            "--cache_dir", REPLAY_CACHE, "--vocab", "xtts_base/vocab.json",
            "--evidence", os.path.join(evidence_dir, "replay_selection.json")]


def check_replay_selection(built_path: str, committed_path: str = REPLAY_SELECTION) -> None:
    """C2 must train on exactly the pre-registered replay set: same clips, byte for byte (each clip's SHA-256)."""
    with open(built_path, "r", encoding="utf-8") as f:
        built = json.load(f)
    with open(committed_path, "r", encoding="utf-8") as f:
        committed = json.load(f)
    if built != committed:
        differ = sorted(k for k in set(built) | set(committed) if built.get(k) != committed.get(k))
        raise SystemExit(f"replay set differs from the committed {committed_path} in {differ}; "
                         "C2 would not train on the pre-registered set")


def should_reuse_cached_base(
    cached_data: dict,
    expected_seeds: List[int],
    expected_subset: int,
    xtts_base_dir: str = "xtts_base",
    eval_batch_size: int = EVAL_BATCH_SIZE,
) -> bool:
    """Validate whether an existing baseline manifest can be reused for A/B comparison."""
    try:
        cached_seeds = cached_data.get("seeds", [])
        cached_subset = cached_data.get("subset_size")
        cached_hashes = cached_data.get("checkpoint_sha256", {})

        # Validate checkpoint hashes against current xtts_base
        for fname in ("model.pth", "config.json", "vocab.json"):
            fpath = os.path.join(xtts_base_dir, fname)
            if not os.path.exists(fpath):
                return False
            h = hashlib.sha256()
            with open(fpath, "rb") as bf:
                for chunk in iter(lambda: bf.read(1 << 20), b""):
                    h.update(chunk)
            if cached_hashes.get(fname) != h.hexdigest():
                return False

        cached_batch = cached_data.get("batch_size")
        return bool(
            sorted(cached_seeds) == sorted(expected_seeds)
            and cached_subset == expected_subset
            and cached_data.get("languages") == ["en"]
            and cached_data.get("asr_model") == "openai/whisper-large-v3"
            and cached_batch == eval_batch_size
        )
    except (KeyError, OSError, TypeError):
        return False


def run_cmd(cmd: Union[str, List[str]], check: bool = True) -> int:
    """Run a command with live output, safely handling argument lists or shell strings."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    if isinstance(cmd, list):
        argv = [str(c) for c in cmd]
    else:
        argv = shlex.split(cmd)
    display = " ".join(shlex.quote(str(c)) for c in argv)
    print(f"\n▶ [{time.strftime('%H:%M:%S')}] {display}")
    proc = subprocess.run(argv, check=check, env=env)
    return proc.returncode


def check_artifacts(after: str):
    """Confirm every XTTS artifact in xtts_base/ still matches artifacts.lock.json."""
    if not os.path.exists("artifacts.lock.json"):
        print("artifacts.lock.json not found, skipping check_artifacts")
        return
    with open("artifacts.lock.json", "r", encoding="utf-8") as f:
        artifacts = json.load(f)
    for name, entry in artifacts.items():
        path = os.path.join("xtts_base", name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing during/after {after}")
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(
                f"{path} changed during {after}: expected {entry['sha256']}, got {digest.hexdigest()}"
            )
    print(f"✅ Base artifacts intact after {after}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Target Voice Adaptation & A/B pipeline headlessly.")
    parser.add_argument("--results-dir", default=None,
                        help="Path to store evidence and manifests (default: runs/results_target_speaker, with "
                             "_<run-tag> appended when --run-tag is given)")
    parser.add_argument("--source", default="art_of_war_short", help="Data source key in speech_data.py")
    parser.add_argument("--speaker", default="moira_fogarty", help="Speaker name in the source")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for Stage 2 GPT")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per forward pass")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Grad accumulation steps (effective batch >= 32)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for GPT fine-tuning")
    parser.add_argument("--subset-size", type=int, default=12, help="Eval subset size per language (default: 12)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999], help="Eval seeds")
    parser.add_argument("--skip-train", action="store_true", help="Skip fine-tuning if checkpoint already exported")
    parser.add_argument("--train-seed", type=int, default=1, help="Stage 2 training seed (data split and order)")
    parser.add_argument("--freeze", nargs="+", default=[], choices=["conditioning"], help="GPT parts held at released values")
    parser.add_argument("--replay", nargs="+", default=[], help="Replay dataset.jsonl files (other speakers, same language)")
    parser.add_argument("--replay-librittsr", action="store_true",
                        help="M1 C2: build the pre-registered LibriTTS-R replay set, check it against the committed "
                             "selection, and train with it")
    parser.add_argument("--run-tag", default="",
                        help="Suffix for the training directory and the default results directory, so candidate runs "
                             "don't overwrite each other")
    args = parser.parse_args()
    check_replay_args(args.replay, args.replay_librittsr)

    if args.speaker != "moira_fogarty":
        raise ValueError(
            f"Speaker '{args.speaker}' is not supported with pre-registered gates. "
            "Only 'moira_fogarty' has pre-registered quality gates in eval/gates_target.json."
        )

    t_start = time.time()
    results_dir = os.path.abspath(resolve_results_dir(args.results_dir, args.run_tag))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("eval/baselines", exist_ok=True)
    evidence_dir = os.path.join(results_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 Starting Open-XTTS-Trainer Target Voice Adaptation & A/B Pipeline")
    print(f"Target Speaker: {args.speaker} (Source: {args.source})")
    print(f"Results Directory: {results_dir}")
    print(f"Subset Size: {args.subset_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Grad Accum: {args.grad_accum_steps}")
    print("=" * 70)

    # 1. Record environment and fingerprint
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    import torch
    import torchaudio
    import transformers
    from eval.harness import get_code_fingerprint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = {
        # On Modal the commit is the container's own (no .git is uploaded); the tree hash and the code fingerprint
        # identify the real commit
        "git": {"commit": commit, "tree": git_tree()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "code_fingerprint": get_code_fingerprint(),
    }
    with open(os.path.join(evidence_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(f"Environment: GPU={env['gpu']}, CUDA={env['cuda']}, Torch={env['torch']}")

    # 2. Fetch and verify base XTTS artifacts
    print("\n📦 Step 1: Fetching and verifying base XTTS artifacts...")
    run_cmd([
        sys.executable, "scripts/fetch_base_artifacts.py",
        "--dest", "xtts_base", "--check_mel_stats",
        "--report", os.path.join(evidence_dir, "artifact_verification.json"),
    ])
    check_artifacts("fetch_base_artifacts")

    # 3. Ensure reference voices exist
    print("\n🎙️ Step 2: Ensuring ARCTIC reference voices exist...")
    run_cmd([sys.executable, "scripts/fetch_reference_voices.py"])

    # 4. Prepare target speaker speech data
    data_dir = f"data_{args.speaker}"
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    print(f"\n🔊 Step 3: Preparing speech data for {args.speaker} from source '{args.source}'...")
    run_cmd([
        sys.executable, "-m", "xtts_trainer.speech_data",
        "--source", args.source,
        "--output_dir", data_dir,
        "--vocab", "xtts_base/vocab.json",
        "--device", device,
    ])

    # 5. Build target voices (hold out reference clips, produce dataset_train.jsonl)
    print(f"\n🎯 Step 4: Holding out reference clips for target speaker '{args.speaker}'...")
    run_cmd([
        sys.executable, "scripts/build_target_voices.py",
        "--dataset", dataset_path,
        "--speakers", args.speaker,
    ])
    train_dataset_path = os.path.join(data_dir, "dataset_train.jsonl")
    assert os.path.exists(train_dataset_path), f"Expected {train_dataset_path} after build_target_voices"

    # Validate dataset_train.jsonl with target holdout lock
    train_val_report_path = os.path.join(evidence_dir, "validation_report.json")
    run_cmd([
        sys.executable, "-m", "xtts_trainer.data.validate",
        train_dataset_path,
        "--vocab", "xtts_base/vocab.json",
        "--report", train_val_report_path,
    ])

    with open(os.path.join(evidence_dir, "data_identity.json"), "w", encoding="utf-8") as f:
        json.dump(data_identity(train_dataset_path, "eval/target_voices"), f, indent=2)

    # Copy prep evidence and target voices
    shutil.copy(os.path.join(data_dir, "prep_report.json"), os.path.join(evidence_dir, "prep_report.json"))
    shutil.copytree("eval/target_voices", os.path.join(evidence_dir, "target_voices"), dirs_exist_ok=True)

    # 6. Stage 2 GPT fine-tuning
    if args.replay_librittsr:
        print("\n🔁 Building the pre-registered LibriTTS-R replay set (M1 C2)...")
        run_cmd(replay_build_command(evidence_dir))
        check_replay_selection(os.path.join(evidence_dir, "replay_selection.json"))
        args.replay = [os.path.join(REPLAY_DIR, "dataset.jsonl")]
    run_dir = f"runs/{args.speaker}_finetune" + (f"_{args.run_tag}" if args.run_tag else "")
    export_dir = os.path.join(run_dir, "export")
    if not args.skip_train:
        print(f"\n🧠 Step 5: Fine-tuning Stage 2 GPT on {train_dataset_path}...")
        run_cmd(train_command(args, train_dataset_path, run_dir))
        check_artifacts("stage2_gpt_train")
    else:
        print(f"\n⏩ Skipping Stage 2 training (--skip-train), using {export_dir}")

    exported_model_path = os.path.join(export_dir, "model.pth")
    if not os.path.exists(exported_model_path):
        raise FileNotFoundError(f"Missing exported model in {export_dir}")
    shutil.copy(os.path.join(export_dir, "training_summary.json"), os.path.join(evidence_dir, "training_summary.json"))

    # 7. Evaluate Base Model on Target Voice (held-out clips)
    base_target_manifest = os.path.join(results_dir, f"base_xtts_s{args.subset_size}_target.json")
    seeds_args = list(map(str, args.seeds))
    subset_args = ["--subset_size", str(args.subset_size)] if args.subset_size else []

    print(f"\n📊 Step 6: Evaluating BASE model on target held-out clips...")
    base_eval_cmd = [
        sys.executable, "scripts/run_eval_baseline.py",
        "--checkpoint_dir", "xtts_base",
        "--voices", "target",
        "--languages", "en",
        "--label", "base",
        "--seeds", *seeds_args,
        *subset_args,
        "--batch_size", str(EVAL_BATCH_SIZE),
        "--output", base_target_manifest,
    ]
    run_cmd(base_eval_cmd)

    # 8. Evaluate Tuned Model on Target Voice (held-out clips)
    tuned_target_manifest = os.path.join(results_dir, f"tuned_xtts_s{args.subset_size}_target.json")
    print(f"\n📊 Step 7: Evaluating TUNED model on target held-out clips...")
    tuned_eval_cmd = [
        sys.executable, "scripts/run_eval_baseline.py",
        "--checkpoint_dir", export_dir,
        "--voices", "target",
        "--languages", "en",
        "--label", "tuned",
        "--seeds", *seeds_args,
        *subset_args,
        "--batch_size", str(EVAL_BATCH_SIZE),
        "--output", tuned_target_manifest,
    ]
    run_cmd(tuned_eval_cmd)

    # 9. Evaluate Tuned Model on ARCTIC voices (preservation check on base voices)
    tuned_arctic_manifest = os.path.join(results_dir, f"tuned_xtts_s{args.subset_size}_arctic_en.json")
    print(f"\n📊 Step 8: Evaluating TUNED model on ARCTIC voices (preservation check)...")
    tuned_arctic_eval_cmd = [
        sys.executable, "scripts/run_eval_baseline.py",
        "--checkpoint_dir", export_dir,
        "--voices", "arctic",
        "--languages", "en",
        "--label", "tuned",
        "--seeds", *seeds_args,
        *subset_args,
        "--batch_size", str(EVAL_BATCH_SIZE),
        "--output", tuned_arctic_manifest,
    ]
    run_cmd(tuned_arctic_eval_cmd)

    # 10. Run Paired A/B Comparisons against Pre-Registered Gates
    print("\n⚖️ Step 9: Running paired A/B comparisons against pre-registered gates...")

    # Gate 1: Target Adaptation Gate (dynamically built for args.speaker)
    target_gates_path = os.path.join(results_dir, "gates_target.json")
    with open("eval/gates_target.json", "r", encoding="utf-8") as f:
        target_gates_data = json.load(f)
    for g in target_gates_data.get("gates", []):
        if "voices" in g:
            g["voices"] = [f"target_{args.speaker}"]
    with open(target_gates_path, "w", encoding="utf-8") as f:
        json.dump(target_gates_data, f, indent=2)

    target_ab_report_path = os.path.join(results_dir, "ab_report_target.json")
    if os.path.exists(target_ab_report_path):
        os.remove(target_ab_report_path)
    ret_target = run_cmd([
        sys.executable, "scripts/compare_ab.py",
        base_target_manifest, tuned_target_manifest,
        "--gates", target_gates_path,
        "--report", target_ab_report_path,
    ], check=False)
    if not os.path.exists(target_ab_report_path):
        raise RuntimeError(
            f"compare_ab failed to produce fresh target report at {target_ab_report_path} (exit code: {ret_target})"
        )
    try:
        with open(target_ab_report_path, "r", encoding="utf-8") as f:
            target_rep = json.load(f)
        if "passed" not in target_rep:
            raise KeyError("missing 'passed' in target report")
    except (json.JSONDecodeError, OSError, KeyError) as e:
        raise RuntimeError(f"Invalid target report produced at {target_ab_report_path}: {e}") from e

    # Gate 2: ARCTIC Preservation Gate (compared against locked Run 5 English baseline if seeds, subset, and checkpoint hashes match, else generate)
    base_arctic_manifest = f"eval/baselines/base_xtts_s{args.subset_size}_en.json"
    reuse_cached_base = False
    if os.path.exists(base_arctic_manifest):
        try:
            with open(base_arctic_manifest, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            reuse_cached_base = should_reuse_cached_base(
                cached_data=cached_data,
                expected_seeds=args.seeds,
                expected_subset=args.subset_size,
                eval_batch_size=EVAL_BATCH_SIZE,
            )
        except (json.JSONDecodeError, OSError):
            reuse_cached_base = False

    if not reuse_cached_base:
        base_arctic_manifest = os.path.join(results_dir, f"base_xtts_s{args.subset_size}_arctic_en.json")
        print(f"Generating base ARCTIC manifest matching requested seeds {args.seeds}: {base_arctic_manifest}...")
        run_cmd([
            sys.executable, "scripts/run_eval_baseline.py",
            "--checkpoint_dir", "xtts_base",
            "--voices", "arctic",
            "--languages", "en",
            "--label", "base",
            "--seeds", *seeds_args,
            *subset_args,
            "--batch_size", str(EVAL_BATCH_SIZE),
            "--output", base_arctic_manifest,
        ])

    arctic_gates_path = "eval/gates.json"
    preservation_ab_report_path = os.path.join(results_dir, "ab_report_preservation.json")
    if os.path.exists(preservation_ab_report_path):
        os.remove(preservation_ab_report_path)
    ret_pres = run_cmd([
        sys.executable, "scripts/compare_ab.py",
        base_arctic_manifest, tuned_arctic_manifest,
        "--gates", arctic_gates_path,
        "--report", preservation_ab_report_path,
    ], check=False)
    if not os.path.exists(preservation_ab_report_path):
        raise RuntimeError(
            f"compare_ab failed to produce fresh preservation report at {preservation_ab_report_path} (exit code: {ret_pres})"
        )
    try:
        with open(preservation_ab_report_path, "r", encoding="utf-8") as f:
            pres_rep = json.load(f)
        if "passed" not in pres_rep:
            raise KeyError("missing 'passed' in preservation report")
    except (json.JSONDecodeError, OSError, KeyError) as e:
        raise RuntimeError(f"Invalid preservation report produced at {preservation_ab_report_path}: {e}") from e

    # 11. Run Phase 5 Canary: Identity Canary on Released Weights
    print("\n🐤 Step 10: Running Phase 5 Identity Canary on released weights...")
    identity_xml_path = os.path.join(evidence_dir, "identity_canary.xml")
    run_cmd([
        sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
        "tests/test_invariants.py", "-k", "test_identity_canary_released_model",
        f"--junitxml={identity_xml_path}",
    ])

    # 12. Final Reporting
    t_elapsed = time.time() - t_start
    with open(target_ab_report_path, "r", encoding="utf-8") as f:
        target_rep = json.load(f)
    with open(preservation_ab_report_path, "r", encoding="utf-8") as f:
        pres_rep = json.load(f)

    all_passed = bool(target_rep.get("passed")) and bool(pres_rep.get("passed"))

    print("\n" + "=" * 70)
    print(f"🏁 Target Voice Adaptation & A/B Pipeline Finished in {t_elapsed / 60:.1f} minutes")
    print("=" * 70)
    print(f"Target Voice Adaptation Gate Status: {'✅ PASSED' if target_rep.get('passed') else '❌ FAILED'}")
    print(f"General Speaker Preservation Status: {'✅ PASSED' if pres_rep.get('passed') else '❌ FAILED'}")
    print(f"Reports saved to:")
    print(f"  - Target A/B:       {target_ab_report_path}")
    print(f"  - Preservation A/B: {preservation_ab_report_path}")
    print(f"  - Canary XML:       {identity_xml_path}")
    print("=" * 70)

    if not all_passed:
        print("\n❌ One or more pre-registered quality gates failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
