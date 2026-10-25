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
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"


def run_cmd(cmd: str, check: bool = True) -> int:
    """Run a shell command with pipefail and live output."""
    print(f"\n▶ [{time.strftime('%H:%M:%S')}] {cmd}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=check, env=env)
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
        assert os.path.exists(path), f"{path} missing during/after {after}"
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        assert digest.hexdigest() == entry["sha256"], (
            f"{path} changed during {after}: expected {entry['sha256']}, got {digest.hexdigest()}"
        )
    print(f"✅ Base artifacts intact after {after}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Target Voice Adaptation & A/B pipeline headlessly.")
    parser.add_argument("--results-dir", default="runs/results_target_speaker", help="Path to store evidence and manifests")
    parser.add_argument("--source", default="art_of_war_short", help="Data source key in speech_data.py")
    parser.add_argument("--speaker", default="moira_fogarty", help="Speaker name in the source")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for Stage 2 GPT")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per forward pass")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Grad accumulation steps (effective batch >= 32)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for GPT fine-tuning")
    parser.add_argument("--subset-size", type=int, default=12, help="Eval subset size per language (default: 12)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999], help="Eval seeds")
    parser.add_argument("--skip-train", action="store_true", help="Skip fine-tuning if checkpoint already exported")
    args = parser.parse_args()

    t_start = time.time()
    results_dir = os.path.abspath(args.results_dir)
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
        "git": {"commit": commit},
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
    run_cmd(
        "python scripts/fetch_base_artifacts.py --dest xtts_base --check_mel_stats "
        f"--report {evidence_dir}/artifact_verification.json"
    )
    check_artifacts("fetch_base_artifacts")

    # 3. Ensure reference voices exist
    print("\n🎙️ Step 2: Ensuring ARCTIC reference voices exist...")
    run_cmd("python scripts/fetch_reference_voices.py")

    # 4. Prepare target speaker speech data
    data_dir = f"data_{args.speaker}"
    dataset_path = f"{data_dir}/dataset.jsonl"
    print(f"\n🔊 Step 3: Preparing speech data for {args.speaker} from source '{args.source}'...")
    run_cmd(
        f"python -m xtts_trainer.speech_data --source {args.source} --output_dir {data_dir} "
        f"--vocab xtts_base/vocab.json --device {device}"
    )

    # 5. Build target voices (hold out reference clips, produce dataset_train.jsonl)
    print(f"\n🎯 Step 4: Holding out reference clips for target speaker '{args.speaker}'...")
    run_cmd(f"python scripts/build_target_voices.py --dataset {dataset_path} --speakers {args.speaker}")
    train_dataset_path = f"{data_dir}/dataset_train.jsonl"
    assert os.path.exists(train_dataset_path), f"Expected {train_dataset_path} after build_target_voices"

    # Copy prep evidence and target voices
    run_cmd(f"cp {data_dir}/prep_report.json {data_dir}/validation_report.json {evidence_dir}/")
    run_cmd(f"cp -r eval/target_voices {evidence_dir}/")

    # 6. Stage 2 GPT fine-tuning
    run_dir = f"runs/{args.speaker}_finetune"
    export_dir = f"{run_dir}/export"
    if not args.skip_train:
        print(f"\n🧠 Step 5: Fine-tuning Stage 2 GPT on {train_dataset_path}...")
        run_cmd(
            f"python -m xtts_trainer.stage2_gpt.train --dataset {train_dataset_path} --base_dir xtts_base "
            f"--output_dir {run_dir} --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--grad_accum_steps {args.grad_accum_steps} --lr {args.lr} --eval_fraction 0.1"
        )
        check_artifacts("stage2_gpt_train")
    else:
        print(f"\n⏩ Skipping Stage 2 training (--skip-train), using {export_dir}")

    assert os.path.exists(f"{export_dir}/model.pth"), f"Missing exported model in {export_dir}"
    run_cmd(f"cp {export_dir}/training_summary.json {evidence_dir}/")

    # 7. Evaluate Base Model on Target Voice (held-out clips)
    base_target_manifest = f"{results_dir}/base_xtts_s{args.subset_size}_target.json"
    seeds_arg = " ".join(map(str, args.seeds))
    subset_arg = f"--subset_size {args.subset_size}" if args.subset_size else ""

    print(f"\n📊 Step 6: Evaluating BASE model on target held-out clips...")
    run_cmd(
        f"python scripts/run_eval_baseline.py --checkpoint_dir xtts_base --voices target --languages en "
        f"--label base --seeds {seeds_arg} {subset_arg} --output {base_target_manifest}"
    )

    # 8. Evaluate Tuned Model on Target Voice (held-out clips)
    tuned_target_manifest = f"{results_dir}/tuned_xtts_s{args.subset_size}_target.json"
    print(f"\n📊 Step 7: Evaluating TUNED model on target held-out clips...")
    run_cmd(
        f"python scripts/run_eval_baseline.py --checkpoint_dir {export_dir} --voices target --languages en "
        f"--label tuned --seeds {seeds_arg} {subset_arg} --output {tuned_target_manifest}"
    )

    # 9. Evaluate Tuned Model on ARCTIC voices (preservation check on base voices)
    tuned_arctic_manifest = f"{results_dir}/tuned_xtts_s{args.subset_size}_arctic_en.json"
    print(f"\n📊 Step 8: Evaluating TUNED model on ARCTIC voices (preservation check)...")
    run_cmd(
        f"python scripts/run_eval_baseline.py --checkpoint_dir {export_dir} --voices arctic --languages en "
        f"--label tuned --seeds {seeds_arg} {subset_arg} --output {tuned_arctic_manifest}"
    )

    # 10. Run Paired A/B Comparisons against Pre-Registered Gates
    print("\n⚖️ Step 9: Running paired A/B comparisons against pre-registered gates...")

    # Gate 1: Target Adaptation Gate
    target_gates_path = "eval/gates_target.json"
    target_ab_report_path = f"{results_dir}/ab_report_target.json"
    run_cmd(
        f"python scripts/compare_ab.py {base_target_manifest} {tuned_target_manifest} "
        f"--gates {target_gates_path} --report {target_ab_report_path}"
    )

    # Gate 2: ARCTIC Preservation Gate (compared against locked Run 5 English baseline if available, or generated base)
    base_arctic_manifest = f"eval/baselines/base_xtts_s{args.subset_size}_en.json"
    if not os.path.exists(base_arctic_manifest):
        print(f"Generating base ARCTIC manifest: {base_arctic_manifest}...")
        run_cmd(
            f"python scripts/run_eval_baseline.py --checkpoint_dir xtts_base --voices arctic --languages en "
            f"--label base --seeds {seeds_arg} {subset_arg} --output {base_arctic_manifest}"
        )

    arctic_gates_path = "eval/gates.json"
    preservation_ab_report_path = f"{results_dir}/ab_report_preservation.json"
    run_cmd(
        f"python scripts/compare_ab.py {base_arctic_manifest} {tuned_arctic_manifest} "
        f"--gates {arctic_gates_path} --report {preservation_ab_report_path}"
    )

    # 11. Run Phase 5 Canary: Identity Canary on Released Weights
    print("\n🐤 Step 10: Running Phase 5 Identity Canary on released weights...")
    identity_xml_path = f"{evidence_dir}/identity_canary.xml"
    run_cmd(
        f"python -m pytest -v -p no:cacheprovider tests/test_invariants.py -k test_identity_canary_released_model "
        f"--junitxml={identity_xml_path}"
    )

    # 12. Final Reporting
    t_elapsed = time.time() - t_start
    with open(target_ab_report_path, "r", encoding="utf-8") as f:
        target_rep = json.load(f)
    with open(preservation_ab_report_path, "r", encoding="utf-8") as f:
        pres_rep = json.load(f)

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


if __name__ == "__main__":
    main()
