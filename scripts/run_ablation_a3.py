"""Ablation A3 Runner: Conditioning Masking Impact (F2 Bug Verification).

Implements Ablation Study A3 from REMEDIATION_PLAN.md §11.5 and eval/pre_registered_ablations.json:
- Evaluates the empirical effect of upstream batch conditioning masking vs. per-sample conditioning masking.
- Uses batched Whisper ASR evaluation (--batch_size 16) by default across all evaluation steps.
- Trains Variant 1 (upstream masking) vs Variant 2 (per-sample masking) on target speaker audio.
- Performs paired evaluation across random seeds and computes 95% bootstrap confidence intervals.
- Gated via scripts/compare_ab.py --ablation A3 against pre-registered decision rules.

Usage:
    # Dry-run validation of data, checkpoints, and execution plan
    python scripts/run_ablation_a3.py --dry_run

    # Full execution across 3 seeds on GPU
    python scripts/run_ablation_a3.py --seeds 1 2 3 --output_dir runs/ablations/a3
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.compare_ab import compare_ab, load_ablation_gates

logger = logging.getLogger("ablation_a3")
EVAL_BATCH_SIZE = 16


def run_cmd(cmd: List[str], check: bool = True) -> int:
    cmd_str = " ".join(cmd)
    logger.info("Executing: %s", cmd_str)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    res = subprocess.run(cmd, env=env, check=check)
    return res.returncode


def merge_manifests(seed_manifests: List[Dict[str, Any]], label: str, output_path: str) -> Dict[str, Any]:
    """Merge per-seed evaluation manifests into a unified multi-seed manifest."""
    from scripts.run_eval_baseline import summaries_from_results

    if not seed_manifests:
        raise ValueError("seed_manifests cannot be empty")
    if len(seed_manifests) == 1:
        merged = dict(seed_manifests[0])
        merged["label"] = label
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        return merged

    merged = dict(seed_manifests[0])
    merged["label"] = label
    merged["seeds"] = [int(s) for m in seed_manifests for s in m.get("seeds", [])]
    merged["runs"] = []

    for r_idx in range(len(seed_manifests[0]["runs"])):
        run = dict(seed_manifests[0]["runs"][r_idx])
        run["per_seed_results"] = {}
        for m in seed_manifests:
            if r_idx < len(m.get("runs", [])):
                run["per_seed_results"].update(m["runs"][r_idx].get("per_seed_results", {}))
        merged["runs"].append(run)

    results_by_language: Dict[str, List[Dict[str, Any]]] = {}
    for r in merged["runs"]:
        lang = r.get("language") or "en"
        results_by_language.setdefault(lang, []).extend(
            item for rs in r["per_seed_results"].values() for item in rs
        )
    merged.update(summaries_from_results(results_by_language))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged


def resolve_dataset(dataset_arg: Optional[str] = None, dry_run: bool = False, source: str = "art_of_war_short") -> str:
    """Resolve a valid dataset_train.jsonl with held-out target voice locks."""
    target_lock = os.path.join(REPO_ROOT, "eval", "target_voices", "target_voices.lock.json")
    if dataset_arg is not None:
        if not os.path.exists(dataset_arg):
            raise FileNotFoundError(f"Supplied explicit --dataset does not exist: {dataset_arg}")
        if not os.path.exists(target_lock):
            raise FileNotFoundError(
                f"Target voice lock not found at {target_lock}. Cannot verify explicit --dataset: {dataset_arg}"
            )
        from eval.target_voices import verify_target_voices
        try:
            verify_target_voices(os.path.join(REPO_ROOT, "eval", "target_voices"))
        except (RuntimeError, FileNotFoundError, ValueError, KeyError, TypeError) as e:
            raise RuntimeError(
                f"Target voice lock verification failed for explicit --dataset {dataset_arg}: {e}"
            ) from e
        return os.path.abspath(dataset_arg)

    candidates = [
        os.path.join(REPO_ROOT, "data_moira_fogarty", "dataset_train.jsonl"),
        os.path.join(REPO_ROOT, "runs", "results_target_speaker_s20", "evidence", "dataset_train.jsonl"),
        os.path.join(REPO_ROOT, "my_data", "dataset_train.jsonl"),
        os.path.join(REPO_ROOT, "sample_audio", "dataset_train.jsonl"),
    ]
    train_candidate = None
    for c in candidates:
        if c and os.path.exists(c):
            train_candidate = os.path.abspath(c)
            break

    target_lock = os.path.join(REPO_ROOT, "eval", "target_voices", "target_voices.lock.json")
    if train_candidate and os.path.exists(target_lock):
        try:
            from eval.target_voices import verify_target_voices
            verify_target_voices(os.path.join(REPO_ROOT, "eval", "target_voices"))
            return train_candidate
        except (RuntimeError, FileNotFoundError, ValueError, KeyError, TypeError) as e:
            logger.warning("Target voices lock verification failed (%s); will rebuild.", e)

    if dry_run:
        if train_candidate and os.path.exists(train_candidate):
            return train_candidate
        fallback = os.path.abspath(os.path.join(REPO_ROOT, "data_moira_fogarty", "dataset_train.jsonl"))
        return fallback

    data_dir = os.path.dirname(train_candidate) if train_candidate else os.path.join(REPO_ROOT, "data_moira_fogarty")
    dataset_path = os.path.join(data_dir, "dataset.jsonl")

    if not os.path.exists(dataset_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Preparing target speech data from source %s on device %s...", source, device)
        run_cmd([
            sys.executable, "-m", "xtts_trainer.speech_data",
            "--source", source,
            "--output_dir", data_dir,
            "--vocab", "xtts_base/vocab.json",
            "--device", device,
        ])

    run_cmd([
        sys.executable, "scripts/build_target_voices.py",
        "--dataset", dataset_path,
        "--speakers", "moira_fogarty",
    ])
    train_dataset = os.path.join(data_dir, "dataset_train.jsonl")
    if os.path.exists(train_dataset):
        return os.path.abspath(train_dataset)

    raise FileNotFoundError(
        f"Failed to resolve or build target voices dataset from {source}."
    )


def run_ablation_a3(
    dataset_path: Optional[str] = None,
    base_dir: str = "xtts_base",
    output_dir: str = "runs/ablations/a3",
    seeds: Optional[List[int]] = None,
    epochs: int = 3,
    batch_size: int = 4,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    subset_size: int = 20,
    dry_run: bool = False,
    skip_training: bool = False,
    voices: str = "target",
) -> dict:
    seeds = seeds or [1, 2, 3]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    gates = load_ablation_gates("A3")

    plan = {
        "ablation_id": "A3",
        "title": gates.get("title"),
        "decision_rule": gates.get("decision_rule"),
        "dataset": dataset_path,
        "base_dir": base_dir,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_subset_size": subset_size,
        "voices": voices,
        "variants": ["upstream", "per_sample"],
    }

    print("\n" + "=" * 70)
    print(" Ablation A3: Conditioning Masking Impact (F2 Bug Verification)")
    print("=" * 70)
    print(f"Dataset:            {dataset_path or '(not required for base evaluation)'}")
    print(f"Base Directory:     {base_dir}")
    print(f"Voices:             {voices} (N={subset_size * len(seeds) * (5 if voices in ('base', 'arctic') else 1)})")
    print(f"Random Seeds:       {seeds}")
    print(f"Epochs per run:     {epochs}")
    print(f"Eval Batch Size:    {eval_batch_size} (batched Whisper inference)")
    print(f"Pre-Registered Decision Rule:\n  {gates.get('decision_rule')}")
    print("=" * 70 + "\n")

    if dry_run:
        prereqs_ok = True
        missing = []
        if not os.path.exists(base_dir):
            prereqs_ok = False
            missing.append(base_dir)
        if not skip_training and (not dataset_path or not os.path.exists(dataset_path)):
            prereqs_ok = False
            missing.append(str(dataset_path))
        if not prereqs_ok:
            print(f"[DRY RUN] Prerequisite check failed: missing {missing}")
            return {"dry_run": True, "passed": False, "plan": plan, "missing": missing}
        print("[DRY RUN] All prerequisites verified. Plan is valid.")
        return {"dry_run": True, "passed": True, "plan": plan}

    variants = {
        "upstream": os.path.join(output_dir, "upstream"),
        "per_sample": os.path.join(output_dir, "per_sample"),
    }

    # 1. Train both variants if not skipped
    for mode, out_path in variants.items():
        if skip_training:
            logger.info("Skipping training for variant %s (--skip_training)", mode)
            continue
        print(f"\n>>> Training Variant: {mode.upper()} masking across seeds {seeds}...")
        for s in seeds:
            seed_out = os.path.join(out_path, f"seed_{s}")
            if os.path.exists(os.path.join(seed_out, "export", "model.pth")):
                logger.info("Checkpoint already exists for %s seed %d, skipping training.", mode, s)
                continue
            run_cmd([
                sys.executable, "-m", "xtts_trainer.stage2_gpt.train",
                "--dataset", dataset_path,
                "--base_dir", base_dir,
                "--output_dir", seed_out,
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--seed", str(s),
                "--masking_mode", mode,
            ])

    # 2. Evaluate both variants with batched Whisper inference
    manifests = {}
    voice_tag = f"_{voices}" if voices != "target" else ""
    eval_voices = "arctic" if voices in ("base", "arctic") else voices
    for mode, out_path in variants.items():
        manifest_path = os.path.join(output_dir, f"manifest_{mode}{voice_tag}_s{subset_size}.json")
        manifests[mode] = manifest_path
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if (
                    cached_data.get("seeds") == seeds
                    and cached_data.get("subset_size") == subset_size
                    and cached_data.get("batch_size") == eval_batch_size
                    and cached_data.get("voices") in (voices, eval_voices)
                ):
                    logger.info("Reusing existing evaluation manifest: %s", manifest_path)
                    continue
                else:
                    logger.warning(
                        "Manifest %s inputs mismatch (seeds=%s, subset=%s, batch=%s, voices=%s); re-evaluating.",
                        manifest_path, cached_data.get("seeds"), cached_data.get("subset_size"),
                        cached_data.get("batch_size"), cached_data.get("voices"),
                    )
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
                logger.warning("Failed to validate cached manifest %s (%s); re-evaluating.", manifest_path, e)

        if len(seeds) > 1:
            seed_exports = [os.path.join(out_path, f"seed_{s}", "export") for s in seeds]
            missing_exports = [p for p in seed_exports if not os.path.exists(p)]
            if missing_exports:
                raise FileNotFoundError(
                    f"Incomplete seed exports for variant '{mode}': missing {missing_exports}. "
                    "All seeds must be trained and exported before multi-seed evaluation."
                )
            seed_manifest_data = []
            for s, s_export in zip(seeds, seed_exports):
                s_manifest_path = os.path.join(output_dir, f"manifest_{mode}{voice_tag}_seed_{s}_s{subset_size}.json")
                if not os.path.exists(s_manifest_path):
                    print(f"\n>>> Evaluating {mode.upper()} seed {s} ({s_export}) with voices '{eval_voices}', batch size {eval_batch_size}...")
                    run_cmd([
                        sys.executable, "scripts/run_eval_baseline.py",
                        "--checkpoint_dir", s_export,
                        "--voices", eval_voices,
                        "--languages", "en",
                        "--label", f"a3_{mode}_{voices}_s{s}",
                        "--seeds", str(s),
                        "--subset_size", str(subset_size),
                        "--batch_size", str(eval_batch_size),
                        "--output", s_manifest_path,
                    ])
                with open(s_manifest_path, "r", encoding="utf-8") as f:
                    seed_manifest_data.append(json.load(f))
            merge_manifests(seed_manifest_data, label=f"a3_{mode}_{voices}", output_path=manifest_path)
        else:
            s = seeds[0]
            s_export = os.path.join(out_path, f"seed_{s}", "export")
            if not os.path.exists(s_export):
                s_export = os.path.join(out_path, "export")
            if not os.path.exists(s_export):
                raise FileNotFoundError(
                    f"No trained checkpoint found for variant '{mode}' in {out_path}. "
                    "Ensure training was not skipped without existing models."
                )

            print(f"\n>>> Evaluating {mode.upper()} model ({s_export}) with voices '{eval_voices}', batch size {eval_batch_size}...")
            run_cmd([
                sys.executable, "scripts/run_eval_baseline.py",
                "--checkpoint_dir", s_export,
                "--voices", eval_voices,
                "--languages", "en",
                "--label", f"a3_{mode}_{voices}",
                "--seeds", str(s),
                "--subset_size", str(subset_size),
                "--batch_size", str(eval_batch_size),
                "--output", manifest_path,
            ])

    # 3. Execute paired A/B comparison
    print("\n>>> Running Paired A/B Hypothesis Testing against Pre-Registered Gates...")
    with open(manifests["upstream"], "r", encoding="utf-8") as f:
        upstream_manifest = json.load(f)
    with open(manifests["per_sample"], "r", encoding="utf-8") as f:
        persample_manifest = json.load(f)

    report = compare_ab(upstream_manifest, persample_manifest, gates)
    report_file = os.path.join(output_dir, f"ab_report_{voices}.json" if voices != "target" else "ab_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")
    print(f"Paired comparisons: {report['pairs']}")
    for g in report["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  [{status}] {g['metric']} {g['rule']} margin {g.get('margin', 0)}")

    passed_str = "PASSED" if report["passed"] else "FAILED"
    print(f"\nAblation A3 ({voices}) Decision: {passed_str}\n")
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run Ablation A3: Conditioning Masking Impact (F2 Bug Verification).")
    parser.add_argument("--dataset", help="Path to dataset_train.jsonl")
    parser.add_argument("--base_dir", default="xtts_base", help="Base model directory (model.pth, dvae.pth, vocab.json)")
    parser.add_argument("--output_dir", default="runs/ablations/a3", help="Output directory for ablation manifests")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Random seeds (default: 1 2 3)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per variant (default: 3)")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE,
                        help=f"Whisper ASR evaluation batch size (default: {EVAL_BATCH_SIZE})")
    parser.add_argument("--subset_size", type=int, default=20, help="Eval subset size (default: 20)")
    parser.add_argument("--dry_run", action="store_true", help="Validate plan without running training or eval")
    parser.add_argument("--skip_training", action="store_true", help="Skip training if checkpoints already exist")
    parser.add_argument("--voices", choices=["target", "base", "arctic"], default="target",
                        help="Voices to evaluate: 'target' (Moira Fogarty, N=60) or 'base'/'arctic' (5 CMU ARCTIC voices, N=300)")
    args = parser.parse_args()

    dataset_path = None
    if not args.skip_training:
        dataset_path = resolve_dataset(args.dataset, dry_run=args.dry_run)
    else:
        dataset_path = args.dataset

    report = run_ablation_a3(
        dataset_path=dataset_path,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        subset_size=args.subset_size,
        dry_run=args.dry_run,
        skip_training=args.skip_training,
        voices=args.voices,
    )
    sys.exit(0 if report.get("passed", False) else 1)


if __name__ == "__main__":
    main()
