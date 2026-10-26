"""Phase 8 Exit 2: Baseline Reproducibility Verification Runner.

Runs evaluation on the baseline model with GPU batching (default batch_size=16),
compares the resulting metrics against the published Run 5 baseline (base_xtts_s12_index.json),
and verifies that all metrics (WER, CER, SECS, UTMOS, duration ratio) reproduce within
their 95% confidence intervals.

Usage:
    # Run locally or inside Modal GPU container:
    python scripts/check_baseline_reproducibility.py --batch_size 16 --subset_size 12

    # Fast verification on a single language:
    python scripts/check_baseline_reproducibility.py --languages en --batch_size 16
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_INDEX_PATH = os.path.join(REPO_ROOT, "eval", "baselines", "base_xtts_s12_index.json")
DEFAULT_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "xtts_base")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "runs", "reproducibility_check")
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "reproducibility_report.json")


def run_cmd(cmd: List[str]) -> None:
    print(f"\n▶ [{time.strftime('%H:%M:%S')}] {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Verify baseline reproducibility within 95% CI bounds.")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR, help="Path to xtts_base weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Whisper ASR inference batch size (default: 16)")
    parser.add_argument("--subset_size", type=int, default=12, help="Number of sentences per language (default: 12)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Seeds to evaluate")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "hi"], help="Languages to evaluate")
    parser.add_argument("--index_path", default=DEFAULT_INDEX_PATH, help="Path to reference baseline index")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save re-run manifests")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Path to write final comparison JSON report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    with open(args.index_path, "r", encoding="utf-8") as f:
        ref_index = json.load(f)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "batch_size": args.batch_size,
        "subset_size": args.subset_size,
        "seeds": args.seeds,
        "languages": args.languages,
        "all_metrics_agree": True,
        "comparisons": {},
    }

    t0 = time.time()
    for lang in args.languages:
        out_manifest = os.path.join(args.output_dir, f"repro_base_s{args.subset_size}_{lang}.json")
        ref_file = ref_index["files"].get(lang, {}).get("file")
        if not ref_file:
            raise ValueError(f"Language '{lang}' not found in reference index '{args.index_path}'.")

        index_dir = os.path.dirname(os.path.abspath(args.index_path))
        ref_manifest = ref_file if os.path.isabs(ref_file) else os.path.join(index_dir, ref_file)
        if not os.path.exists(ref_manifest):
            raise FileNotFoundError(f"Reference manifest '{ref_manifest}' does not exist.")

        # Hash reference manifest and verify against base_xtts_s12_index.json
        expected_sha256 = ref_index["files"][lang].get("sha256")
        with open(ref_manifest, "rb") as f:
            computed_sha256 = hashlib.sha256(f.read()).hexdigest()
        if expected_sha256 and computed_sha256 != expected_sha256:
            raise ValueError(
                f"Reference manifest hash mismatch for {lang}: "
                f"computed {computed_sha256} != expected {expected_sha256}"
            )

        print(f"\n=======================================================")
        print(f"🔄 Evaluating {lang.upper()} (batch_size={args.batch_size})...")
        print(f"=======================================================")

        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "scripts", "run_eval_baseline.py"),
            "--checkpoint_dir", str(args.checkpoint_dir),
            "--languages", str(lang),
            "--label", "repro",
            "--seeds", *[str(s) for s in args.seeds],
            "--subset_size", str(args.subset_size),
            "--batch_size", str(args.batch_size),
            "--output", str(out_manifest),
        ]
        run_cmd(cmd)

        # Compare against baseline manifest
        print(f"\n📊 Comparing {lang.upper()} against baseline manifest {ref_file}...")
        from scripts.compare_baselines import compare
        with open(ref_manifest, "r", encoding="utf-8") as f:
            m1 = json.load(f)
        with open(out_manifest, "r", encoding="utf-8") as f:
            m2 = json.load(f)
        comp_data = compare(m1, m2)

        # Gate on provenance: tolerate only intentional code_fingerprint and batch_size differences
        tolerated_mismatches = {"code_fingerprint", "batch_size"}
        unauthorized_mismatches = [k for k in comp_data.get("input_mismatches", []) if k not in tolerated_mismatches]
        if unauthorized_mismatches:
            print(f"  ❌ Unauthorized input mismatches for {lang.upper()}: {unauthorized_mismatches}")
            metrics_agree = False
        else:
            metrics_agree = True

        if comp_data.get("input_mismatches"):
            print(f"  Note: Runs used different metadata inputs: {comp_data['input_mismatches']}")

        for r in comp_data.get("rows", []):
            if "reason" in r:
                print(f"  [FAIL] {r['lang']}/{r['metric']}: {r['reason']}")
                metrics_agree = False
            else:
                agrees = r.get("agrees", False)
                status = "OK  " if agrees else "FAIL"
                print(f"  [{status}] {r['lang']}/{r['metric']}: {r['mean_run1']:.4f} vs {r['mean_run2']:.4f} "
                      f"(|diff| {r['abs_diff']:.4f} <= {r['tolerance']:.4f})")
                if not agrees:
                    metrics_agree = False

        comp_data["all_metrics_agree"] = metrics_agree

        report_lang = os.path.join(args.output_dir, f"compare_{lang}.json")
        with open(report_lang, "w", encoding="utf-8") as f:
            json.dump(comp_data, f, indent=2)
        results["comparisons"][lang] = comp_data
        if not metrics_agree:
            results["all_metrics_agree"] = False
            print(f"❌ {lang.upper()} reproducibility check failed!")
        else:
            print(f"✅ {lang.upper()} metrics reproduced within 95% CI!")

    total_time = time.time() - t0
    results["total_elapsed_seconds"] = round(total_time, 2)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=======================================================")
    print(f"🏁 Reproducibility Check Complete in {total_time / 60:.1f} min")
    print(f"Report saved to: {args.report}")
    print(f"Verdict: {'PASSED (All metrics within 95% CI)' if results['all_metrics_agree'] else 'FAILED'}")
    print(f"=======================================================\n")

    if not results["all_metrics_agree"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
