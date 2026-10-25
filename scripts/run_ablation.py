"""Ablation experiment runner and hypothesis evaluator for Open-XTTS-Trainer.

Implements REMEDIATION_PLAN.md §12.3.C Item 8 (Ablations runner):
- Discovers and lists pre-registered ablations (A1..A8) from eval/pre_registered_ablations.json.
- Discovers or verifies baseline manifests matching requested ablation parameters.
- Consumes pre-computed candidate manifests and compares them to a base manifest via scripts/compare_ab.py.
- Evaluates hypotheses headlessly via scripts/compare_ab.py against pre-registered decision rules.

Usage:
    # List all pre-registered ablations
    python scripts/run_ablation.py --list

    # View details and pre-registered decision rules for A3
    python scripts/run_ablation.py --info A3

    # Run paired evaluation for candidate manifest against A3 pre-registered rules
    python scripts/run_ablation.py --ablation A3 --candidate_manifest runs/candidates/a3_tuned.json

    # Dry-run validation of inputs and gates
    python scripts/run_ablation.py --ablation A3 --candidate_manifest runs/candidates/a3_tuned.json --dry_run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.compare_ab import compare_ab, load_ablation_gates, utterances, validate_manifest_pair


PREREG_PATH = os.path.join(REPO_ROOT, "eval", "pre_registered_ablations.json")


def load_preregistered_ablations(path: str = PREREG_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pre-registration file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_ablations(prereg_data: Dict[str, Any]) -> None:
    ablations = prereg_data.get("ablations", {})
    print(f"Pre-Registered Ablation Studies ({len(ablations)} total):")
    print(f"{'ID':<6} {'Title':<45} {'Status':<16} {'Citations'}")
    print("-" * 85)
    for aid, info in sorted(ablations.items()):
        cites = ", ".join(info.get("citations", []))[:30]
        print(f"{aid:<6} {info.get('title', ''):<45} {info.get('status', 'unknown'):<16} {cites}")


def print_ablation_info(prereg_data: Dict[str, Any], aid: str) -> None:
    ablations = prereg_data.get("ablations", {})
    if aid not in ablations:
        valid = ", ".join(sorted(ablations.keys()))
        raise ValueError(f"Unknown ablation ID {aid!r}. Valid IDs: {valid}")
    info = ablations[aid]
    print(f"\n=== Ablation {aid}: {info.get('title')} ===")
    print(f"Status:        {info.get('status')}")
    print(f"Question:      {info.get('question')}")
    print("\nSetup Matrix:")
    print(json.dumps(info.get("setup", {}), indent=2))
    print(f"\nPrimary Metrics: {', '.join(info.get('primary_metrics', []))}")
    print(f"\nPre-Registered Decision Rule:\n  {info.get('decision_rule')}")
    print("\nCitations:")
    for c in info.get("citations", []):
        print(f"  • {c}")
    print()


def find_default_baseline(subset_size: int = 12, language: str = "en") -> Optional[str]:
    """Find matching baseline in eval/baselines/ for given subset size and language."""
    filename = f"base_xtts_s{subset_size}_{language}.json"
    candidate = os.path.join(REPO_ROOT, "eval", "baselines", filename)
    if os.path.exists(candidate):
        return candidate
    return None


def run_ablation(
    ablation_id: str,
    candidate_manifest_path: str,
    base_manifest_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    report_path: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    prereg = load_preregistered_ablations()
    ablations = prereg.get("ablations", {})
    if ablation_id not in ablations:
        valid = ", ".join(sorted(ablations.keys()))
        raise ValueError(f"Unknown ablation ID {ablation_id!r}. Valid IDs: {valid}")

    ablation_info = ablations[ablation_id]
    output_dir = output_dir or os.path.join(REPO_ROOT, "runs", "ablations", ablation_id.lower())
    os.makedirs(output_dir, exist_ok=True)
    report_path = report_path or os.path.join(output_dir, "ab_report.json")

    # Resolve candidate manifest
    if not os.path.exists(candidate_manifest_path):
        raise FileNotFoundError(f"Candidate manifest not found at {candidate_manifest_path}")
    with open(candidate_manifest_path, "r", encoding="utf-8") as f:
        candidate_data = json.load(f)

    # Resolve baseline manifest
    if not base_manifest_path:
        subset_size = candidate_data.get("subset_size", 12)
        languages = candidate_data.get("languages", ["en"])
        language = languages[0] if languages else "en"
        base_manifest_path = find_default_baseline(subset_size, language)
        if not base_manifest_path:
            raise FileNotFoundError(
                f"No default baseline manifest found for subset size {subset_size} and language {language!r}. "
                "Specify --base_manifest explicitly."
            )

    if not os.path.exists(base_manifest_path):
        raise FileNotFoundError(f"Base manifest not found at {base_manifest_path}")
    with open(base_manifest_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    gates = load_ablation_gates(ablation_id, PREREG_PATH)

    print(f"Ablation: {ablation_id} - {ablation_info.get('title')}")
    print(f"Base manifest:      {base_manifest_path}")
    print(f"Candidate manifest: {candidate_manifest_path}")
    print(f"Decision rule:      {ablation_info.get('decision_rule')}")

    # Validate manifest compatibility, schema, and utterance pairing
    validate_manifest_pair(base_data, candidate_data)

    # Validate gate voice/category requirements against manifest utterances
    b_utts = utterances(base_data)
    available_voices = {u["speaker_id"] for u in b_utts.values()}
    available_categories = {u.get("category") for u in b_utts.values() if u.get("category")}
    for g in gates.get("gates", []):
        if g.get("voices") and not any(v in available_voices for v in g["voices"]):
            raise ValueError(f"Gate names voices {g['voices']} not found in manifests")
        if g.get("category") and g["category"] not in available_categories:
            raise ValueError(f"Gate names category {g['category']!r} not found in manifests")

    if dry_run:
        print("\n[DRY RUN] Inputs and pre-registered gates verified successfully. Skipping paired computation.")
        return {
            "ablation_id": ablation_id,
            "dry_run": True,
            "base_manifest": base_manifest_path,
            "candidate_manifest": candidate_manifest_path,
            "passed": True,
        }

    report = compare_ab(base_data, candidate_data, gates)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_path}")
    print(f"Evaluated {report['pairs']} paired utterances.")
    for g in report["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  [{status}] {g['metric']} {g['rule']} margin {g.get('margin', 0)}")

    result_str = "PASSED" if report["passed"] else "FAILED"
    print(f"\nAblation {ablation_id} {result_str} against pre-registered gates.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run and evaluate pre-registered ablation studies.")
    parser.add_argument("--list", action="store_true", help="List all pre-registered ablations")
    parser.add_argument("--info", metavar="ID", help="Show details and decision rules for an ablation (e.g. A1)")
    parser.add_argument("--ablation", metavar="ID", help="Ablation ID to evaluate (e.g. A1..A8)")
    parser.add_argument("--base_manifest", help="Path to base manifest (default: auto-detected from eval/baselines/)")
    parser.add_argument("--candidate_manifest", help="Path to evaluated candidate manifest")
    parser.add_argument("--output_dir", help="Directory to store ablation outputs")
    parser.add_argument("--report", help="Path to write evaluation report JSON")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and gates without running comparison")
    args = parser.parse_args()

    prereg = load_preregistered_ablations()

    if args.list:
        list_ablations(prereg)
        sys.exit(0)

    if args.info:
        print_ablation_info(prereg, args.info.upper())
        sys.exit(0)

    if not args.ablation:
        parser.print_help()
        sys.exit(1)

    ablation_id = args.ablation.upper()
    if not args.candidate_manifest:
        print(f"Error: --candidate_manifest is required when running ablation {ablation_id}.", file=sys.stderr)
        sys.exit(1)

    report = run_ablation(
        ablation_id=ablation_id,
        candidate_manifest_path=args.candidate_manifest,
        base_manifest_path=args.base_manifest,
        output_dir=args.output_dir,
        report_path=args.report,
        dry_run=args.dry_run,
    )
    sys.exit(0 if report.get("passed", False) else 1)


if __name__ == "__main__":
    main()
