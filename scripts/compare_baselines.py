"""Phase 1 exit check: do two baseline runs agree within seed noise?

Two manifests from scripts/run_eval_baseline.py must share every input (eval set, checkpoint and
reference-voice hashes, seeds, languages, ASR model). For each language and metric, the difference in
means must lie within the combined 95% interval: |m1 - m2| <= 1.96 * sqrt(se1^2 + se2^2).

Usage:
    python scripts/compare_baselines.py run1.json run2.json [--report out.json]
"""

import argparse
import json
import math
import sys
from typing import Any, Dict, List

SAME_INPUT_KEYS = ("metrics_version", "code_fingerprint", "eval_set_sha256", "checkpoint_sha256", "reference_voices", "seeds",
                   "subset_size", "languages", "asr_model")
METRICS = ("wer", "cer", "secs", "secs_all", "utmos", "duration_ratio")


def compare(m1: Dict[str, Any], m2: Dict[str, Any]) -> Dict[str, Any]:
    input_mismatches = [k for k in SAME_INPUT_KEYS if m1.get(k) != m2.get(k)]
    rows: List[Dict[str, Any]] = []
    for lang in m1["summary_by_language"]:
        for metric in METRICS:
            a = m1["summary_by_language"][lang][metric]
            b = m2["summary_by_language"].get(lang, {}).get(metric)
            if b is None:
                rows.append({"lang": lang, "metric": metric, "agrees": False, "reason": "missing in run 2"})
                continue
            se_a = a["std"] / math.sqrt(a["n"]) if a["n"] > 1 else 0.0
            se_b = b["std"] / math.sqrt(b["n"]) if b["n"] > 1 else 0.0
            tolerance = 1.96 * math.sqrt(se_a ** 2 + se_b ** 2)
            diff = abs(a["mean"] - b["mean"])
            rows.append({
                "lang": lang,
                "metric": metric,
                "mean_run1": a["mean"],
                "mean_run2": b["mean"],
                "abs_diff": diff,
                "tolerance": tolerance,
                "agrees": bool(diff <= tolerance),
            })
    return {
        "input_mismatches": input_mismatches,
        "rows": rows,
        "reproducible": not input_mismatches and all(r["agrees"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Check that two baseline manifests agree within seed noise.")
    parser.add_argument("run1")
    parser.add_argument("run2")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    with open(args.run1, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(args.run2, "r", encoding="utf-8") as f:
        m2 = json.load(f)
    result = compare(m1, m2)

    if result["input_mismatches"]:
        print(f"Runs used different inputs: {result['input_mismatches']}")
    for r in result["rows"]:
        if "reason" in r:
            print(f"  [FAIL] {r['lang']}/{r['metric']}: {r['reason']}")
        else:
            status = "OK  " if r["agrees"] else "FAIL"
            print(f"  [{status}] {r['lang']}/{r['metric']}: {r['mean_run1']:.4f} vs {r['mean_run2']:.4f} "
                  f"(|diff| {r['abs_diff']:.4f} <= {r['tolerance']:.4f})")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print("REPRODUCIBLE" if result["reproducible"] else "NOT REPRODUCIBLE")
    sys.exit(0 if result["reproducible"] else 1)


if __name__ == "__main__":
    main()
