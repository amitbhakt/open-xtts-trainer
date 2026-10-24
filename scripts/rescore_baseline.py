"""Re-score a baseline manifest under the current metric definitions, without re-synthesizing anything.

Only what is derived from stored per-utterance values changes: duration ratio, the too-long flag, categories
and every summary. Synthesis, ASR text, WER/CER, SECS and UTMOS values are kept exactly as measured.

Usage:
    python scripts/rescore_baseline.py eval/baselines/pilot_en.json --output pilot_en_v2.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.harness import METRICS_VERSION, summarize_by_category, summarize_results
from eval.metrics import compute_duration_ratio, is_too_long
from scripts.run_eval_baseline import summaries_from_results


def rescore(manifest: Dict[str, Any], eval_set: Dict[str, Any]) -> Dict[str, Any]:
    if manifest.get("eval_set_sha256") != eval_set.get("sha256"):
        raise ValueError("manifest was measured on a different eval set; categories can't be recovered")
    sentences = list(eval_set["english_sentences"]) + [s for ss in eval_set["target_languages"].values() for s in ss]
    by_text = {(s["text"]): s for s in sentences}
    results_by_language: Dict[str, List[Dict[str, Any]]] = {}
    for run in manifest["runs"]:
        run_results = []
        for results in run["per_seed_results"].values():
            for r in results:
                item = by_text[r["text"]]
                r["id"], r["category"] = item["id"], item.get("category")
                r["duration_ratio"] = compute_duration_ratio(r["duration_s"], r["text"], lang=r["lang"])
                r["failures"]["is_too_long"] = is_too_long(r["duration_s"], r["text"], lang=r["lang"])
                run_results.append(r)
                results_by_language.setdefault(r["lang"], []).append(r)
        kept = {k: run["summary"][k] for k in ("num_sentences", "num_seeds", "total_evaluations")}
        run["summary"] = {**kept, **summarize_results(run_results), "by_category": summarize_by_category(run_results)}
        run["metrics_version"] = METRICS_VERSION
    manifest.update(summaries_from_results(results_by_language))
    manifest["rescored_from_metrics_version"] = manifest.get("metrics_version", 1)
    manifest["metrics_version"] = METRICS_VERSION
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Re-score a baseline manifest under the current metric definitions.")
    parser.add_argument("manifest")
    parser.add_argument("--eval_set", default="eval/dataset/eval_set.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.eval_set, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    result = rescore(manifest, eval_set)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary_by_language"], indent=1))


if __name__ == "__main__":
    main()
