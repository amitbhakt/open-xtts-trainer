"""Phase 8 A/B: does a fine-tuned model beat (or at least match) stock XTTS-v2? Paired, with pre-registered gates.

Two manifests from scripts/run_eval_baseline.py, measured on the same inputs (eval set, reference voices,
seeds, languages, ASR model, metric definitions). Utterances are paired by (language, voice, sentence, seed),
so every comparison is the same sentence in the same voice with the same seed, and sentence difficulty
cancels out. For each metric:
  - the per-pair difference d = tuned - base, its mean, and a 95% bootstrap interval;
  - the same per category and per voice, so a regression hidden in the average still shows.
Headline SECS pairs only utterances where both takes are >= HEADLINE_SECS_MIN_S long (eval/harness.py).

Gates come from a JSON file written *before* the results are seen (eval/gates.json):
  {"metric": "wer", "rule": "not_worse", "margin": 0.02}  -> upper end of the interval of d <= margin
  {"metric": "secs", "rule": "better", "margin": 0.0, "voices": [...]}  -> lower end of the interval of d > margin
Metrics where higher is better (secs, utmos) are oriented automatically: "worse" means lower.
  {"metric": "failures", "rule": "not_worse", "margin": 0}  -> tuned failure count <= base count + margin

Usage:
    python scripts/compare_ab.py base.json tuned.json --gates eval/gates.json [--report ab.json]
Exit code 0 if every gate passes, 1 otherwise.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from eval.harness import HEADLINE_SECS_MIN_S

SAME_INPUT_KEYS = ("metrics_version", "eval_set_sha256", "voices", "reference_voices", "seeds", "subset_size", "languages", "asr_model", "batch_size")
METRICS = ("wer", "cer", "secs", "utmos", "duration_ratio")
HIGHER_IS_BETTER = {"secs", "utmos"}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0

Key = Tuple[str, str, str, int]


def utterances(manifest: Dict[str, Any]) -> Dict[Key, Dict[str, Any]]:
    """(lang, voice, sentence id, seed) -> per-utterance result."""
    out = {}
    for run in manifest["runs"]:
        for seed, results in run["per_seed_results"].items():
            for r in results:
                key = (r["lang"], run["speaker_id"], r["id"], int(seed))
                if key in out:
                    raise ValueError(f"duplicate utterance {key} in one manifest")
                out[key] = {**r, "speaker_id": run["speaker_id"]}
    return out


def bootstrap_ci(diffs: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """95% percentile interval of the mean of paired differences."""
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(samples, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]], metric: str) -> Dict[str, Any]:
    if metric == "secs":  # headline rule: both takes long enough for a reliable speaker embedding
        pairs = [(b, t) for b, t in pairs if min(b["duration_s"], t["duration_s"]) >= HEADLINE_SECS_MIN_S]
    values = [(b[metric], t[metric]) for b, t in pairs if np.isfinite(b[metric]) and np.isfinite(t[metric])]
    if not values:
        return {"n": 0, "base": None, "tuned": None, "diff": None, "ci95": [None, None]}
    base, tuned = np.array(values).T
    lo, hi = bootstrap_ci(tuned - base)
    return {"n": len(values), "base": float(base.mean()), "tuned": float(tuned.mean()),
            "diff": float((tuned - base).mean()), "ci95": [lo, hi]}


def failures(results) -> int:
    return sum(bool(any(r["failures"].values())) for r in results)


def check_gate(gate: Dict[str, Any], pairs) -> Dict[str, Any]:
    voices = gate.get("voices")
    category = gate.get("category")
    selected = [
        p for p in pairs
        if (voices is None or p[0]["speaker_id"] in voices)
        and (category is None or p[0].get("category") == category)
    ]
    if voices and not selected:
        raise ValueError(f"gate {gate} names voices not in the manifests")
    if category and not selected:
        raise ValueError(f"gate {gate} names category {category!r} not in the manifests")
    if gate["metric"] == "failures":
        base, tuned = failures(b for b, _ in selected), failures(t for _, t in selected)
        return {**gate, "base": base, "tuned": tuned, "passed": tuned <= base + gate.get("margin", 0)}
    stats = paired_stats(selected, gate["metric"])
    if stats["n"] < 2:
        return {**gate, **stats, "passed": False, "reason": "fewer than 2 pairs"}
    lo, hi = stats["ci95"]
    sign = 1.0 if gate["metric"] in HIGHER_IS_BETTER else -1.0  # "improvement" direction of d
    worst, best = (lo, hi) if sign > 0 else (-hi, -lo)  # interval of the improvement
    margin = gate.get("margin", 0.0)
    if gate["rule"] == "not_worse":
        passed = worst >= -margin
    elif gate["rule"] == "better":
        passed = worst > margin
    else:
        raise ValueError(f"unknown gate rule {gate['rule']!r}")
    return {**gate, **stats, "improvement_ci95": [worst, best], "passed": bool(passed)}


def load_ablation_gates(ablation_id: str, prereg_path: str = "eval/pre_registered_ablations.json") -> Dict[str, Any]:
    """Load pre-registered decision rules and guardrails for a specific ablation."""
    with open(prereg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ablations = data.get("ablations", {})
    if ablation_id not in ablations:
        valid = ", ".join(sorted(ablations.keys()))
        raise ValueError(f"Unknown ablation ID {ablation_id!r}. Valid IDs: {valid}")
    ablation = ablations[ablation_id]
    gates = list(data.get("global_guardrails", []))
    for g in ablation.get("gates", []):
        gates.append(g)
    return {
        "version": data.get("version", 1),
        "ablation_id": ablation_id,
        "title": ablation.get("title"),
        "decision_rule": ablation.get("decision_rule"),
        "gates": gates,
    }


def _input(manifest: Dict[str, Any], key: str) -> Any:
    # Manifests from before the "voices" field could only measure the eval set's ARCTIC voices
    if key == "voices":
        return manifest.get("voices", "arctic")
    if key == "batch_size":
        return manifest.get("batch_size", 1)
    return manifest.get(key)


def compare_ab(base: Dict[str, Any], tuned: Dict[str, Any], gates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mismatched = [k for k in SAME_INPUT_KEYS if _input(base, k) != _input(tuned, k)]
    if mismatched:
        raise ValueError(f"manifests were measured on different inputs: {mismatched}; an A/B needs identical inputs")
    if base.get("checkpoint_sha256") == tuned.get("checkpoint_sha256"):
        raise ValueError("both manifests measured the same checkpoint")
    b_utts, t_utts = utterances(base), utterances(tuned)
    if b_utts.keys() != t_utts.keys():
        raise ValueError(f"utterance sets differ: {len(b_utts.keys() ^ t_utts.keys())} unpaired")
    keys = sorted(b_utts)
    pairs = [(b_utts[k], t_utts[k]) for k in keys]

    def group(field):
        out: Dict[str, list] = {}
        for b, t in pairs:
            out.setdefault(b.get(field) or "_all", []).append((b, t))
        return out

    by_category, by_voice = group("category"), group("speaker_id")
    report = {
        "base": {"label": base.get("label"), "checkpoint_sha256": base.get("checkpoint_sha256")},
        "tuned": {"label": tuned.get("label"), "checkpoint_sha256": tuned.get("checkpoint_sha256")},
        "pairs": len(pairs),
        "overall": {m: paired_stats(pairs, m) for m in METRICS},
        "failures": {"base": failures(b for b, _ in pairs), "tuned": failures(t for _, t in pairs)},
        "by_category": {c: {m: paired_stats(p, m) for m in METRICS} for c, p in by_category.items()},
        "by_voice": {v: {m: paired_stats(p, m) for m in METRICS} for v, p in by_voice.items()},
    }
    if gates is not None:
        report["gates_version"] = gates.get("version")
        report["gates"] = [check_gate(g, pairs) for g in gates["gates"]]
        report["passed"] = all(g["passed"] for g in report["gates"])
    return report


def main():
    parser = argparse.ArgumentParser(description="Paired A/B of two eval manifests against pre-registered gates.")
    parser.add_argument("base")
    parser.add_argument("tuned")
    parser.add_argument("--gates", default=None, help="Path to gates.json (default: eval/gates.json if --ablation not specified)")
    parser.add_argument("--ablation", default=None, help="Pre-registered ablation ID (e.g. A1..A8) from eval/pre_registered_ablations.json")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base_manifest = json.load(f)
    with open(args.tuned, "r", encoding="utf-8") as f:
        tuned_manifest = json.load(f)

    if args.ablation:
        gates = load_ablation_gates(args.ablation)
    else:
        gates_path = args.gates or "eval/gates.json"
        with open(gates_path, "r", encoding="utf-8") as f:
            gates = json.load(f)

    report = compare_ab(base_manifest, tuned_manifest, gates)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(f"{report['pairs']} paired utterances")
    if args.ablation:
        print(f"Ablation: {args.ablation} - {gates.get('title')}")
        print(f"Decision rule: {gates.get('decision_rule')}")
    for m, s in report["overall"].items():
        if s["n"]:
            print(f"  {m:15s} base {s['base']:.4f}  tuned {s['tuned']:.4f}  diff {s['diff']:+.4f}  "
                  f"95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  (n={s['n']})")
    for g in report["gates"]:
        details = []
        if g.get("voices"):
            details.append(f"voices {g['voices']}")
        if g.get("category"):
            details.append(f"category {g['category']}")
        det_str = f" ({', '.join(details)})" if details else ""
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['metric']} {g['rule']} margin {g.get('margin', 0)}{det_str}")
    print("A/B PASSED" if report["passed"] else "A/B FAILED")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
