"""O1 analysis: the locked confirmatory test of F2 and runaway takes, plus A3's own gates on all 900 pairs.

Protocol: docs/evidence/o1_a3_completion/PROTOCOL.md. Every rule here was fixed before any O1 run.

Primary (confirmatory) question, judged only on the 40 sentences A3 never generated: are runaway takes (WER above
compare_ab.RUNAWAY_WER) more common with upstream masking (F2) than with per-sample masking?
  - exact one-sided McNemar test on the paired takes, alpha 0.05;
  - a sentence-cluster bootstrap (resampling the 40 new sentences) of the runaway-rate difference, one-sided 95%
    lower bound, which says whether the effect spreads over sentences or sits on one or two.
Decision: "established" if both agree, "supported, concentrated on few sentences" if only McNemar does,
"not confirmed" otherwise. "Not confirmed" is not evidence that F2 is harmless.

Usage:
    python scripts/analyze_o1.py runs/ablations/o1/manifest_upstream_s60.json \
        runs/ablations/o1/manifest_per_sample_s60.json --report o1_report.json
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.compare_ab import RUNAWAY_WER, compare_ab, load_ablation_gates, utterances

# The 20 sentences A3 already generated and analysed (first 20 of eval.harness.select_subset on the frozen set)
A3_SENTENCE_IDS = frozenset({
    "en_01", "en_02", "en_03", "en_04", "en_11", "en_12", "en_13", "en_14", "en_21", "en_22",
    "en_23", "en_31", "en_32", "en_33", "en_41", "en_42", "en_43", "en_51", "en_52", "en_53",
})
EXPECTED_NEW_SENTENCES = 40
EXPECTED_NEW_PAIRS = 600  # 40 sentences x 5 voices x 3 seeds
ALPHA = 0.05
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
BAD_SHORT_CER = 0.5
SHORT_CATEGORY = "short"

Pair = Tuple[Dict[str, Any], Dict[str, Any]]  # (upstream take, per-sample take)


def is_runaway(take: Dict[str, Any]) -> bool:
    return take["wer"] > RUNAWAY_WER


def paired_takes(upstream: Dict[str, Any], per_sample: Dict[str, Any]) -> Dict[tuple, Pair]:
    u, p = utterances(upstream), utterances(per_sample)
    if u.keys() != p.keys():
        raise ValueError(f"arms are not paired: {len(u.keys() ^ p.keys())} unpaired takes")
    return {k: (u[k], p[k]) for k in sorted(u)}


def new_pairs(pairs: Dict[tuple, Pair]) -> Dict[tuple, Pair]:
    """Only takes of sentences A3 never generated; key = (lang, voice, sentence id, seed)."""
    out = {k: v for k, v in pairs.items() if k[2] not in A3_SENTENCE_IDS}
    sentences = {k[2] for k in out}
    if len(sentences) != EXPECTED_NEW_SENTENCES or len(out) != EXPECTED_NEW_PAIRS:
        raise ValueError(f"expected {EXPECTED_NEW_PAIRS} new pairs over {EXPECTED_NEW_SENTENCES} sentences, "
                         f"got {len(out)} over {len(sentences)}")
    return out


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(at least b of the b + c discordant pairs favour upstream-only events) under no difference."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def discordant(pairs: Dict[tuple, Pair], event) -> Tuple[int, int]:
    """b: event with upstream only; c: event with per-sample only."""
    b = sum(1 for u, p in pairs.values() if event(u) and not event(p))
    c = sum(1 for u, p in pairs.values() if event(p) and not event(u))
    return b, c


def cluster_bootstrap_lower(pairs: Dict[tuple, Pair], event, samples: int = BOOTSTRAP_SAMPLES,
                            seed: int = BOOTSTRAP_SEED) -> float:
    """One-sided 95% lower bound of rate(upstream) - rate(per-sample), resampling whole sentences."""
    sentences = sorted({k[2] for k in pairs})
    diff_sum = {s: 0.0 for s in sentences}
    count = {s: 0 for s in sentences}
    for k, (u, p) in pairs.items():
        diff_sum[k[2]] += float(event(u)) - float(event(p))
        count[k[2]] += 1
    sums = np.array([diff_sum[s] for s in sentences])
    counts = np.array([count[s] for s in sentences], dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sentences), size=(samples, len(sentences)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(np.percentile(means, 100 * ALPHA))


def decide(p_value: float, lower: float) -> str:
    if p_value < ALPHA and lower > 0:
        return "established"
    if p_value < ALPHA:
        return "supported, concentrated on few sentences"
    return "not confirmed"


def event_table(pairs: Dict[tuple, Pair], event, field: int) -> Dict[str, Dict[str, int]]:
    """Event counts per arm, grouped by one key field (1 voice, 2 sentence, 3 seed)."""
    out: Dict[str, Dict[str, int]] = {}
    for k, (u, p) in pairs.items():
        row = out.setdefault(str(k[field]), {"upstream": 0, "per_sample": 0, "pairs": 0})
        row["upstream"] += int(event(u))
        row["per_sample"] += int(event(p))
        row["pairs"] += 1
    return out


def primary(pairs: Dict[tuple, Pair]) -> Dict[str, Any]:
    b, c = discordant(pairs, is_runaway)
    p_value = mcnemar_one_sided(b, c)
    lower = cluster_bootstrap_lower(pairs, is_runaway)
    return {
        "pairs": len(pairs),
        "runaways": {"upstream": sum(is_runaway(u) for u, _ in pairs.values()),
                     "per_sample": sum(is_runaway(p) for _, p in pairs.values())},
        "discordant": {"upstream_only": b, "per_sample_only": c},
        "mcnemar_one_sided_p": p_value,
        "cluster_bootstrap_lower_95": lower,
        "decision": decide(p_value, lower),
        "by_sentence": event_table(pairs, is_runaway, 2),
        "by_seed": event_table(pairs, is_runaway, 3),
        "by_voice": event_table(pairs, is_runaway, 1),
    }


def secondary_short(pairs: Dict[tuple, Pair]) -> Dict[str, Any]:
    short = {k: v for k, v in pairs.items() if v[0].get("category") == SHORT_CATEGORY}

    def bad(t):
        return t["cer"] > BAD_SHORT_CER

    b, c = discordant(short, bad)
    return {"pairs": len(short),
            "bad": {"upstream": sum(bad(u) for u, _ in short.values()),
                    "per_sample": sum(bad(p) for _, p in short.values())},
            "discordant": {"upstream_only": b, "per_sample_only": c},
            "mcnemar_one_sided_p": mcnemar_one_sided(b, c)}


def reproduction(pairs: Dict[tuple, Pair], a3_upstream: Dict[str, Any], a3_per_sample: Dict[str, Any]) -> Dict[str, Any]:
    """Exploratory: do the 20 sentences A3 already scored come out the same under today's code?"""
    out = {}
    for arm, a3, idx in (("upstream", a3_upstream, 0), ("per_sample", a3_per_sample, 1)):
        old = utterances(a3)
        keys = [k for k in old if k in pairs]
        same_text = sum(old[k]["asr_text"] == pairs[k][idx]["asr_text"] for k in keys)
        out[arm] = {"takes": len(keys), "identical_transcripts": same_text,
                    "old_runaways": sum(is_runaway(old[k]) for k in keys),
                    "new_runaways": sum(is_runaway(pairs[k][idx]) for k in keys)}
    return out


def analyze(upstream: Dict[str, Any], per_sample: Dict[str, Any], a3_upstream=None, a3_per_sample=None) -> Dict[str, Any]:
    pairs = paired_takes(upstream, per_sample)
    fresh = new_pairs(pairs)
    report = {
        "protocol": "docs/evidence/o1_a3_completion/PROTOCOL.md",
        "primary_new_sentences": primary(fresh),
        "secondary_short_new_sentences": secondary_short(fresh),
        "a3_gates_all_pairs": compare_ab(upstream, per_sample, load_ablation_gates("A3")),
    }
    if a3_upstream is not None and a3_per_sample is not None:
        report["exploratory_reproduction_of_a3"] = reproduction(pairs, a3_upstream, a3_per_sample)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("upstream")
    parser.add_argument("per_sample")
    parser.add_argument("--a3_upstream", default="docs/evidence/ablation_a3/manifest_upstream_s20_base.json")
    parser.add_argument("--a3_per_sample", default="docs/evidence/ablation_a3/manifest_per_sample_s20_base.json")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    def load(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    report = analyze(load(args.upstream), load(args.per_sample), load(args.a3_upstream), load(args.a3_per_sample))
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    prim = report["primary_new_sentences"]
    print(f"Primary: {prim['decision']} (McNemar p = {prim['mcnemar_one_sided_p']:.4f}, "
          f"cluster lower bound = {prim['cluster_bootstrap_lower_95']:+.4f}, runaways {prim['runaways']})")
    print(f"A3 gates on all pairs: passed = {report['a3_gates_all_pairs']['passed']}")


if __name__ == "__main__":
    main()
