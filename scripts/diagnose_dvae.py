"""DVAE diagnostic: how well does a DVAE fit your audio? (docs/recipe_design.md §3, Stage 1 decision.)

The recipe keeps the released DVAE unless it fits the target audio clearly worse than reference speech it
handles well; a trained DVAE forces a full Stage 2 retrain (F1). This measures, with the DVAE in eval mode
(the same code as Stage 1 validation, stage1_dvae.train.evaluate_dvae):
- reconstruction error of whole clips decoded from their codes,
- active codes and perplexity of the code distribution,
- with --compare, code drift: the share of frames whose code differs between the two DVAEs.
With --reference_dataset, the same numbers on reference speech, and the ratios between them. The thresholds
for "clearly worse" are [E5] until ablation A2 calibrates them, so this reports and doesn't decide.

    python scripts/diagnose_dvae.py --dataset my_data/dataset.jsonl --checkpoint xtts_base/dvae.pth \\
        [--reference_dataset en_ref/dataset.jsonl] [--compare checkpoints/dvae/dvae.pth] [--report out.json]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from xtts_trainer.data import load_validated_dataset
from xtts_trainer.models import load_dvae
from xtts_trainer.stage1_dvae.dataset import MelDataset
from xtts_trainer.stage1_dvae.train import evaluate_dvae

MAX_FRAMES = 2048  # per clip, as Stage 1 validation


def dataset_mels(dataset_path: str, device: str):
    records, dataset_dir, _ = load_validated_dataset(dataset_path, None, require_text=False)
    mels = MelDataset([r.audio_path(dataset_dir) for r in records], MAX_FRAMES, random_crop=False)
    return [m.to(device) for m in mels]


def diagnose(dataset_path: str, checkpoint: str, reference_dataset: Optional[str] = None,
             compare: Optional[str] = None, device: str = "cpu") -> Dict[str, Any]:
    dvae = load_dvae(checkpoint, device=device)
    other = load_dvae(compare, device=device) if compare else None
    report: Dict[str, Any] = {"checkpoint": checkpoint, "dataset": dataset_path,
                              "target": evaluate_dvae(dvae, dataset_mels(dataset_path, device), other)}
    if compare:
        report["compare"] = compare
    if reference_dataset:
        ref = evaluate_dvae(dvae, dataset_mels(reference_dataset, device), other)
        report["reference_dataset"] = reference_dataset
        report["reference"] = ref
        report["target_vs_reference"] = {
            "recon_ratio": report["target"]["val_recon"] / ref["val_recon"],
            "perplexity_ratio": report["target"]["perplexity"] / ref["perplexity"],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="How well does a DVAE fit this audio? (reports, doesn't decide)")
    parser.add_argument("--dataset", required=True, help="dataset.jsonl of the target audio")
    parser.add_argument("--checkpoint", default="xtts_base/dvae.pth")
    parser.add_argument("--reference_dataset", default=None, help="dataset.jsonl of reference speech")
    parser.add_argument("--compare", default=None, help="A second DVAE, e.g. a Stage 1 result, for code drift")
    parser.add_argument("--report", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = diagnose(args.dataset, args.checkpoint, args.reference_dataset, args.compare, args.device)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
