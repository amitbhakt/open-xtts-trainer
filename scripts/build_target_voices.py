"""Hold out target-speaker reference clips for evaluation, and write the training dataset without them.

    python scripts/build_target_voices.py --dataset my_data/dataset.jsonl [--speakers me]

Then train on my_data/dataset_train.jsonl, and evaluate both models with --voices target (or all):
    python scripts/run_eval_baseline.py --checkpoint_dir xtts_base --voices target --output eval/baselines/base.json
See eval/target_voices.py.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.target_voices import DEFAULT_TARGET_DIR, LOCK_FILENAME, build_target_voices


def main():
    parser = argparse.ArgumentParser(description="Hold out target-speaker reference clips for evaluation.")
    parser.add_argument("--dataset", required=True, help="dataset.jsonl of the speaker(s) you will fine-tune on")
    parser.add_argument("--speakers", nargs="+", default=None, help="Speakers to hold out (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # Always the canonical location: the only one training checks for held-out clips
    lock = build_target_voices(args.dataset, DEFAULT_TARGET_DIR, args.speakers, args.seed)
    for vid, entry in lock["voices"].items():
        print(f"{vid}: {entry['duration_s']} s from {len(entry['source_clips'])} held-out clips")
    print(f"Lock: {os.path.join(DEFAULT_TARGET_DIR, LOCK_FILENAME)}")
    print(f"Train on: {lock['train_dataset']}  (training refuses the held-out clips while this lock exists)")
    print(json.dumps({"gate_voices": sorted(lock["voices"])}))


if __name__ == "__main__":
    main()
