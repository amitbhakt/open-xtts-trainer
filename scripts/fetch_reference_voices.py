"""Download, build and hash-lock the eval set's reference voice clips (CMU ARCTIC).

Usage:
    python scripts/fetch_reference_voices.py                    # build, verifying against the lock if present
    python scripts/fetch_reference_voices.py --update_lock      # rebuild and rewrite the lock (after review)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.dataset import DEFAULT_EVAL_SET_PATH
from eval.reference_voices import DEFAULT_CACHE_DIR, DEFAULT_LOCK_PATH, DEFAULT_OUT_DIR, build_reference_voices


def main():
    parser = argparse.ArgumentParser(description="Build hash-locked reference voice clips for the eval set.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lock", type=str, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="Where source tarballs are kept")
    parser.add_argument("--eval_set", type=str, default=DEFAULT_EVAL_SET_PATH)
    parser.add_argument("--update_lock", action="store_true", help="Rewrite the lock instead of verifying against it")
    args = parser.parse_args()

    lock = build_reference_voices(
        out_dir=args.out_dir,
        lock_path=args.lock,
        cache_dir=args.cache_dir,
        eval_set_path=args.eval_set,
        update_lock=args.update_lock,
    )
    print(f"{len(lock['voices'])} reference voices ready in {args.out_dir}.")


if __name__ == "__main__":
    main()
