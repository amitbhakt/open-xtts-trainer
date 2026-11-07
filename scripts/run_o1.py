"""O1: finish ablation A3 by re-scoring its existing checkpoints on all 60 English sentences (no training).

Protocol: docs/evidence/o1_a3_completion/PROTOCOL.md (locked before any O1 run).

A3 trained one Stage 2 model per masking arm and training seed (6 models, on the runs volume) and scored each on the
first 20 sentences of the frozen eval set. O1 scores the same 6 models on all 60 sentences, with the generation seed
equal to the training seed, exactly as A3 did, so each seed gets the 300 paired takes A3's pre-registration asked for.
Both arms run in one container (one eval subprocess per arm and seed): M1 found that separate containers can give
slightly different results for the same model (M1 PROTOCOL.md §10.3, deviation 2), and that difference must not
fall on one arm only. The arms alternate seed by seed, so a run that stops early leaves complete pairs for every seed
it finished. It never resumes: a stopped run is repeated in full in a new --output_dir.

Usage (one container, both arms):
    python scripts/run_o1.py
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

A3_DIR = "runs/ablations/a3"
OUTPUT_DIR = "runs/ablations/o1"
ARMS = ("upstream", "per_sample")
SEEDS = (1, 2, 3)
SUBSET_SIZE = 60
EVAL_BATCH_SIZE = 16
# SHA-256 of each A3 model.pth, from A3's own manifests (docs/evidence/ablation_a3/per_seed_manifests/)
EXPECTED_MODEL_SHA256: Dict[Tuple[str, int], str] = {
    ("upstream", 1): "4a4666bbb82924d5098c42af19da9116187f0413f4da04c98c0e78567257522b",
    ("upstream", 2): "b60dd6835cd583d42c40853ee11a67c677a8ec92f116fe17432bfe9e8c58056b",
    ("upstream", 3): "979037679abaf3067ac6b63ccded6ebfff9637733a72da6e8e85e40c1e96b3f4",
    ("per_sample", 1): "8c26801f9e4fa28af32b9f2e095826eb528e70253d961fc878c25e95e842a48a",
    ("per_sample", 2): "198621a06fa0c1dcc5bdd4b8a4158c67dc25b5b550bf569f469baae9f75b73bb",
    ("per_sample", 3): "71769b2d2e34d46bf20d8f0f0e0c00ea6093cfbb151b03f6c80c93feb2bc92e6",
}


def checkpoint_dir(arm: str, seed: int, a3_dir: str = A3_DIR) -> str:
    return os.path.join(a3_dir, arm, f"seed_{seed}", "export")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_checkpoints(arm: str, a3_dir: str = A3_DIR) -> None:
    """Refuse to run unless every model.pth of this arm is byte-identical to the one A3 scored."""
    for seed in SEEDS:
        path = os.path.join(checkpoint_dir(arm, seed, a3_dir), "model.pth")
        if not os.path.isfile(path):
            raise SystemExit(f"missing A3 checkpoint {path}")
        got = sha256_file(path)
        if got != EXPECTED_MODEL_SHA256[(arm, seed)]:
            raise SystemExit(f"{path} has SHA-256 {got}, not the {EXPECTED_MODEL_SHA256[(arm, seed)]} A3 scored")


def eval_command(arm: str, seed: int, output: str, a3_dir: str = A3_DIR) -> list:
    return [sys.executable, "scripts/run_eval_baseline.py",
            "--checkpoint_dir", checkpoint_dir(arm, seed, a3_dir),
            "--voices", "arctic", "--languages", "en",
            "--label", f"o1_{arm}_s{seed}",
            "--seeds", str(seed),
            "--subset_size", str(SUBSET_SIZE),
            "--batch_size", str(EVAL_BATCH_SIZE),
            "--output", output]


def per_seed_path(output_dir: str, arm: str, seed: int) -> str:
    return os.path.join(output_dir, "per_seed", f"manifest_{arm}_seed_{seed}_s{SUBSET_SIZE}.json")


def merged_path(output_dir: str, arm: str) -> str:
    return os.path.join(output_dir, f"manifest_{arm}_s{SUBSET_SIZE}.json")


def environment_path(output_dir: str) -> str:
    return os.path.join(output_dir, "environment.json")


def run_plan() -> list:
    """(arm, seed) in run order: both arms for seed 1, then seed 2, then seed 3."""
    return [(arm, seed) for seed in SEEDS for arm in ARMS]


def environment() -> dict:
    import torch
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "python": platform.python_version()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    existing = [p for p in [environment_path(args.output_dir)] + [merged_path(args.output_dir, a) for a in ARMS]
                + [per_seed_path(args.output_dir, a, s) for a, s in run_plan()] if os.path.exists(p)]
    if existing:
        raise SystemExit(f"{existing} exist; O1 never overwrites a result (record a rerun as a new attempt)")
    for arm in ARMS:
        verify_checkpoints(arm)
    os.makedirs(os.path.join(args.output_dir, "per_seed"), exist_ok=True)
    with open(environment_path(args.output_dir), "w", encoding="utf-8") as f:
        json.dump(environment(), f, indent=2)

    env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"}
    manifests = {arm: [] for arm in ARMS}
    for arm, seed in run_plan():
        out = per_seed_path(args.output_dir, arm, seed)
        subprocess.run(eval_command(arm, seed, out), env=env, cwd=REPO_ROOT, check=True)
        with open(out, encoding="utf-8") as f:
            manifests[arm].append(json.load(f))

    from scripts.run_ablation_a3 import merge_manifests
    for arm in ARMS:
        merge_manifests(manifests[arm], f"o1_{arm}", merged_path(args.output_dir, arm))
        print(f"O1 {arm}: wrote {merged_path(args.output_dir, arm)}")


if __name__ == "__main__":
    main()
