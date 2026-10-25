"""Run the complete Run 5 pipeline headlessly.

Executes all verification, testing, sample preparation, Stage 2 smoke training,
Stage 3 smoke training, multilingual baseline evaluation, and evidence packaging.
Can be executed locally or inside a Modal container on an A100 GPU.

Usage:
    python scripts/run_run5_pipeline.py [--results-dir runs/results_run5] [--subset-size 12]
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"


def run_cmd(cmd: str, check: bool = True) -> int:
    """Run a shell command with pipefail and live output."""
    print(f"\n▶ [{time.strftime('%H:%M:%S')}] {cmd}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=check, env=env)
    return proc.returncode


def check_artifacts(after: str):
    """Confirm every XTTS artifact in xtts_base/ still matches artifacts.lock.json."""
    if not os.path.exists("artifacts.lock.json"):
        print("artifacts.lock.json not found, skipping check_artifacts")
        return
    artifacts = json.load(open("artifacts.lock.json"))
    for name, entry in artifacts.items():
        path = os.path.join("xtts_base", name)
        assert os.path.exists(path), f"{path} missing during/after {after}"
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        assert digest.hexdigest() == entry["sha256"], (
            f"{path} changed during {after}: expected {entry['sha256']}, got {digest.hexdigest()}"
        )
    print(f"✅ Base artifacts intact after {after}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Run 5 pipeline headlessly.")
    parser.add_argument("--results-dir", default="runs/results_run5", help="Path to store evidence and manifests")
    parser.add_argument("--subset-size", type=int, default=12, help="Eval subset size per language (0 for full)")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "hi"], help="Eval languages")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999], help="Eval seeds")
    parser.add_argument("--skip-tests", action="store_true", help="Skip full test suite (for fast dev)")
    args = parser.parse_args()

    t_start = time.time()
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("docs/evidence", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("eval/baselines", exist_ok=True)

    print("=" * 70)
    print("🚀 Starting Open-XTTS-Trainer Run 5 Pipeline")
    print(f"Results Directory: {results_dir}")
    print(f"Subset Size: {args.subset_size}")
    print(f"Languages: {args.languages}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    # Record environment evidence
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    import torch
    import torchaudio
    import transformers
    from eval.harness import get_code_fingerprint

    env = {
        "git": {"commit": commit},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "code_fingerprint": get_code_fingerprint(),
    }
    json.dump(env, open("docs/evidence/environment.json", "w"), indent=2)
    print(f"Environment: GPU={env['gpu']}, CUDA={env['cuda']}, Torch={env['torch']}")

    # Phase 0 / Step 4: Fetch and verify base XTTS artifacts
    print("\n📦 Step 4: Fetching and verifying base XTTS artifacts...")
    os.makedirs("docs/evidence/phase0", exist_ok=True)
    run_cmd(
        "python scripts/fetch_base_artifacts.py --dest xtts_base --check_mel_stats "
        "--report docs/evidence/phase0/artifact_verification.json"
    )
    check_artifacts("step 4")

    # Step 5: Test suite
    if not args.skip_tests:
        print("\n🧪 Step 5: Running unit tests...")
        run_cmd("python -m pytest -q -p no:cacheprovider")
        check_artifacts("step 5")
    else:
        print("\n⏩ Skipping unit tests (--skip-tests)")

    # Step 6: Fetch reference voices
    print("\n🎙️ Step 6: Fetching reference voices...")
    run_cmd("python scripts/fetch_reference_voices.py")

    # Step 6b: Model gated tests
    if not args.skip_tests:
        print("\n🧪 Step 6b: Running model gated invariant tests...")
        os.makedirs("docs/evidence/phase5", exist_ok=True)
        run_cmd(
            'python -m pytest -v -rs -p no:cacheprovider tests/test_invariants.py tests/test_spec_parity.py '
            'tests/test_data_contract.py -k "round_trip or upstream_or_local"'
        )
        check_artifacts("step 6b")

    # Step 7a & 7b: Sample audio data preparation & validation
    print("\n🔊 Step 7a & 7b: Preparing and validating sample dataset...")
    run_cmd('python -c "import torchaudio; torchaudio.pipelines.MMS_FA.get_model()"')
    run_cmd("python -m pytest -v -p no:cacheprovider tests/test_prepare.py -k mms")
    run_cmd("rm -rf sample_audio && python -m xtts_trainer.speech_data --source sample --output_dir sample_audio --vocab xtts_base/vocab.json")
    run_cmd(
        "python -m xtts_trainer.data.validate sample_audio/dataset.jsonl --vocab xtts_base/vocab.json "
        "--realign --report sample_audio/realign_report.json"
    )
    os.makedirs("docs/evidence/phase6", exist_ok=True)
    from xtts_trainer.speech_data import SOURCE_SETS, SOURCES
    pins = {name: SOURCES[name].sha256 for name in SOURCE_SETS["sample"]}
    json.dump(pins, open("docs/evidence/phase6/source_pins_used.json", "w"), indent=2)
    run_cmd(
        "cp sample_audio/prep_report.json sample_audio/validation_report.json sample_audio/realign_report.json "
        "sample_audio/dataset.jsonl docs/evidence/phase6/"
    )

    # Step 7c: Stage 2 GPT smoke train & export & inference
    print("\n🧠 Step 7c: Stage 2 GPT smoke training & inference check...")
    os.makedirs("docs/evidence/phase7", exist_ok=True)
    run_cmd(
        "python -m xtts_trainer.stage2_gpt.train --dataset sample_audio/dataset.jsonl --base_dir xtts_base "
        "--output_dir runs/gpt_smoke --epochs 3 --batch_size 2 --grad_accum_steps 1 --eval_fraction 0.25"
    )
    from xtts_trainer.stage2_gpt.export import load_for_inference
    from xtts_trainer.spec import SPEC
    import soundfile as sf
    xtts = load_for_inference("runs/gpt_smoke/export", "cuda" if torch.cuda.is_available() else "cpu")
    lat, spk = xtts.get_conditioning_latents(audio_path=["eval/reference_voices/arctic_slt.wav"])
    torch.manual_seed(42)
    wav = xtts.inference("The quick brown fox jumps over the lazy dog.", "en", lat, spk, temperature=0.7)["wav"]
    sf.write("docs/evidence/phase7/stage2_smoke_sample.wav", wav, SPEC.output_sample_rate)
    run_cmd("cp runs/gpt_smoke/export/training_summary.json docs/evidence/phase7/")
    check_artifacts("step 7c")

    # Step 7d: Stage 3 HiFiGAN smoke train
    print("\n🎵 Step 7d: Stage 3 HiFiGAN smoke training...")
    import gc
    del xtts, lat, spk, wav
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from xtts_trainer.config import HiFiGANConfig
    from xtts_trainer.stage3_hifigan.extract_latents import extract_latents
    from xtts_trainer.stage3_hifigan.train import train_hifigan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_latents("sample_audio/dataset.jsonl", "runs/latents", checkpoint_dir="runs/gpt_smoke/export", device=device)
    cfg3 = HiFiGANConfig(batch_size=4, epochs=15, disc_warmup_steps=10, val_every_steps=2, sample_every_steps=10)
    best3 = train_hifigan(
        "sample_audio/dataset.jsonl", "runs/latents", "runs/hifigan_smoke", cfg3,
        checkpoint_dir="runs/gpt_smoke/export", device=device
    )
    os.makedirs("docs/evidence/phase7/stage3", exist_ok=True)
    run_cmd("cp runs/hifigan_smoke/metrics.jsonl docs/evidence/phase7/stage3/ && cp -r runs/hifigan_smoke/samples docs/evidence/phase7/stage3/")
    check_artifacts("step 7d")

    # Step 8: Multilingual baseline evaluation
    print(f"\n📊 Step 8: Multilingual baseline evaluation across {args.languages}...")
    from eval.provenance import expected_baseline_inputs, index_manifest_mismatches, manifest_mismatches, write_baseline_index

    tag = f"s{args.subset_size}" if args.subset_size else "full"
    index_path = f"eval/baselines/base_xtts_{tag}_index.json"
    saved_index_path = f"{results_dir}/base_xtts_{tag}_index.json"
    existing_index = json.load(open(saved_index_path)) if os.path.exists(saved_index_path) else None

    expected = expected_baseline_inputs(commit, seeds=args.seeds, subset_size=args.subset_size)
    paths = {}
    for lang in args.languages:
        name = f"base_xtts_{tag}_{lang}.json"
        local_path = f"eval/baselines/{name}"
        saved_path = f"{results_dir}/{name}"

        if os.path.exists(saved_path) and existing_index:
            problems = manifest_mismatches(json.load(open(saved_path)), expected, lang)
            index_problems = index_manifest_mismatches(saved_path, existing_index, lang)
            if not problems and not index_problems:
                run_cmd(f"cp {saved_path} {local_path}")
                print(f"✅ {lang}: Reused from saved results (hash verified)")
                paths[lang] = local_path
                continue

        t0_lang = time.time()
        seeds_arg = " ".join(map(str, args.seeds))
        subset_arg = f"--subset_size {args.subset_size}" if args.subset_size else ""
        run_cmd(
            f"python scripts/run_eval_baseline.py --checkpoint_dir xtts_base --languages {lang} "
            f"--label base --seeds {seeds_arg} {subset_arg} --output {local_path}"
        )
        problems = manifest_mismatches(json.load(open(local_path)), expected, lang)
        assert not problems, f"{local_path} does not match expected inputs:\n" + "\n".join(problems)
        run_cmd(f"cp {local_path} {saved_path}")
        paths[lang] = local_path
        print(f"⏱️ {lang} evaluation took {(time.time() - t0_lang) / 60:.1f} min")

        existing_index = write_baseline_index(paths, index_path, existing_index=existing_index)
        run_cmd(f"cp {index_path} {results_dir}/")

    check_artifacts("step 8")

    # Step 9: Package all evidence
    print("\n📦 Step 9: Packaging run evidence...")
    evidence_zip = "runs/run5_evidence.zip"
    run_cmd(
        f"zip -qr {evidence_zip} docs/evidence eval/baselines eval/dataset/reference_voices.lock.json "
        f"eval/dataset/eval_set.json eval/reference_voices"
    )
    run_cmd(f"cp {evidence_zip} {results_dir}/")

    elapsed_min = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print(f"🎉 Run 5 Pipeline Completed Successfully in {elapsed_min:.1f} minutes!")
    print(f"Evidence archived: {evidence_zip}")
    print(f"Results copied to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
