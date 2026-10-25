"""Run open-xtts-trainer evaluations or smoke runs headlessly on Modal.

Allows executing any command on an A10G / L4 / A100 GPU in Modal without
keeping a local Jupyter session open. Results are synced to a persistent volume.

Usage:
    # Run the test suite:
    uvx modal run scripts/modal_run_eval.py --cmd "pytest -q"

    # Fetch base artifacts:
    uvx modal run scripts/modal_run_eval.py --cmd "python scripts/fetch_base_artifacts.py --dest xtts_base --check_mel_stats"

    # Run English baseline on A10G:
    uvx modal run scripts/modal_run_eval.py --gpu A10G --cmd "python scripts/run_eval_baseline.py --checkpoint_dir xtts_base --languages en --label base --seeds 42 123 999 --subset_size 12 --output eval/baselines/base_xtts_s12_en.json"
"""

import subprocess
import modal

app = modal.App.lookup("open-xtts-runner", create_if_missing=True)

volume = modal.Volume.from_name("open-xtts-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv==0.12.13")
    .run_commands(
        "git clone https://github.com/amitbhakt/open-xtts-trainer.git /root/open-xtts-trainer",
        "cd /root/open-xtts-trainer && git checkout colab-run5",
        "cd /root/open-xtts-trainer && uv export --frozen --no-hashes --no-emit-project --extra dev --extra eval -o /root/reqs.txt",
        "uv pip install --system --break-system-packages -r /root/reqs.txt",
        "uv pip install --system --break-system-packages -e /root/open-xtts-trainer",
        "pip uninstall -y -q torchvision",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/open-xtts-trainer/xtts_base": volume},
    timeout=7200,
)
def run_command_remote(cmd: str):
    """Execute a bash command inside the Modal GPU container."""
    print(f"🔧 Running in Modal container: {cmd}")
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd="/root/open-xtts-trainer",
        check=True,
    )
    volume.commit()
    return proc.returncode


@app.local_entrypoint()
def main(cmd: str = "pytest -q", gpu: str = "A10G"):
    print(f"Submitting job to Modal ({gpu})...")
    # Dynamically select requested GPU
    fn = run_command_remote.with_options(gpu=gpu)
    fn.remote(cmd)
    print("✅ Completed successfully on Modal!")
