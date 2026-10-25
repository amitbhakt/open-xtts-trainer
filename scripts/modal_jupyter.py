"""Launch a GPU-accelerated JupyterLab server on Modal.

Allows running Open-XTTS-Trainer notebooks on fast GPUs (A10G, L4, A100)
with persistent storage for base models, training runs, and baseline results.

Usage:
    # 1. Authenticate with Modal (one-time):
    #    uvx modal setup
    #
    # 2. Run JupyterLab on an A100 (40GB):
    #    uvx modal run scripts/modal_jupyter.py --gpu A100-40GB
    #
    #    Or on an L4 GPU (24GB VRAM):
    #    uvx modal run scripts/modal_jupyter.py --gpu L4
"""

import secrets
import sys
from pathlib import Path
import modal

app = modal.App("open-xtts-jupyter")

# Pinned commit SHA for reproducible container build
PINNED_COMMIT = "2f1f67ee5261aabd37579d704e62fab4ac177bc4"

# Persistent volumes for base models, training runs, and baseline evidence
volume_cache = modal.Volume.from_name("open-xtts-cache", create_if_missing=True)
volume_runs = modal.Volume.from_name("open-xtts-runs", create_if_missing=True)
volume_checkpoints = modal.Volume.from_name("open-xtts-checkpoints", create_if_missing=True)
volume_baselines = modal.Volume.from_name("open-xtts-baselines", create_if_missing=True)

repo_root = Path(__file__).resolve().parent.parent

# Container image with PyTorch, locked dependencies, and JupyterLab
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv==0.12.13", "packaging", "jupyterlab~=4.2")
    .run_commands(
        f"git clone https://github.com/amitbhakt/open-xtts-trainer.git /root/open-xtts-trainer && cd /root/open-xtts-trainer && git checkout {PINNED_COMMIT}",
        "cd /root/open-xtts-trainer && uv export --frozen --no-hashes --no-emit-project --extra dev --extra eval -o /root/reqs.txt",
        (
            'python3 -c "'
            'from packaging.requirements import Requirement; '
            'lines = [l.strip() for l in open(\'/root/reqs.txt\') if l.strip() and not l.lstrip().startswith(\'#\')]; '
            'codec = [l for l in lines if Requirement(l).name == \'torchcodec\' and Requirement(l).marker.evaluate()]; '
            'open(\'/root/reqs.main.txt\', \'w\').write(\'\\n\'.join(l for l in lines if Requirement(l).name != \'torchcodec\') + \'\\n\'); '
            'open(\'/root/reqs.torchcodec.txt\', \'w\').write(codec[0].split(\';\')[0].strip() + \'\\n\'); '
            '"'
        ),
        "uv pip install --system --break-system-packages --no-deps -r /root/reqs.main.txt",
        "uv pip install --system --break-system-packages --no-deps --index-url https://download.pytorch.org/whl/cpu -r /root/reqs.torchcodec.txt",
        "uv pip install --system --break-system-packages --no-deps -e /root/open-xtts-trainer",
        "pip uninstall -y -q torchvision",
    )
    .add_local_dir(
        repo_root,
        remote_path="/root/open-xtts-trainer",
        ignore=[".git", ".venv", "__pycache__", "runs", "checkpoints", "eval/baselines", "xtts_base"],
    )
)

JUPYTER_PORT = 8888


@app.local_entrypoint()
def main(gpu: str = "A100-40GB", timeout_hours: float = 4.0):
    """Launch JupyterLab inside a Modal Sandbox with the requested GPU."""
    token = secrets.token_urlsafe(16)
    timeout_seconds = int(timeout_hours * 3600)

    print(f"🚀 Launching JupyterLab on Modal with GPU: {gpu} (timeout: {timeout_hours}h)...")

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "jupyter",
            "lab",
            "--no-browser",
            "--allow-root",
            "--ip=0.0.0.0",
            f"--port={JUPYTER_PORT}",
            f"--ServerApp.token={token}",
            "--ServerApp.allow_origin='*'",
            "--ServerApp.allow_remote_access=1",
            encrypted_ports=[JUPYTER_PORT],
            image=image,
            app=app,
            gpu=gpu,
            volumes={
                "/root/open-xtts-trainer/xtts_base": volume_cache,
                "/root/open-xtts-trainer/runs": volume_runs,
                "/root/open-xtts-trainer/checkpoints": volume_checkpoints,
                "/root/open-xtts-trainer/eval/baselines": volume_baselines,
            },
            timeout=timeout_seconds,
            workdir="/root/open-xtts-trainer",
        )

        tunnels = sandbox.tunnels()
        if JUPYTER_PORT not in tunnels:
            print("Failed to establish tunnel for JupyterLab port.", file=sys.stderr)
            sandbox.terminate()
            return

        tunnel = tunnels[JUPYTER_PORT]
        url = f"{tunnel.url}?token={token}"

        print("\n" + "=" * 68)
        print(f"✅ JupyterLab is live on Modal ({gpu})!")
        print(f"🔗 Open in your browser:\n   {url}")
        print("=" * 68)
        print("\nPersistent storage mounted at:")
        print(" - /root/open-xtts-trainer/xtts_base (base models)")
        print(" - /root/open-xtts-trainer/runs (training checkpoints)")
        print(" - /root/open-xtts-trainer/checkpoints (documented model exports & checkpoints)")
        print(" - /root/open-xtts-trainer/eval/baselines (baseline manifests)")
        print("\nPress Ctrl+C in this terminal to shut down the GPU sandbox.\n")

        try:
            sandbox.wait()
        except KeyboardInterrupt:
            print("\nShutting down Modal sandbox...")
            sandbox.terminate()
            print("Sandbox terminated.")
