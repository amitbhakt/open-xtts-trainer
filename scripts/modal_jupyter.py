"""Launch a GPU-accelerated JupyterLab server on Modal.

Allows running Open-XTTS-Trainer notebooks on fast GPUs (A10G, L4, A100)
with persistent storage for base models, training runs, and baseline results.

Usage:
    # 1. Authenticate with Modal (one-time):
    #    uvx modal setup
    #
    # 2. Run JupyterLab on an A10G GPU (24GB VRAM):
    #    uvx modal run scripts/modal_jupyter.py --gpu A10G
    #
    #    Or on an L4 GPU (24GB VRAM):
    #    uvx modal run scripts/modal_jupyter.py --gpu L4
    #
    #    Or on an A100 (40GB/80GB):
    #    uvx modal run scripts/modal_jupyter.py --gpu A100
"""

import secrets
import sys
import modal

app = modal.App.lookup("open-xtts-jupyter", create_if_missing=True)

# Persistent volume for base models, cache, and baseline evidence
volume = modal.Volume.from_name("open-xtts-cache", create_if_missing=True)

# Container image with PyTorch, locked dependencies, and JupyterLab
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv==0.12.13", "jupyterlab~=4.2")
    .run_commands(
        "git clone https://github.com/amitbhakt/open-xtts-trainer.git /root/open-xtts-trainer",
        "cd /root/open-xtts-trainer && git checkout colab-run5",
        "cd /root/open-xtts-trainer && uv export --frozen --no-hashes --no-emit-project --extra dev --extra eval -o /root/reqs.txt",
        "uv pip install --system --break-system-packages -r /root/reqs.txt",
        "uv pip install --system --break-system-packages -e /root/open-xtts-trainer",
        "pip uninstall -y -q torchvision",
    )
)

JUPYTER_PORT = 8888


@app.local_entrypoint()
def main(gpu: str = "A10G", timeout_hours: float = 4.0):
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
            "--ServerApp.disable_check_xsrf=True",
            encrypted_ports=[JUPYTER_PORT],
            image=image,
            app=app,
            gpu=gpu,
            volumes={
                "/root/open-xtts-trainer/xtts_base": volume,
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
        print("\nPersistent storage mounted at /root/open-xtts-trainer/xtts_base")
        print("Press Ctrl+C in this terminal to shut down the GPU sandbox.\n")

        try:
            sandbox.wait()
        except KeyboardInterrupt:
            print("\nShutting down Modal sandbox...")
            sandbox.terminate()
            print("Sandbox terminated.")
