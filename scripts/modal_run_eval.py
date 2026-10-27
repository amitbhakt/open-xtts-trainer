"""Run open-xtts-trainer evaluations or smoke runs headlessly on Modal.

Allows executing any command on an A100 / L4 / A10G GPU in Modal without
keeping a local Jupyter session open. Results and artifacts are synced to
persistent volumes.

Usage:
    # Run the full Run 5 pipeline on an A100-40GB:
    uvx modal run scripts/modal_run_eval.py --gpu A100-40GB --cmd "python scripts/run_run5_pipeline.py"

    # Quick smoke test or test suite:
    uvx modal run scripts/modal_run_eval.py --gpu A100-40GB --cmd "pytest -q"
"""

import subprocess
from pathlib import Path
import modal

app = modal.App("open-xtts-runner")

# Pinned commit SHA for reproducible container build
PINNED_COMMIT = "2f1f67ee5261aabd37579d704e62fab4ac177bc4"

# Persistent volumes for base models, training runs, checkpoints, and baseline results
volume_cache = modal.Volume.from_name("open-xtts-cache", create_if_missing=True)
volume_runs = modal.Volume.from_name("open-xtts-runs", create_if_missing=True)
volume_checkpoints = modal.Volume.from_name("open-xtts-checkpoints", create_if_missing=True)
volume_baselines = modal.Volume.from_name("open-xtts-baselines", create_if_missing=True)

repo_root = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv==0.12.13", "packaging")
    .run_commands(
        f"git clone https://github.com/amitbhakt/open-xtts-trainer.git /root/open-xtts-trainer && cd /root/open-xtts-trainer && git checkout {PINNED_COMMIT}",
        "cd /root/open-xtts-trainer && uv export --frozen --no-hashes --no-emit-project --extra dev --extra eval -o /root/reqs.txt",
        (
            'python3 -c "'
            "from packaging.requirements import Requirement; "
            "lines = [l.strip() for l in open('/root/reqs.txt') if l.strip() and not l.lstrip().startswith('#')]; "
            "codec = [l for l in lines if Requirement(l).name == 'torchcodec' and Requirement(l).marker.evaluate()]; "
            "open('/root/reqs.main.txt', 'w').write('\\n'.join(l for l in lines if Requirement(l).name != 'torchcodec') + '\\n'); "
            "open('/root/reqs.torchcodec.txt', 'w').write(codec[0].split(';')[0].strip() + '\\n'); "
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


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/root/open-xtts-trainer/xtts_base": volume_cache,
        "/root/open-xtts-trainer/runs": volume_runs,
        "/root/open-xtts-trainer/checkpoints": volume_checkpoints,
        "/root/open-xtts-trainer/eval/baselines": volume_baselines,
    },
    timeout=7200,
)
def run_command_remote(cmd: str):
    """Execute a bash command inside the Modal GPU container and commit outputs."""
    # Stage new overlay source files while excluding generated runtime and output directories.
    # Set fixed author/committer timestamps so the overlay commit SHA is deterministic and
    # stable across container restarts, preserving resume compatibility across runs.
    subprocess.run(
        'git config user.email "bot@example.com" && git config user.name "bot" && '
        "git add -A && "
        '(GIT_COMMITTER_DATE="2026-10-25T00:00:00Z" GIT_AUTHOR_DATE="2026-10-25T00:00:00Z" '
        'git commit -m "modal overlay" || true)',
        shell=True,
        cwd="/root/open-xtts-trainer",
        check=False,
    )
    print(f"🔧 Running in Modal container: {cmd}")
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd="/root/open-xtts-trainer",
        check=True,
    )
    volume_cache.commit()
    volume_runs.commit()
    volume_checkpoints.commit()
    volume_baselines.commit()

    # Collect any newly created/modified evidence files under docs/evidence/phase6_heldout/
    evidence_dir = Path("/root/open-xtts-trainer/docs/evidence/phase6_heldout")
    artifacts = {}
    if evidence_dir.exists():
        for p in evidence_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to("/root/open-xtts-trainer").as_posix()
                try:
                    artifacts[rel] = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    artifacts[rel] = p.read_bytes()
    return {"returncode": proc.returncode, "artifacts": artifacts}


@app.local_entrypoint()
def main(cmd: str = "python scripts/run_run5_pipeline.py", gpu: str = "A100-40GB"):
    print(f"Submitting job to Modal ({gpu})...")
    fn = run_command_remote.with_options(gpu=gpu)
    res = fn.remote(cmd)
    if isinstance(res, dict) and "artifacts" in res:
        for rel_path, content in res["artifacts"].items():
            dest = repo_root / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                dest.write_text(content, encoding="utf-8")
            else:
                dest.write_bytes(content)
            print(f"📥 Retrieved artifact: {rel_path}")
    code = res.get("returncode") if isinstance(res, dict) else res
    if code:
        raise SystemExit(f"remote command failed with exit code {code}; retrieved artifacts may be partial")
    print("✅ Completed successfully on Modal!")
