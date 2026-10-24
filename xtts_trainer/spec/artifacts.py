"""Vendored upstream XTTS-v2 artifacts (mel_stats.pth, config.json) and their integrity checks.

Expected hashes must match artifacts.lock.json; tests/test_spec_parity.py checks that they do.
"""

import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default path to local mel_stats.pth normalized tensor (matching upstream XTTS-v2)
DEFAULT_MEL_STATS_PATH = os.path.join(PACKAGE_DIR, "mel_stats.pth")
XTTS_MEL_STATS_SHA256 = "1f69422a8a8f344c4fca2f0c6b8d41d2151d6615b7321e48e6bb15ae949b119c"
XTTS_MEL_STATS_URL = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"

XTTS_CONFIG_SHA256 = "ef262b1454dd2a77e1461b0b2cd53e19b8a7624cc131b837d36df67356bc75e8"


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_mel_stats(target_path: Optional[str] = None) -> str:
    """Ensure that the upstream XTTS-v2 mel_stats.pth file exists locally and matches SHA-256.

    Handles read-only package installations (e.g. site-packages) by falling back to a user cache.
    """
    path = target_path or DEFAULT_MEL_STATS_PATH
    user_cache = os.path.expanduser("~/.cache/open_xtts_trainer/mel_stats.pth")
    import urllib.request

    # Verify candidate paths on every load
    for candidate in [path, user_cache]:
        if os.path.exists(candidate):
            try:
                file_hash = sha256_of_file(candidate)
                if file_hash == XTTS_MEL_STATS_SHA256:
                    return candidate
                logger.warning(
                    f"mel_stats.pth at {candidate} SHA-256 mismatch (got {file_hash}, expected {XTTS_MEL_STATS_SHA256})."
                )
                try:
                    os.remove(candidate)
                except (OSError, PermissionError):
                    pass
            except (OSError, PermissionError):
                continue

    # Determine destination: prefer target path, fall back to user_cache if read-only
    dest = path
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        test_file = dest + ".tmp_probe"
        with open(test_file, "w") as f:
            f.write("probe")
        os.remove(test_file)
    except (PermissionError, OSError):
        dest = user_cache
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    logger.info(f"Downloading upstream XTTS-v2 mel_stats.pth from {XTTS_MEL_STATS_URL} to {dest}...")
    urllib.request.urlretrieve(XTTS_MEL_STATS_URL, dest)

    file_hash = sha256_of_file(dest)
    if file_hash != XTTS_MEL_STATS_SHA256:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise RuntimeError(
            f"Downloaded mel_stats.pth SHA-256 mismatch! Expected {XTTS_MEL_STATS_SHA256}, got {file_hash}"
        )

    logger.info(f"Successfully downloaded and verified upstream mel_stats.pth at {dest}.")
    return dest


def verify_vendored_config(path: Optional[str] = None) -> str:
    """Raise if the vendored XTTS-v2 config.json differs from the hash-locked upstream file."""
    from xtts_trainer.spec.xtts_spec import VENDORED_CONFIG_PATH

    path = path or VENDORED_CONFIG_PATH
    file_hash = sha256_of_file(path)
    if file_hash != XTTS_CONFIG_SHA256:
        raise RuntimeError(f"Vendored config {path} SHA-256 {file_hash} != locked {XTTS_CONFIG_SHA256}")
    return path
