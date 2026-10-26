"""F7: the pinned fork's Xtts._load_checkpoint rewrites a legacy model.pth in place (docs/recipe_design.md §2).

The released XTTS-v2 model.pth stores its config as a pickled XttsConfig object. On the first load the fork converts
it to a dict and torch.save()s the result over the original file, which changed the file's SHA-256 from the locked
c7ea2000... to 3077a05b... in Colab runs 1-2. patch_xtts_prevent_checkpoint_mutation() does the conversion in memory.
"""

import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from xtts_trainer import patch_xtts_prevent_checkpoint_mutation

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _legacy_checkpoint(path):
    """A checkpoint shaped like the released model.pth: the config is a pickled XttsConfig object, with its classes
    under the legacy module path the released file uses (upstream's _legacy_safe_globals allowlists exactly those)."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import _legacy_safe_globals

    with _legacy_safe_globals():
        torch.save({"config": XttsConfig(), "model": {"gpt.w": torch.arange(4.0)}}, path)
    return path


def test_upstream_f7_load_rewrites_a_legacy_checkpoint_in_place(tmp_path):
    from TTS.tts.models.xtts import Xtts

    patch_xtts_prevent_checkpoint_mutation()  # keeps upstream's loader as _open_xtts_upstream_load_checkpoint
    path = _legacy_checkpoint(str(tmp_path / "model.pth"))
    before = _sha256(path)
    state = Xtts._open_xtts_upstream_load_checkpoint(Xtts.__new__(Xtts), path)
    assert torch.equal(state["gpt.w"], torch.arange(4.0))
    assert _sha256(path) != before, "upstream no longer rewrites the file: F7 is gone, revisit the guard"
    assert isinstance(torch.load(path, weights_only=True)["config"], dict)  # the object was replaced by a dict


def test_guarded_load_leaves_the_checkpoint_untouched(tmp_path):
    from TTS.tts.models.xtts import Xtts

    patch_xtts_prevent_checkpoint_mutation()
    path = _legacy_checkpoint(str(tmp_path / "model.pth"))
    before = _sha256(path)
    state = Xtts._load_checkpoint(Xtts.__new__(Xtts), path)
    assert torch.equal(state["gpt.w"], torch.arange(4.0))
    assert _sha256(path) == before


def test_every_xtts_checkpoint_load_applies_the_guard():
    """The guard is applied per loader, so a new loader that forgets it would bring F7 back."""
    unguarded = []
    for top in ("xtts_trainer", "scripts", "eval"):
        for dirpath, _, names in os.walk(os.path.join(REPO_ROOT, top)):
            for name in names:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                code = re.sub(r"#.*", "", Path(path).read_text(encoding="utf-8"))
                # Method calls (.load_checkpoint, ._load_checkpoint) and direct calls; not definitions
                calls = re.search(r"(?<!def )(?<![\w])\.?_?load_checkpoint\(", code)
                if calls and "patch_xtts_prevent_checkpoint_mutation()" not in code:
                    unguarded.append(os.path.relpath(path, REPO_ROOT))
    assert not unguarded, f"these modules load XTTS checkpoints without the F7 guard: {unguarded}"
