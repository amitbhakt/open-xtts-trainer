"""F7: the pinned fork's Xtts._load_checkpoint rewrites a legacy model.pth in place (docs/recipe_design.md §2).

The released XTTS-v2 model.pth stores its config as a pickled XttsConfig object. On the first load the fork converts
it to a dict and torch.save()s the result over the original file, which changed the file's SHA-256 from the locked
c7ea2000... to 3077a05b... in Colab runs 1-2. patch_xtts_prevent_checkpoint_mutation() does the conversion in memory.
"""

import hashlib
import ast
import os
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


LOADERS = {"load_checkpoint", "_load_checkpoint"}
GUARD = "patch_xtts_prevent_checkpoint_mutation"


def _call_name(call):
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def _scope_calls(scope):
    """(line, name) of every call made directly in this scope; nested functions are scopes of their own."""
    calls, stack = [], list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Call):
            calls.append((node.lineno, _call_name(node)))
        stack.extend(ast.iter_child_nodes(node))
    return calls


def unguarded_loads(source):
    """Lines of loader calls whose scope (function or module body) doesn't call the guard before them."""
    tree = ast.parse(source)
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    bad = []
    for scope in scopes:
        calls = _scope_calls(scope)
        guards = [line for line, name in calls if name == GUARD]
        bad += [line for line, name in calls if name in LOADERS and not any(g < line for g in guards)]
    return sorted(bad)


def test_every_xtts_checkpoint_load_applies_the_guard():
    """The guard is applied per loader, so a new loader that forgets it would bring F7 back. The guard must run
    earlier in the same function as the load (or earlier in the module body, for module-level loads)."""
    unguarded = []
    for top in ("xtts_trainer", "scripts", "eval"):
        for dirpath, _, names in os.walk(os.path.join(REPO_ROOT, top)):
            for name in names:
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)
                    lines = unguarded_loads(Path(path).read_text(encoding="utf-8"))
                    unguarded += [f"{os.path.relpath(path, REPO_ROOT)}:{line}" for line in lines]
    assert not unguarded, f"XTTS checkpoint loads without the F7 guard before them: {unguarded}"


def test_guard_check_requires_the_guard_first_in_the_same_function():
    ordered = "def f(m, p):\n    patch_xtts_prevent_checkpoint_mutation()\n    m.load_checkpoint(p)\n"
    assert unguarded_loads(ordered) == []
    late = "def f(m, p):\n    m.load_checkpoint(p)\n    patch_xtts_prevent_checkpoint_mutation()\n"
    assert unguarded_loads(late) == [2]
    elsewhere = ("def setup():\n    patch_xtts_prevent_checkpoint_mutation()\n\n"
                 "def f(m, p):\n    return Xtts._load_checkpoint(m, p)\n")
    assert unguarded_loads(elsewhere) == [5]  # guarded in another function isn't guarded here
    assert unguarded_loads("state = _load_checkpoint(m)\n") == [1]
    assert unguarded_loads("def load_checkpoint(self, p):\n    pass\n") == []  # a definition isn't a load
