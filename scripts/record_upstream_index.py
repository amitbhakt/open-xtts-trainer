"""Record the tensor names and shapes inside upstream XTTS-v2 checkpoints, without loading the weights.

PyTorch checkpoints are zip files whose first entry (`<name>/data.pkl`) is the pickled index: every tensor's
name and shape, with the data stored separately. This reads only that pickle, from a local file when present
or via an HTTP range request (the first few MB) otherwise, and writes tests/fixtures/upstream_checkpoint_index.json.
tests/test_spec_parity.py then checks that our model factory builds exactly these names and shapes.

Usage:
    python scripts/record_upstream_index.py                 # range requests to the lockfile URLs
    python scripts/record_upstream_index.py --local_dir xtts_base
"""

import argparse
import collections
import io
import json
import os
import pickle
import struct
import sys
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "tests", "fixtures", "upstream_checkpoint_index.json")
ARTIFACTS = ("model.pth", "dvae.pth")
RANGE_BYTES = 8 << 20  # 8 MiB


class _Opaque:
    """Stand-in for any non-tensor object in the pickle (configs, optimizer state)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _IndexUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return None  # tensor storage lives elsewhere in the zip; only its shape matters here

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *rest: ("tensor", list(size))
        if name == "OrderedDict":
            return collections.OrderedDict
        return type(name, (_Opaque,), {})


def _index_from_pickle(raw: bytes):
    obj = _IndexUnpickler(io.BytesIO(raw)).load()
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return {k: v[1] for k, v in state.items() if isinstance(v, tuple) and v and v[0] == "tensor"}


def index_from_local(path: str):
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        return _index_from_pickle(zf.read(name)), "local zip entry " + name


def index_from_url(url: str):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{RANGE_BYTES - 1}",
                                               "User-Agent": "open-xtts-trainer/record_upstream_index"})
    with urllib.request.urlopen(req) as resp:
        head = resp.read()
    if head[:4] != b"PK\x03\x04":
        raise ValueError(f"{url} did not return a zip local file header")
    fn_len, extra_len = struct.unpack("<HH", head[26:30])
    entry = head[30:30 + fn_len].decode()
    if not entry.endswith("/data.pkl"):
        raise ValueError(f"First zip entry is {entry}, expected */data.pkl")
    # data.pkl is stored uncompressed and the unpickler stops at its STOP opcode
    return _index_from_pickle(head[30 + fn_len + extra_len:]), f"HTTP range 0-{RANGE_BYTES - 1}, zip entry {entry}"


def main():
    parser = argparse.ArgumentParser(description="Record tensor names/shapes of upstream XTTS-v2 checkpoints.")
    parser.add_argument("--local_dir", default=None, help="Read from local files instead of range requests")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    with open(os.path.join(REPO_ROOT, "artifacts.lock.json"), "r", encoding="utf-8") as f:
        lock = json.load(f)

    result = {"pinned_files": {}, "note": "Tensor names and shapes only, read from each checkpoint's pickled index."}
    for name in ARTIFACTS:
        local = os.path.join(args.local_dir, name) if args.local_dir else None
        if local and os.path.exists(local):
            tensors, method = index_from_local(local)
        else:
            tensors, method = index_from_url(lock[name]["url"])
        result["pinned_files"][name] = {
            "url": lock[name]["url"],
            "locked_sha256": lock[name]["sha256"],
            "method": method,
            "n_tensors": len(tensors),
            "tensors": dict(sorted(tensors.items())),
        }
        print(f"{name}: {len(tensors)} tensors ({method})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
