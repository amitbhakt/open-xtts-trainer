#!/usr/bin/env bash
# Fresh-install check (docs/recipe_design.md §6, phase 9): does this tree install and pass its tests with nothing local?
#
#   scripts/check_fresh_install.sh            # a fresh clone of HEAD (committed code only)
#   scripts/check_fresh_install.sh --worktree # the working tree as git would ship it (tracked + untracked,
#                                             # minus ignored), for checking uncommitted changes
# Installs exactly what uv.lock pins into a new environment, fetches only the small hash-locked artifacts,
# and runs the test suite. Nothing from the current checkout's .venv, xtts_base/ or caches is used.
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [[ "${1:-}" == "--worktree" ]]; then
  (cd "$repo" && git ls-files --cached --others --exclude-standard -z | while IFS= read -r -d '' f; do
     [[ -f "$f" ]] && mkdir -p "$work/src/$(dirname "$f")" && cp "$f" "$work/src/$f"; done)
else
  git clone --quiet "$repo" "$work/src"
fi

cd "$work/src"
export UV_PROJECT_ENVIRONMENT="$work/venv"
uv sync --frozen --extra dev
uv run python scripts/fetch_base_artifacts.py --dest xtts_base --only vocab.json config.json mel_stats.pth
uv run pytest -q -p no:cacheprovider
echo "FRESH INSTALL OK ($([[ "${1:-}" == "--worktree" ]] && echo worktree || echo "HEAD $(git rev-parse --short HEAD)"))"
