#!/usr/bin/env bash
# Apply the eval-side (mortal2) patch and rebuild libriichi.
# Idempotent: skips the patch step if it is already applied.
set -euo pipefail

REPO="$HOME/mahjong-llm/Mortal"
PATCHER="/mnt/d/mahjong-llm/work/patch/apply_eval_patch.py"
VENV_PY="$HOME/mahjong-llm/venv/bin/python"

cd "$REPO"

if grep -q "Mortal2" libriichi/src/agent/py_agent.rs; then
    echo "== eval patch already applied, skipping patch =="
else
    echo "== applying eval patch =="
    "$VENV_PY" "$PATCHER" "$REPO"
fi

echo "== building libriichi (release, pymod, no mimalloc) =="
source "$HOME/.cargo/env"
PYO3_PYTHON="$VENV_PY" cargo build -p libriichi --lib --release --no-default-features --features pymod

ls -la target/release/libriichi.so
echo "== build done =="
