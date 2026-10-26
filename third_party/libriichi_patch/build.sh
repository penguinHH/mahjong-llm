#!/usr/bin/env bash
# Patch, build, deploy and self-test libriichi in one go.
#
#   bash build.sh [all|0001|0002] [--no-deploy]
#
# Paths come from the environment (defaults are the layout this project used under WSL):
#   MORTAL_DIR       Mortal checkout at 0cff2b5         (default: ~/mahjong-llm/Mortal)
#   LLM4CARDGAME_DIR the paper's code (LLM4CardGame)    (default: /mnt/d/mahjong-llm/refs/LLM4CardGame)
#   VENV_PY          Python that builds/imports pyo3    (default: ~/mahjong-llm/venv/bin/python)
#
# Build flags: --no-default-features drops mimalloc, which is more stable with the multi-process data
# pipeline (it forks); --features pymod builds the Python extension.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MORTAL_DIR="${MORTAL_DIR:-$HOME/mahjong-llm/Mortal}"
LLM4CARDGAME_DIR="${LLM4CARDGAME_DIR:-/mnt/d/mahjong-llm/refs/LLM4CardGame}"
VENV_PY="${VENV_PY:-$HOME/mahjong-llm/venv/bin/python}"
WHICH="${1:-all}"
DEPLOY=1
[ "${2:-}" = "--no-deploy" ] && DEPLOY=0
export MORTAL_DIR LLM4CARDGAME_DIR

echo "== patch =="
bash "$HERE/apply.sh" "$MORTAL_DIR" "$WHICH"

echo "== cargo build (release, pymod, no mimalloc) =="
source "$HOME/.cargo/env" 2>/dev/null || true
cd "$MORTAL_DIR"
PYO3_PYTHON="$VENV_PY" cargo build -p libriichi --lib --release --no-default-features --features pymod
SO="$MORTAL_DIR/target/release/libriichi.so"
ls -la "$SO"

if [ "$DEPLOY" = 1 ]; then
    DST="$LLM4CARDGAME_DIR/util/mahjong_util/libriichi.so"
    echo "== deploy to $DST =="
    [ -f "$DST" ] && cp -f "$DST" "$DST.bak_prev" && echo "backup: $DST.bak_prev"
    cp -f "$SO" "$DST.tmp" && mv -f "$DST.tmp" "$DST"     # replace in one step; a running process keeps the old inode
    echo "== selftest against the deployed .so =="
    "$VENV_PY" "$HERE/selftest_raw.py"
fi
echo "== done =="
