#!/usr/bin/env bash
# Deploy an already-built libriichi.so (0001 + 0002) into the paper tree, then run the raw selftest
# against the deployed copy. Use this after `build.sh all --no-deploy`, once nothing is using the old .so
# (e.g. after a data-generation run has finished).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MORTAL_DIR="${MORTAL_DIR:-$HOME/mahjong-llm/Mortal}"
LLM4CARDGAME_DIR="${LLM4CARDGAME_DIR:-/mnt/d/mahjong-llm/refs/LLM4CardGame}"
VENV_PY="${VENV_PY:-$HOME/mahjong-llm/venv/bin/python}"
export MORTAL_DIR LLM4CARDGAME_DIR
SRC="$MORTAL_DIR/target/release/libriichi.so"
DST="$LLM4CARDGAME_DIR/util/mahjong_util/libriichi.so"

[ -f "$DST" ] && cp -f "$DST" "$DST.bak_prev" && echo "backup: $DST.bak_prev"
cp -f "$SRC" "$DST.tmp" && mv -f "$DST.tmp" "$DST"
echo "deployed:"; ls -la "$DST"*
echo "== raw selftest against deployed .so =="
"$VENV_PY" "$HERE/selftest_raw.py"
echo "== deploy done =="
