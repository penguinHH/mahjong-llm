#!/usr/bin/env bash
# Stub-engine arena smoke test of the mortal2 path:  bash run_arena_smoke.sh [seed_count=4] [seed=10000] [log_dir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MORTAL_DIR="${MORTAL_DIR:-$HOME/mahjong-llm/Mortal}"
VENV_PY="${VENV_PY:-$HOME/mahjong-llm/venv/bin/python}"
"$VENV_PY" "$HERE/arena_stub_smoke.py" \
    --lib-dir "$MORTAL_DIR/target/release" \
    --seed-count "${1:-4}" \
    --seed "${2:-10000}" \
    --log-dir "${3:-$HOME/mahjong-llm/work/arena_smoke}"
