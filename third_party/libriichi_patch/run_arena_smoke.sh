#!/usr/bin/env bash
set -euo pipefail
"$HOME/mahjong-llm/venv/bin/python" /mnt/d/mahjong-llm/work/patch/arena_stub_smoke.py \
    --lib-dir "$HOME/mahjong-llm/Mortal/target/release" \
    --seed-count "${1:-2}" \
    --seed "${2:-10000}" \
    --log-dir "${3:-$HOME/mahjong-llm/work/arena_smoke}"
