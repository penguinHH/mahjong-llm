#!/usr/bin/env bash
# Deploy the combined libriichi.so (take_obs_raw + mortal2 patches) into the
# paper tree, then re-run the raw selftest against the deployed copy.
#
# Run this ONLY after the full data-generation task has finished.
set -euo pipefail
SRC="$HOME/mahjong-llm/Mortal/target/release/libriichi.so"
DST_DIR="/mnt/d/mahjong-llm/refs/LLM4CardGame/util/mahjong_util"
DST="$DST_DIR/libriichi.so"
V="$HOME/mahjong-llm/venv/bin/python"

if [ -f "$DST" ]; then
    cp -f "$DST" "$DST.bak_prev2"
    echo "backup: $DST.bak_prev2"
fi
cp -f "$SRC" "$DST"
echo "deployed:"; ls -la "$DST"*

echo "== raw selftest against deployed .so =="
"$V" /mnt/d/mahjong-llm/work/patch/selftest_raw.py
echo "== deploy done =="
