#!/bin/bash
set -e
set -o pipefail

M=$HOME/mahjong-llm/Mortal
P=/mnt/d/mahjong-llm
PL=$P/work/patch
VENV=$HOME/mahjong-llm/venv/bin/python

source $HOME/.cargo/env 2>/dev/null || true

echo "=== git status (should be clean) ==="
git -C $M status --porcelain | head -20
echo "=== copying raw.rs ==="
cp $PL/raw.rs $M/libriichi/src/state/raw.rs
echo "=== applying patches ==="
$VENV $PL/apply_patch.py $M
echo "=== cargo build (no-mimalloc, pymod) ==="
cd $M
PYO3_PYTHON=$VENV cargo build -p libriichi --lib --release --no-default-features --features pymod 2>&1 | tail -30
SO=$M/target/release/libriichi.so
ls -la $SO
echo "=== deploying .so ==="
cp $P/refs/LLM4CardGame/util/mahjong_util/libriichi.so $P/refs/LLM4CardGame/util/mahjong_util/libriichi.so.bak_prev
cp $SO $P/refs/LLM4CardGame/util/mahjong_util/libriichi.so
echo "deployed: $(ls -la $P/refs/LLM4CardGame/util/mahjong_util/libriichi.so)"
echo "=== selftest ==="
$VENV $PL/selftest_raw.py
