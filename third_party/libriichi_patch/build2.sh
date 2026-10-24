#!/bin/bash
set -e
set -o pipefail

M=$HOME/mahjong-llm/Mortal
P=/mnt/d/mahjong-llm
PL=$P/work/patch
VENV=$HOME/mahjong-llm/venv/bin/python

source $HOME/.cargo/env 2>/dev/null || true

echo "=== updating raw.rs + rebuild ==="
cp $PL/raw.rs $M/libriichi/src/state/raw.rs
cd $M
PYO3_PYTHON=$VENV cargo build -p libriichi --lib --release --no-default-features --features pymod 2>&1 | tail -30
SO=$M/target/release/libriichi.so
ls -la $SO
cp $SO $P/refs/LLM4CardGame/util/mahjong_util/libriichi.so
echo "=== selftest ==="
$VENV $PL/selftest_raw.py
