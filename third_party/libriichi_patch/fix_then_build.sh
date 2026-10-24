#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mahjong-llm/Mortal"
"$HOME/mahjong-llm/venv/bin/python" /mnt/d/mahjong-llm/work/patch/fix_eval_build.py "$(pwd)"
bash /mnt/d/mahjong-llm/work/patch/build_eval.sh
