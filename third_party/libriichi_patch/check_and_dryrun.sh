#!/usr/bin/env bash
set -euo pipefail
V="$HOME/mahjong-llm/venv/bin/python"
"$V" -c "import numpy; print('numpy', numpy.__version__)"
"$V" /mnt/d/mahjong-llm/work/patch/arena_llmchain_dryrun.py
