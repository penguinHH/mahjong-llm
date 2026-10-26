#!/usr/bin/env bash
# End-to-end dry run of the paper's LLMEngine against the patched arena, with a fake LLM (no model needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${VENV_PY:-$HOME/mahjong-llm/venv/bin/python}"
"$VENV_PY" -c "import numpy; print('numpy', numpy.__version__)"
"$VENV_PY" "$HERE/arena_llmchain_dryrun.py"
