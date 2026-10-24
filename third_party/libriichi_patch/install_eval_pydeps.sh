#!/usr/bin/env bash
# Install the light pure-python deps used by the paper's eval chain
# (util/llm_client.py + util/mortal_util/llm_engine.py imports). No LLM involved.
set -euo pipefail
V="$HOME/mahjong-llm/venv/bin/python"

if "$V" -c "import openai, requests, tenacity, rlcard" 2>/dev/null; then
    echo "deps already present"
    "$V" -c "import openai, rlcard; print('openai', openai.__version__)"
    exit 0
fi

echo "installing openai requests tenacity rlcard ..."
if "$V" -m pip --version >/dev/null 2>&1; then
    "$V" -m pip install -q openai requests tenacity rlcard
else
    UV="$(command -v uv || true)"
    if [ -z "$UV" ]; then
        for cand in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
            if [ -x "$cand" ]; then UV="$cand"; break; fi
        done
    fi
    if [ -z "$UV" ]; then
        echo "ERROR: neither pip nor uv available" >&2
        exit 1
    fi
    "$UV" pip install --python "$V" -q openai requests tenacity rlcard
fi
"$V" -c "import openai, requests, tenacity, rlcard; print('deps ok:', openai.__version__)"
