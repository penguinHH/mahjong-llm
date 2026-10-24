#!/usr/bin/env bash
set -uo pipefail
DIR="${1:-$HOME/mahjong-llm/work/arena_smoke}"
PY="$HOME/mahjong-llm/venv/bin/python"

echo "== arena log dir: $DIR =="
ls -la "$DIR" | head -12
echo "== file count =="
ls "$DIR" | wc -l

echo "== sample (first file, first 6 events) =="
f=$(ls "$DIR" | head -1)
zcat "$DIR/$f" 2>/dev/null | head -6

echo "== event census =="
zcat "$DIR"/* 2>/dev/null | "$PY" -c '
import sys, json, collections
c = collections.Counter()
n = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        ev = json.loads(line)
    except Exception:
        continue
    n += 1
    c[ev.get("type")] += 1
print("total events:", n)
print(dict(sorted(c.items())))
'
