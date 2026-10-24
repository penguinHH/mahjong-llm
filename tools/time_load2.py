#!/usr/bin/env python3
"""Deeper timing: python-side baseline vs rust load, repeats, logging disable, bulk scaling."""
import gzip
import json
import logging
import os
import sys
import time

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")
sys.path.insert(0, MJ_DIR)

print("cpu_count:", os.cpu_count(), flush=True)
try:
    print("sched affinity:", len(os.sched_getaffinity(0)), flush=True)
except Exception:  # noqa: BLE001
    print("sched affinity: n/a", flush=True)
try:
    with open("/sys/fs/cgroup/cpu.max") as f:
        print("cgroup cpu.max:", f.read().strip(), flush=True)
except Exception:  # noqa: BLE001
    print("cgroup cpu.max: n/a", flush=True)

logs = sys.argv[1]
files = sorted(f for f in os.listdir(logs) if f.endswith(".json.gz"))
sel = files[: min(16, len(files))]
paths = [os.path.join(logs, f) for f in sel]

t = time.time()
n_ev = 0
with gzip.open(paths[0], "rt", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            json.loads(line)
            n_ev += 1
print(f"python gzip+json parse file0: {n_ev} events in {time.time() - t:.3f}s", flush=True)

from libriichi.dataset import GameplayLoader  # noqa: E402

loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)

t = time.time()
g = loader.load_gz_log_files([paths[0]])
print(f"rust load file0 (cold): {time.time() - t:.3f}s, {len(g)} games", flush=True)

for i in range(2):
    t = time.time()
    loader.load_gz_log_files([paths[0]])
    print(f"rust load file0 (repeat {i + 1}): {time.time() - t:.3f}s", flush=True)

logging.disable(logging.CRITICAL)
t = time.time()
loader.load_gz_log_files([paths[1]])
print(f"rust load file1 (python logging disabled): {time.time() - t:.3f}s", flush=True)
logging.disable(logging.NOTSET)

for k in (4, 8, 16):
    ps = paths[:k]
    t = time.time()
    loader.load_gz_log_files(ps)
    print(f"rust bulk k={k}: {time.time() - t:.3f}s", flush=True)
