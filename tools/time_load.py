#!/usr/bin/env python3
"""Time each stage of the pipeline to locate slowness/hangs.

Usage: python time_load.py <logs_dir_with_gz> <write_test_outdir>
"""
import json
import os
import sys
import time

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")
sys.path.insert(0, MJ_DIR)

logs = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

from libriichi.dataset import GameplayLoader  # noqa: E402

files = sorted(f for f in os.listdir(logs) if f.endswith(".json.gz"))
paths = [os.path.join(logs, f) for f in files]
loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)

print(f"PHASE A: single-file loads, {len(paths)} files", flush=True)
t_all = time.time()
slow = 0
for i, (f, p) in enumerate(zip(files, paths), 1):
    t = time.time()
    g = loader.load_gz_log_files([p])
    dt = time.time() - t
    if dt > 1.0:
        slow += 1
    if dt > 1.0 or i <= 5 or i % 50 == 0:
        print(f"  [{i}/{len(paths)}] {f}: {len(g)} games, {dt:.3f}s", flush=True)
print(f"PHASE A done: total {time.time() - t_all:.1f}s ({slow} files > 1s)", flush=True)

print("PHASE B: bulk parallel load (same as paper)", flush=True)
t = time.time()
data = loader.load_gz_log_files(paths)
print(f"PHASE B done: {len(data)} files in {time.time() - t:.2f}s", flush=True)

print("PHASE C: obs extraction, first game", flush=True)
game = data[0][0]
t = time.time()
raw = game.take_obs_raw()
dt_c = time.time() - t
t = time.time()
obs = game.take_obs()
print(f"  take_obs_raw: {len(raw)} events in {dt_c:.3f}s; take_obs: {obs.shape} in {time.time() - t:.3f}s", flush=True)

print("PHASE E: 300x open-append-close write pattern", flush=True)
t = time.time()
pt = os.path.join(outdir, "pattern-test.json")
for i in range(300):
    with open(pt, "a", encoding="utf-8") as f:
        f.write(f"{i}\n")
print(f"PHASE E done: {time.time() - t:.3f}s", flush=True)

print("PHASE D: full process_file repro on file #1", flush=True)
from gen_action_data import process_file  # noqa: E402

t = time.time()
process_file(files[0], data[0], outdir)
print(f"PHASE D done: {time.time() - t:.2f}s -> {sorted(os.listdir(outdir))[:3]}", flush=True)
