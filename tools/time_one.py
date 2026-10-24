#!/usr/bin/env python3
"""Measure wall vs CPU time for single/bulk loads using resource module.

Usage: time_one.py <logs_dir> single|N
"""
import os
import resource
import sys
import time

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
sys.path.insert(0, os.path.join(PAPER, "util", "mahjong_util"))

from libriichi.dataset import GameplayLoader  # noqa: E402

logs = sys.argv[1]
mode = sys.argv[2]
files = sorted(f for f in os.listdir(logs) if f.endswith(".json.gz"))
loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)


def measure(paths, tag):
    u0 = resource.getrusage(resource.RUSAGE_SELF)
    w0 = time.time()
    loader.load_gz_log_files(paths)
    w = time.time() - w0
    u1 = resource.getrusage(resource.RUSAGE_SELF)
    print(f"{tag}: wall {w:.3f}s user {u1.ru_utime - u0.ru_utime:.3f}s sys {u1.ru_stime - u0.ru_stime:.3f}s", flush=True)


if mode == "single":
    for i in range(3):
        measure([os.path.join(logs, files[i])], f"single[{i}]")
else:
    k = int(mode)
    measure([os.path.join(logs, f) for f in files[:k]], f"bulk{k}")
