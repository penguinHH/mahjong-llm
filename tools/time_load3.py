#!/usr/bin/env python3
"""Experiment 3: io location (ext4 vs 9p) + write pattern + bulk200 timing.

Usage: time_load3.py <logs_dir> home|write|bulk200
"""
import os
import shutil
import sys
import time

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")
sys.path.insert(0, MJ_DIR)

logs = sys.argv[1]
mode = sys.argv[2]

from libriichi.dataset import GameplayLoader  # noqa: E402

loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)

if mode == "home":
    home = os.path.expanduser("~/mahjong-llm/work/test_logs")
    os.makedirs(home, exist_ok=True)
    files = sorted(f for f in os.listdir(logs) if f.endswith(".json.gz"))[:3]
    for f in files:
        shutil.copy2(os.path.join(logs, f), os.path.join(home, f))
    for f in files:
        t = time.time()
        loader.load_gz_log_files([os.path.join(home, f)])
        print(f"ext4 single load {f}: {time.time() - t:.3f}s", flush=True)

elif mode == "write":
    files = sorted(f for f in os.listdir(logs) if f.endswith(".json.gz"))[:2]

    out_mnt = "/mnt/d/mahjong-llm/work/time_test_mnt"
    os.makedirs(out_mnt, exist_ok=True)
    pt = os.path.join(out_mnt, "pattern.json")
    t = time.time()
    for i in range(300):
        with open(pt, "a", encoding="utf-8") as f:
            f.write(f"{i}\n")
    print(f"pattern300 mnt(9p): {time.time() - t:.3f}s", flush=True)

    out_home = os.path.expanduser("~/mahjong-llm/work/time_test_home")
    os.makedirs(out_home, exist_ok=True)
    pt = os.path.join(out_home, "pattern.json")
    t = time.time()
    for i in range(300):
        with open(pt, "a", encoding="utf-8") as f:
            f.write(f"{i}\n")
    print(f"pattern300 home(ext4): {time.time() - t:.3f}s", flush=True)

    from gen_action_data import process_file  # noqa: E402

    for tag, outdir in (("mnt", out_mnt), ("home", out_home)):
        g = loader.load_gz_log_files([os.path.join(logs, files[0])])
        t = time.time()
        process_file(files[0], g[0], outdir)
        print(f"process_file[{tag}]: {time.time() - t:.3f}s", flush=True)

elif mode == "bulk200":
    files = sorted(f for f in os.listdir(logs) if f.endswith(".json.gz"))
    paths = [os.path.join(logs, f) for f in files]
    print(f"bulk200 loading {len(paths)} files...", flush=True)
    t = time.time()
    loader.load_gz_log_files(paths)
    print(f"bulk200 done: {time.time() - t:.2f}s", flush=True)
