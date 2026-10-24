#!/usr/bin/env python3
"""Stage 3 of run_pipeline_fast.py without loading everything into memory.

The paper's convert_riichi() starts with read_jsonl(), which parses the whole input into a list of dicts.
On the full 7000-game core_all.jsonl (8.3 GB, 4.58M lines) that needs far more than the 16 GB WSL has,
so the original stage 3 dies silently (OOM). convert_riichi processes each line independently
(filter rank==0 / legal_actions>=2, then fill prompt_riichi), so calling it on each per-game
<name>-core.json in the same sorted order as the stage-2 concatenation gives byte-identical output.

Usage:
  ~/mahjong-llm/venv/bin/python /mnt/d/mahjong-llm/tools/run_convert_streaming.py --work ~/mahjong-llm/work/full7000 [--processes 6]
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
sys.path.insert(0, os.path.join(PAPER, "util", "mahjong_util"))
sys.path.insert(0, PAPER)

from convert_data import convert_riichi  # noqa: E402


def _convert(path):
    return convert_riichi(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--processes", type=int, default=6)
    args = ap.parse_args()

    fmt_dir = os.path.join(args.work, "logs_format4")
    core_files = sorted(glob.glob(os.path.join(fmt_dir, "*-core.json")))    # same order as stage 2
    sft_path = os.path.join(args.work, "sft_riichi.jsonl")
    tmp_path = sft_path + ".part"
    print(f"convert_riichi per game file: {len(core_files)} files, {args.processes} processes", flush=True)

    t0, n_items, first = time.time(), 0, None
    with open(tmp_path, "w", encoding="utf-8") as f, mp.Pool(args.processes) as pool:
        for i, items in enumerate(pool.imap(_convert, core_files, chunksize=8), 1):   # imap keeps order
            for it in items:
                f.write(it)
                f.write("\n")
            n_items += len(items)
            if first is None and items:
                first = items[0]
            if i % 500 == 0 or i == len(core_files):
                el = time.time() - t0
                print(f"  [{i}/{len(core_files)}] sft={n_items} elapsed={el:.0f}s", flush=True)
    os.replace(tmp_path, sft_path)                # only a complete run produces sft_riichi.jsonl
    print(f"sft samples: {n_items} -> {sft_path}", flush=True)

    if first:
        js = json.loads(first)
        print("---- sample preview ----", flush=True)
        print(js["instruction"][:900], flush=True)
        print("OUTPUT:", js["output"], flush=True)


if __name__ == "__main__":
    main()
