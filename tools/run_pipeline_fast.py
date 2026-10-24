#!/usr/bin/env python3
"""Robust/parallel pipeline (fallback path when stock load_data is too slow).

Per-file loading (single file per call), resumable, multiprocessing, ext4-friendly workdir.

Steps:
  0. stage sources (.mjai.json.zst / .mjson / gz) -> <work>/logs/*.json.gz   (resume: skip existing)
  1. for each staged log: load single + paper's process_file -> <work>/logs_format4/<name>-core.json
     (resume: skip existing output; failures -> <work>/failed.txt, partial output removed)
  2. concatenate -> <work>/core_all.jsonl
  3. paper's convert_riichi -> <work>/sft_riichi.jsonl (+preview)

Usage:
  python run_pipeline_fast.py --raw <src_dir> --work <work_dir> [--num N] [--processes P] [--fresh]
  (--num 0 = all sources; --fresh wipes logs/ and logs_format4/ first)
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import time

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")
sys.path.insert(0, MJ_DIR)
sys.path.insert(0, PAPER)

import libriichi  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402
from gen_action_data import process_file  # noqa: E402
from convert_data import convert_riichi  # noqa: E402

_LOADER = None


def stage_log(src, dst_dir):
    """Convert one source log into gzip JSONL at dst_dir. Returns conversion mode."""
    base = os.path.basename(src)
    for suf in (".mjai.json.zst", ".json.zst", ".mjson", ".json.gz", ".gz", ".json"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    out_path = os.path.join(dst_dir, base + ".json.gz")

    with open(src, "rb") as f:
        head = f.read(4)
    if head[:2] == b"\x1f\x8b":  # already gzip
        shutil.copy2(src, out_path)
        return "gz"
    if head[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        with open(src, "rb") as fin, gzip.open(out_path, "wb") as fout:
            dctx.copy_stream(fin, fout)
        return "zstd->gz"
    with open(src, "rb") as fin, gzip.open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return "plain->gz"


def _init_worker():
    global _LOADER
    _LOADER = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)


def _work(t):
    fname, logs_dir, out_dir = t
    src = os.path.join(logs_dir, fname)
    dst = os.path.join(out_dir, fname + "-core.json")
    if os.path.exists(dst):
        return fname, -2, 0.0, None
    t0 = time.time()
    try:
        games_ls = _LOADER.load_gz_log_files([src])
        process_file(fname, games_ls[0], out_dir)
        return fname, len(games_ls[0]), time.time() - t0, None
    except Exception as e:  # noqa: BLE001
        try:
            os.remove(dst)
        except OSError:
            pass
        return fname, -1, time.time() - t0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--num", type=int, default=0, help="limit source files (0 = all)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N sorted source files (e.g. --skip 7000 --num 1000 = held-out games 7001-8000)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    logs_dir = os.path.join(args.work, "logs")
    fmt_dir = os.path.join(args.work, "logs_format4")
    if args.fresh:
        for d in (logs_dir, fmt_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(fmt_dir, exist_ok=True)

    print(f"libriichi: {libriichi.__file__}", flush=True)

    srcs = []
    for pat in ("**/*.mjson", "**/*.mjai.json.zst", "**/*.json.zst"):
        srcs.extend(glob.glob(os.path.join(args.raw, pat), recursive=True))
    srcs = sorted(set(srcs))[args.skip:]
    if args.num > 0:
        srcs = srcs[: args.num]
    print(f"source logs: {len(srcs)}", flush=True)

    staged = 0
    for i, s in enumerate(srcs, 1):
        base = os.path.basename(s)
        for suf in (".mjai.json.zst", ".json.zst", ".mjson", ".json.gz", ".gz", ".json"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        dst = os.path.join(logs_dir, base + ".json.gz")
        if os.path.exists(dst):
            continue
        stage_log(s, logs_dir)
        staged += 1
        if i % 500 == 0 or i == len(srcs):
            print(f"  staged {i}/{len(srcs)}", flush=True)
    print(f"staged {staged} new logs -> {logs_dir}", flush=True)

    todo = [f for f in sorted(os.listdir(logs_dir)) if f.endswith(".json.gz")
            and not os.path.exists(os.path.join(fmt_dir, f + "-core.json"))]
    print(f"to process: {len(todo)}", flush=True)

    ok = errs = 0
    failed = []
    t_start = time.time()

    if args.processes <= 1:
        global _LOADER
        _LOADER = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)
        for i, f in enumerate(todo, 1):
            fname, n, dur, err = _work((f, logs_dir, fmt_dir))
            if err:
                errs += 1
                failed.append(f"{fname}\t{err}")
                print(f"  ERROR {fname}: {err}", flush=True)
            else:
                ok += 1
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t_start
                print(f"  [{i}/{len(todo)}] ok={ok} err={errs} elapsed={el:.0f}s rate={i / max(el, 1e-9):.2f}/s", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(args.processes, initializer=_init_worker) as pool:
            tasks = [(f, logs_dir, fmt_dir) for f in todo]
            for i, (fname, n, dur, err) in enumerate(pool.imap_unordered(_work, tasks), 1):
                if err:
                    errs += 1
                    failed.append(f"{fname}\t{err}")
                    print(f"  ERROR {fname}: {err}", flush=True)
                else:
                    ok += 1
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t_start
                    print(f"  [{i}/{len(todo)}] ok={ok} err={errs} elapsed={el:.0f}s rate={i / max(el, 1e-9):.2f}/s", flush=True)

    if failed:
        with open(os.path.join(args.work, "failed.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        print(f"failures written to {os.path.join(args.work, 'failed.txt')}", flush=True)

    core_files = sorted(glob.glob(os.path.join(fmt_dir, "*-core.json")))
    core_all = os.path.join(args.work, "core_all.jsonl")
    n_core = 0
    with open(core_all, "w", encoding="utf-8") as out:
        for cf in core_files:
            with open(cf, encoding="utf-8") as fin:
                for line in fin:
                    out.write(line)
                    n_core += 1
    print(f"core samples: {n_core} from {len(core_files)} files -> {core_all}", flush=True)

    # convert_riichi() read_jsonl()s its whole input into memory; on the full core_all.jsonl (8.3 GB for
    # 7000 games) that OOMs the 16 GB WSL. It handles lines independently, so convert per game file in the
    # same order as the concatenation above (byte-identical output; see run_convert_streaming.py).
    print("convert_riichi per game file (filters: rank==0, legal_actions>=2) ...", flush=True)
    sft_path = os.path.join(args.work, "sft_riichi.jsonl")
    n_items, first_item = 0, None
    with open(sft_path, "w", encoding="utf-8") as f:
        for cf in core_files:
            for it in convert_riichi(cf):
                f.write(it)
                f.write("\n")
                n_items += 1
                if first_item is None:
                    first_item = it
    print(f"sft samples: {n_items} -> {sft_path}", flush=True)

    if first_item:
        first = json.loads(first_item)
        print("---- sample preview ----", flush=True)
        print(first["instruction"][:900], flush=True)
        print("OUTPUT:", first["output"], flush=True)


if __name__ == "__main__":
    main()
