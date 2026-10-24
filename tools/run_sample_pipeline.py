#!/usr/bin/env python3
"""One-shot sample pipeline: mjai logs -> obs/action samples -> SFT jsonl.

Steps (paper's original code unchanged for steps 2-4):
  0. pick <num> source logs from <raw> (recursive), convert to gzip JSONL into
     <work>/logs/   (handles .mjai.json.zst / .mjson / gzip / plain input)
  1. gen_action_data.load_data  -> <work>/logs_format4/*-core.json
  2. concatenate all *-core.json -> <work>/core_all.jsonl
  3. convert_riichi             -> <work>/sft_riichi.jsonl  (instruction/output)

Usage (inside WSL):
    ~/mahjong-llm/venv/bin/python tools/run_sample_pipeline.py \
        --raw /mnt/d/mahjong-llm/data/raw/tenhou-houou-2026 \
        --num 200 \
        --work /mnt/d/mahjong-llm/work/sample200
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import sys

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")


def die(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


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
    # assume plain JSONL text
    with open(src, "rb") as fin, gzip.open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return "plain->gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with source logs (recursive)")
    ap.add_argument("--num", type=int, default=200, help="how many games to process (0 = all)")
    ap.add_argument("--work", required=True, help="working directory for outputs")
    args = ap.parse_args()

    sys.path.insert(0, MJ_DIR)
    sys.path.insert(0, PAPER)

    try:
        import libriichi  # noqa: F401
        from libriichi.dataset import GameplayLoader  # noqa: F401
    except Exception as e:  # noqa: BLE001
        die(
            "libriichi 不可用。先编译 Mortal 的 libriichi 并把产物拷到 "
            f"{MJ_DIR}/libriichi.so 。原始错误: {e}"
        )

    from gen_action_data import load_data  # noqa: E402
    from convert_data import convert_riichi  # noqa: E402

    files = []
    for pat in ("**/*.mjson", "**/*.mjai.json.zst", "**/*.json.zst"):
        files.extend(glob.glob(os.path.join(args.raw, pat), recursive=True))
    files = sorted(set(files))
    if not files:
        listing = []
        for root, _dirs, fs in os.walk(args.raw):
            for f in fs[:20]:
                listing.append(os.path.join(root, f))
        die(f"在 {args.raw} 里没找到可用的牌谱。目录里前几个文件: {listing}")
    sel = files if args.num <= 0 else files[: args.num]
    print(f"found {len(files)} source logs, using {len(sel)}")

    logs_dir = os.path.join(args.work, "logs")
    fmt_dir = os.path.join(args.work, "logs_format4")
    for d in (logs_dir, fmt_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    modes = {}
    for f in sel:
        m = stage_log(f, logs_dir)
        modes[m] = modes.get(m, 0) + 1
    print(f"staged {len(sel)} logs -> {logs_dir}  {modes}")

    print("[1/3] gen_action_data ...")
    load_data(logs_dir, fmt_dir)

    core_files = sorted(glob.glob(os.path.join(fmt_dir, "*-core.json")))
    if not core_files:
        die("gen_action_data 没有产出任何 *-core.json")
    core_all = os.path.join(args.work, "core_all.jsonl")
    n_core = 0
    with open(core_all, "w", encoding="utf-8") as out:
        for cf in core_files:
            with open(cf, "r", encoding="utf-8") as fin:
                for line in fin:
                    out.write(line)
                    n_core += 1
    print(f"[2/3] core samples: {n_core} ({len(core_files)} files) -> {core_all}")

    print("[3/3] convert_riichi (filters: rank==0, legal_actions>=2) ...")
    items = convert_riichi(core_all)
    sft_path = os.path.join(args.work, "sft_riichi.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item)
            f.write("\n")
    print(f"       sft samples: {len(items)} -> {sft_path}")

    if items:
        first = json.loads(items[0])
        instr = first["instruction"]
        print("\n---- sample preview ----")
        print("INSTRUCTION (first 900 chars):")
        print(instr[:900] + ("..." if len(instr) > 900 else ""))
        print("\nOUTPUT:")
        print(first["output"])
        print("------------------------")


if __name__ == "__main__":
    main()
