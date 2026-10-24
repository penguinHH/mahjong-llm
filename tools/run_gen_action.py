#!/usr/bin/env python3
"""Run the paper's gen_action_data.py on a directory of .mjson log files.

This wraps THUDM/LLM4CardGame's util/mahjong_util/gen_action_data.py and
keeps its exact logic (libriichi GameplayLoader v4 -> obs/action JSONL).

Usage:
    python run_gen_action.py <data_dir_with_mjson_files> <out_dir>

Output: one '<input_file_name>-core.json' (JSONL, appended) per input file
inside <out_dir>, same as the paper's script.
"""
import os
import sys

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")

sys.path.insert(0, MJ_DIR)

from gen_action_data import load_data  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    data_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.isdir(data_path):
        print(f"data dir not found: {data_path}")
        sys.exit(1)
    os.makedirs(out_path, exist_ok=True)
    load_data(data_path, out_path)


if __name__ == "__main__":
    main()
