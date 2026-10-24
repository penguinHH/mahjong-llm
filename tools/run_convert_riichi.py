#!/usr/bin/env python3
"""Run the paper's convert_riichi (obs/action -> SFT instruction/output).

Wraps THUDM/LLM4CardGame's convert_data.py::convert_riichi with the exact
same filters as the paper:
  1) keep only samples where obs['rank'] == 0 (final 1st place)
  2) skip samples with less than 2 legal actions

Usage:
    python run_convert_riichi.py <input_core_jsonl> <output_sft_jsonl>
"""
import os
import sys

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
sys.path.insert(0, PAPER)

from convert_data import convert_riichi  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    items = convert_riichi(src)
    out_dir = os.path.dirname(dst)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item)
            f.write("\n")
    print(f"wrote {len(items)} items -> {dst}")


if __name__ == "__main__":
    main()
