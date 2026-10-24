#!/usr/bin/env python3
"""把全量 108 万条决策（full7000，7000 局）转成局面张量 + Mortal 标签，分块写入 npy（内存映射，不一次性占满内存）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/build_full.py
    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/build_full.py --extra   # 加第 8001 局以后 → fullx_*

输入：full7000/sft_riichi.jsonl（prompt + 牌谱动作）、distill/mortal_full7000.jsonl（md5 → Mortal 动作、Q 值、局 id）
输出 work/mmj/full_*.npy（行对齐）：
  planes uint8 [N,58,34]、scalars float32 [N,71]、y_mortal / y_human int16（46 动作下标，-1 = 无标签）、
  q float16 [N,46]（Mortal 的 Q 值，非合法动作为 NaN，软标签蒸馏用）、game int32（局编号，按局划验证集）、
  in100k bool（是否属于 LLM 用的那 10 万条子集）；full_games.json = 局 id 列表。
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from numpy.lib.format import open_memmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actions import A, N_ACTIONS, norm  # noqa: E402
from state_features import N_PLANES, N_SCALARS, check, encode, fields  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work")
OUT = f"{W}/mmj"


def work(line):
    r = json.loads(line)
    f = fields(r["instruction"])
    pl, sc = encode(f)
    bad = check(f, pl, sc)
    human = norm(json.loads(r["output"])["action"])
    return pl, sc, hashlib.md5(r["instruction"].encode()).hexdigest(), A.get(human, -1), len(bad)


def build(src, labels, out, game_offset=0, sub=frozenset()):
    """src（SFT jsonl）+ labels（Mortal 标签 jsonl）→ {OUT}/{out}_*.npy；局编号从 game_offset 开始。"""
    n = sum(1 for _ in open(src, "rb"))
    print(f"[{out}] {n} 行；读入 Mortal 标签……", flush=True)
    lab, games = {}, {}
    for l in open(labels):
        d = json.loads(l)
        g = games.setdefault(d["game"], len(games)) + game_offset
        q = np.full(N_ACTIONS, np.nan, np.float16)
        for a, v in d["q"].items():
            q[A[norm(a)]] = v
        lab[d["md5"]] = (A[norm(d["mortal"])], q, g)

    mm = {
        "planes": open_memmap(f"{OUT}/{out}_planes.npy", "w+", np.uint8, (n, N_PLANES, 34)),
        "scalars": open_memmap(f"{OUT}/{out}_scalars.npy", "w+", np.float32, (n, N_SCALARS)),
        "q": open_memmap(f"{OUT}/{out}_q.npy", "w+", np.float16, (n, N_ACTIONS)),
    }
    y_m = np.full(n, -1, np.int16)
    y_h = np.full(n, -1, np.int16)
    game = np.full(n, -1, np.int32)
    in100k = np.zeros(n, bool)
    bad = 0
    with mp.Pool(8) as pool, open(src, encoding="utf-8") as fi:
        for i, (pl, sc, md5, h, nb) in enumerate(pool.imap(work, fi, chunksize=1000)):
            mm["planes"][i] = pl
            mm["scalars"][i] = sc
            y_h[i] = h
            bad += nb
            if md5 in lab:
                y_m[i], mm["q"][i], game[i] = lab[md5]
            else:
                mm["q"][i] = np.nan
            in100k[i] = md5 in sub
            if (i + 1) % 200000 == 0:
                print(f"  {i + 1}/{n}", flush=True)
    for v in mm.values():
        v.flush()
    np.save(f"{OUT}/{out}_y_mortal.npy", y_m)
    np.save(f"{OUT}/{out}_y_human.npy", y_h)
    np.save(f"{OUT}/{out}_game.npy", game)
    np.save(f"{OUT}/{out}_in100k.npy", in100k)
    json.dump(sorted(games, key=games.get), open(f"{OUT}/{out}_games.json", "w"))
    print(f"[{out}] 完成：自检不一致 {bad} 项；Mortal 标签 {(y_m >= 0).sum()}，牌谱动作可解析 {(y_h >= 0).sum()}，"
          f"10 万子集 {in100k.sum()}，局数 {len(games)}")


def concat(a, b, out):
    """{a}_* 后面接 {b}_* → {out}_*（大数组逐块拷，不一次性进内存）。"""
    for k in ("planes", "scalars", "q"):
        x, y = np.load(f"{OUT}/{a}_{k}.npy", mmap_mode="r"), np.load(f"{OUT}/{b}_{k}.npy", mmap_mode="r")
        z = open_memmap(f"{OUT}/{out}_{k}.npy", "w+", x.dtype, (len(x) + len(y),) + x.shape[1:])
        for i in range(0, len(x), 200000):
            j = min(i + 200000, len(x))
            z[i:j] = x[i:j]
        for i in range(0, len(y), 200000):
            j = min(i + 200000, len(y))
            z[len(x) + i:len(x) + j] = y[i:j]
        z.flush()
    for k in ("y_mortal", "y_human", "game", "in100k"):
        np.save(f"{OUT}/{out}_{k}.npy", np.concatenate([np.load(f"{OUT}/{a}_{k}.npy"), np.load(f"{OUT}/{b}_{k}.npy")]))
    print(f"[{out}] = {a} + {b}：{len(np.load(f'{OUT}/{out}_game.npy'))} 条")


def main():
    if "--extra" in sys.argv:      # 第 8001 局以后（extra8000）→ extra_*，再和 full_* 拼成 fullx_*（局编号顺延，测试局 7001-8000 不在里面）
        offset = int(np.load(f"{OUT}/full_game.npy").max()) + 1
        build(f"{W}/extra8000/sft_riichi.jsonl", f"{W}/distill/mortal_extra8000.jsonl", "extra", offset)
        concat("full", "extra", "fullx")
        return
    sub = {hashlib.md5(json.loads(l)["instruction"].encode()).hexdigest()
           for l in open(f"{W}/sft/riichi_100k.jsonl", encoding="utf-8")}
    build(f"{W}/full7000/sft_riichi.jsonl", f"{W}/distill/mortal_full7000.jsonl", "full", 0, sub)


if __name__ == "__main__":
    main()
