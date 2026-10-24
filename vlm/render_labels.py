#!/usr/bin/env python3
"""给 VLM 渲染图生成逐图块的稠密标签：每个 16×16 图块"是哪张牌、属于哪个区域"（感知模块的辅助监督）。

做法：复用 render_board.render 的布局，只把 tile_img 换成"一块纯色"（颜色编码这是第几次画牌）再渲染一遍。
纯色牌和真牌的尺寸、旋转、位置完全相同，按像素颜色就能找回每张牌的外框，不用手算几何。
画牌的顺序与 render() 一致：宝牌指示牌 → 四家牌河 → 四家副露 → 手牌 → 摸牌，据此给每张牌配上区域。
落到网格：图块中心落在某张牌的框里，这个图块就标成那张牌（1024×768 的图正好是 48×64 个图块，处理器不缩放）。

输出 work/vlm/<--outdir，默认 dense2>/{split}_kind.npy、_region.npy、_center.npy（第一版只有 kind/region，在 dense/）：uint8 [N, 48, 64]，与 work/vlm/img/{split}/<行号>.png 行对齐。
  kind：0-33 = KINDS 顺序的 34 种牌，34-36 = 赤 5m/5p/5s，37 = 背景
  region：0 背景，1 手牌，2 摸牌，3-6 四家牌河（自家/下家/対面/上家），7-10 四家副露，11 宝牌指示牌
  center：每张牌框中心所在的那一格标牌种（同 kind 的编号），其余为背景——数"有几张"用（相邻同种牌在 kind 图上连成一片）

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/vlm/render_labels.py            # 全部 10.5 万张
    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/vlm/render_labels.py --check 0   # 只画一张核对图
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_board as rb  # noqa: E402

SFT = os.path.expanduser("~/mahjong-llm/work/sft")
OUT = os.path.expanduser("~/mahjong-llm/work/vlm")
KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + list("ESWNPFC")
KIND = {k: i for i, k in enumerate(KINDS)} | {"5mr": 34, "5pr": 35, "5sr": 36}
BG_KIND = 37
REGIONS = ["bg", "hand", "tsumo"] + [f"kawa{p}" for p in range(4)] + [f"fuuro{p}" for p in range(4)] + ["dora"]
REGION = {r: i for i, r in enumerate(REGIONS)}
GH, GW, PATCH = 48, 64, 16
MARK_G, MARK_B = 3, 251            # 纯色牌的 G、B 分量（画面上别的颜色都不会是这两个值），R = 序号 + 1

_counter = [0]


def kind_of(t):
    t = t.rstrip("|^!").rstrip("|^!")
    return KIND[t] if t in KIND else KIND[rb.base(t)]


def tiles_in_order(f):
    """与 render() 画牌顺序一致的 (牌, 区域) 列表。"""
    seq = [(t, "dora") for t in f["dora indicators"]]
    for p in range(4):
        seq += [(t, f"kawa{p}") for t in f["kawa overview"][p] if t]
    for p in range(4):
        melds = [m for m in f["fuuro overview"][p]] + [[t] * 4 for t in f["ankan overview"][p]]
        seq += [(t, f"fuuro{p}") for m in melds for t in m if t]
    hand = list(f["my hands"])
    tsumo = f.get("last self tsumo")
    drawn = None
    if len(hand) % 3 == 2 and tsumo and tsumo in hand:
        hand.remove(tsumo)
        drawn = tsumo
    seq += [(t, "hand") for t in hand]
    if drawn:
        seq.append((drawn, "tsumo"))
    return seq


def solid_tile(t, h, sideways=False):
    """替身：与 rb.tile_img 同尺寸的纯色块，R 分量编码这是第几张牌。"""
    from PIL import Image
    _counter[0] += 1
    im = Image.new("RGBA", (int(h * 0.74), h), (_counter[0], MARK_G, MARK_B, 255))
    return im.rotate(90, expand=True) if sideways else im


def tile_boxes(prompt):
    """每张牌的外框：[(x0, y0, x1, y1, 牌, 区域名)]，按 render() 的画牌顺序。"""
    f = rb.fields(prompt)
    seq = tiles_in_order(f)
    _counter[0] = 0
    orig = rb.tile_img
    rb.tile_img = solid_tile
    try:
        img = np.asarray(rb.render(f))
    finally:
        rb.tile_img = orig
    assert _counter[0] == len(seq), f"画了 {_counter[0]} 张牌，顺序表有 {len(seq)} 张"
    assert len(seq) < 255
    m = (img[..., 1] == MARK_G) & (img[..., 2] == MARK_B)
    ys, xs = np.nonzero(m)
    ids = img[ys, xs, 0].astype(np.int32) - 1
    if not len(ids):
        return []
    order = np.argsort(ids, kind="stable")
    ids, ys, xs = ids[order], ys[order], xs[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    out = []
    for s, e in zip(starts, np.r_[starts[1:], len(ids)]):
        t, r = seq[ids[s]]
        out.append((int(xs[s:e].min()), int(ys[s:e].min()), int(xs[s:e].max()), int(ys[s:e].max()), t, r))
    return out


def labels_for(prompt):
    """逐图块标签：kind / region（图块中心落在牌框里），center（牌框中心所在的那一格标牌种，其余为背景）。"""
    kind = np.full((GH, GW), BG_KIND, np.uint8)
    region = np.zeros((GH, GW), np.uint8)
    center = np.full((GH, GW), BG_KIND, np.uint8)
    centers = np.arange(GH) * PATCH + PATCH // 2, np.arange(GW) * PATCH + PATCH // 2
    for x0, y0, x1, y1, t, r in tile_boxes(prompt):
        ri = (centers[0] >= y0) & (centers[0] <= y1)
        ci = (centers[1] >= x0) & (centers[1] <= x1)
        kind[np.ix_(ri, ci)] = kind_of(t)
        region[np.ix_(ri, ci)] = REGION[r]
        center[min(GH - 1, (y0 + y1) // 2 // PATCH), min(GW - 1, (x0 + x1) // 2 // PATCH)] = kind_of(t)
    return kind, region, center


def work(prompt):
    return labels_for(prompt)


def check(i):
    """第 i 张测试图：真实渲染 + 图块标签着色，存成 dense_check_<i>.png 用来核对位置。"""
    from PIL import Image, ImageDraw
    prompt = json.loads(open(f"{SFT}/riichi_test5k.jsonl", encoding="utf-8").readlines()[i])["instruction"]
    kind, region, _ = labels_for(prompt)
    im = Image.open(f"{OUT}/img/test/{i:06d}.png").convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    pal = [(0, 0, 0, 0), (255, 0, 0, 90), (255, 128, 0, 90)] + [(0, 0, 255, 90)] * 4 + [(160, 0, 255, 90)] * 4 + [(0, 200, 0, 110)]
    for a in range(GH):
        for b in range(GW):
            if region[a, b]:
                d.rectangle([b * PATCH, a * PATCH, b * PATCH + PATCH - 1, a * PATCH + PATCH - 1], fill=pal[region[a, b]])
    out = f"{OUT}/dense_check_{i}.png"
    im.save(out)
    seq = tiles_in_order(rb.fields(prompt))
    print(f"第 {i} 张：{len(seq)} 张牌，标成牌的图块 {int((region > 0).sum())} 个 → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=None, help="只为第 N 张测试图画核对图")
    ap.add_argument("--outdir", default="dense2", help="输出到 work/vlm/<此目录>（不要覆盖正在被训练读取的目录）")
    args = ap.parse_args()
    if args.check is not None:
        check(args.check)
        return
    d = f"{OUT}/{args.outdir}"
    os.makedirs(d, exist_ok=True)
    for split, fn in (("test", "riichi_test5k.jsonl"), ("train", "riichi_100k.jsonl")):
        prompts = [json.loads(l)["instruction"] for l in open(f"{SFT}/{fn}", encoding="utf-8")]
        mm = {n: np.lib.format.open_memmap(f"{d}/{split}_{n}.npy", "w+", np.uint8, (len(prompts), GH, GW))
              for n in ("kind", "region", "center")}
        with mp.Pool(6) as pool:
            for i, (k, r, c) in enumerate(pool.imap(work, prompts, chunksize=100)):
                mm["kind"][i], mm["region"][i], mm["center"][i] = k, r, c
                if (i + 1) % 20000 == 0:
                    print(f"  [{split}] {i + 1}/{len(prompts)}", flush=True)
        for a in mm.values():
            a.flush()
        region = mm["region"]
        tile = region > 0
        print(f"[{split}] {len(prompts)} 张；平均每张 {tile.sum() / len(prompts):.0f} 个图块是牌", flush=True)


if __name__ == "__main__":
    main()
