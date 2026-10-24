"""三合一感知：把检测到的牌（牌种、区域、中心坐标）组装成牌局字段，再用 state_features.encode 算出与牌有关的平面。

和 YOLO 路线（majsoul_yolo 的 state.py）同一思路：检测只负责"哪里有哪张牌"，牌局结构交给确定性的规则。
平面定义直接复用 mmj/state_features.encode，保证与训练编码器时完全一致；没法从牌的位置得到的字段填默认值，
只取下面 TILE_PLANES 这些平面，其余平面和标量仍用感知模型自己读出的结果。

四家牌河按 render_board 的摆法还原打出顺序（6 张一行，按座位旋转；PIL 的 rotate 是逆时针）：
  自家 0°：行从上往下排，行内从左往右      下家 90°：行从左往右排，行内从下往上
  対面 180°：行从下往上排，行内从右往左    上家 270°：行从右往左排，行内从上往下
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mmj"))   # 放最后：mmj 也有 model.py
from state_features import KINDS, PLANES, encode  # noqa: E402

AKA = {34: "5mr", 35: "5pr", 36: "5sr"}
REGIONS = ["bg", "hand", "tsumo"] + [f"kawa{p}" for p in range(4)] + [f"fuuro{p}" for p in range(4)] + ["dora"]
# 只由"哪些牌、在哪、什么顺序"决定的平面。不含：wait（要算听牌）、ankan_cand / kakan_cand、ankan{p}（暗杠与明杠画法相同）、
# tedashi（静态画面看不到）、riichi_tile{p}（横放的宣言牌没有单独标）、last_kawa（要知道谁刚打）
# last_tsumo 也不在内：等别人出牌（决定鸣不鸣）时，上一张摸的牌不单独摆出来，画面上看不到
TILE_PLANES = ([f"hand>={c}" for c in (1, 2, 3, 4)] + ["hand_aka", "dora_indicator", "dora"]
               + [f"seen>={c}" for c in (1, 2, 3, 4)]
               + [f"kawa{p}_{x}" for p in range(4) for x in ("any", ">=2", "early6", "last3", "last")]
               + [f"fuuro{p}_{x}" for p in range(4) for x in ("any", ">=2")]
               + ["legal_dahai", "legal_dahai_aka"])
TILE_IDX = [PLANES.index(n) for n in TILE_PLANES]

DEFAULTS = {
    "Your identifier": 0, "bakaze": "E", "jikaze": "E", "kyoku": 0, "honba": 0, "kyotaku": 0, "oya": False,
    "Scores": [25000] * 4, "Your rank": 0, "at turn": 0, "title left": 0, "shanten": 0, "wait tiles": [],
    "dora owned": [0] * 4, "akas in your hand": [False] * 3, "doras seen": 0, "akas seen": [False] * 3,
    "ankan candidates": [], "kakan candidates": [], "ankan overview": [[], [], [], []],
    "last tedashis": [""] * 4, "riichi sutehais": [""] * 4, "last kawa tile": "",
    "riichi declared": [False] * 4, "riichi accepted": [False] * 4, "can riichi": False, "is riichi": False,
    "at furiten": False, "is menzen": True,
}


def name(k):
    return AKA.get(int(k), KINDS[int(k)] if int(k) < 34 else None)


def order_key(p):
    """牌河第 p 家：(行坐标, 行内坐标) 的取法。"""
    return [lambda x, y: (y, x), lambda x, y: (x, -y), lambda x, y: (-y, -x), lambda x, y: (-x, y)][p]


def order_river(tiles, p, row_gap=20):
    """tiles: [(x, y, 牌)] → 按打出顺序排好的牌列表。行与行相距约 36 像素，行坐标差超过 row_gap 就算新的一行。"""
    if not tiles:
        return []
    key = order_key(p)
    ks = sorted((key(x, y), t) for x, y, t in tiles)
    rows, cur, last_r = [], [], None
    for (r, w), t in ks:
        if last_r is not None and r - last_r > row_gap:
            rows.append(cur)
            cur = []
        cur.append((w, t))
        last_r = r
    rows.append(cur)
    return [t for row in rows for _, t in sorted(row)]


def base(t):
    return t[:2]


def split_melds(seq):
    """一家的副露按摆放顺序连成一排（render 里组与组之间没有间隔）：4 张同种算杠，否则 3 张一组。"""
    melds, i = [], 0
    while i < len(seq):
        n = 4 if i + 4 <= len(seq) and len({base(t) for t in seq[i:i + 4]}) == 1 else 3
        melds.append(seq[i:i + n])
        i += n
    return melds


def fields_from_tiles(tiles, row_gap=20):
    """tiles: [(x, y, 牌种 0-36, 区域名)] → encode 需要的字段字典（无法从牌得到的填默认值）。"""
    by = {r: [] for r in REGIONS}
    for x, y, k, r in tiles:
        t = name(k)
        if t is not None and r in by:
            by[r].append((x, y, t))
    hand = [t for _, _, t in sorted(by["hand"])]
    tsumo = [t for _, _, t in by["tsumo"]]
    my = hand + tsumo
    kawa = [order_river(by[f"kawa{p}"], p, row_gap) for p in range(4)]
    # 副露按摆放顺序排（与牌河行内方向相同），切成一组组；每组最后一张是被鸣的牌（render 按 fuuro overview 的顺序画）
    within = [lambda x, y: x, lambda x, y: -y, lambda x, y: -x, lambda x, y: y]
    melds = [split_melds([t for x, y, t in sorted(by[f"fuuro{p}"], key=lambda a, p=p: within[p](a[0], a[1]))])
             for p in range(4)]
    fuuro = [[[t for g in m for t in g]] if m else [] for m in melds]
    dora = [t for _, _, t in sorted(by["dora"])]
    # 见过的牌：画面上所有牌，但被鸣的牌在打出者的河里和副露里各画了一次，引擎只算一次 → 每组副露扣掉最后那张
    seen = my + [t for k in kawa for t in k] + [t for m in fuuro for g in m for t in g] + dora
    for m in melds:
        for g in m:
            b = base(g[-1])
            j = next((i for i, t in enumerate(seen) if base(t) == b), None)
            if j is not None:
                seen.pop(j)
    legal = sorted({f"dahai:{t}" for t in my}) if len(my) % 3 == 2 else []
    f = dict(DEFAULTS)
    f.update({"my hands": my, "last self tsumo": tsumo[0] if tsumo else "", "kawa overview": kawa,
              "fuuro overview": fuuro, "dora indicators": dora, "tiles seen": seen, "Legal actions": legal,
              "is menzen": not any(fuuro)})
    return f


def tile_planes(tiles, row_gap=20):
    """tiles → 与牌有关的平面 uint8 [len(TILE_PLANES), 34]（顺序同 TILE_PLANES）。"""
    pl, _ = encode(fields_from_tiles(tiles, row_gap))
    return pl[TILE_IDX]


def tiles_from_dense(center_logit, region_logit, patch=16):
    """感知模型的逐图块输出 → tiles。center_logit [38, H, W]（37 = 不是牌中心），region_logit [12, H, W]。"""
    c = center_logit.argmax(0)
    r = region_logit.argmax(0)
    ys, xs = np.nonzero(c != 37)
    return [(int(x) * patch + patch // 2, int(y) * patch + patch // 2, int(c[y, x]), REGIONS[int(r[y, x])])
            for y, x in zip(ys, xs) if r[y, x] > 0]
