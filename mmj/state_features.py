"""第 1 步：把论文 prompt 里的 36 项局面信息转成张量（多模态麻将模型的"局面模态"输入）。

信息来源只用 prompt 本身（和纯文字版完全相同的信息量，比较才公平），不用 libriichi / Mortal 的编码。
输出两部分：
  planes  uint8 [N_PLANES, 34]：按 34 种牌排列的平面（手牌、见过的牌、四家牌河、副露……），给 1D 卷积用
  scalars float32 [N_SCALARS]：场况标量（场风、点数、巡目、立直状态、合法的非打牌动作……），已缩放到大致 [-1, 1]
所有"四家"的列表都是相对座位：下标 0 = 自己，1 = 下家，2 = 対面，3 = 上家（与 prompt 相同）。
"""
import json
import re

import numpy as np

KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + list("ESWNPFC")
IDX = {k: i for i, k in enumerate(KINDS)}
WINDS = "ESWN"
DORA_NEXT = {**{f"{n}{s}": f"{n % 9 + 1}{s}" for s in "mps" for n in range(1, 10)},
             "E": "S", "S": "W", "W": "N", "N": "E", "P": "F", "F": "C", "C": "P"}
FIELD = re.compile(r"\n\n\d+\. ([^\n]+):\n(.*?)(?=\n\n\d+\. |\n\nPlease tell me)", re.S)
CALLS = ["reach", "chi_low", "chi_mid", "chi_high", "pon", "kan", "hora", "ryukyoku", "pass"]


def base(tile):
    """'5mr' / '6m|' / '7s^|' / '1p!' → '5m' / '6m' / '7s' / '1p'。"""
    t = tile.rstrip("|^!")
    t = t.rstrip("|^!")
    return t[:2] if t[:1].isdigit() else t[:1]


def k34(tile):
    return IDX[base(tile)]


def fields(prompt):
    return {k: json.loads(v) for k, v in FIELD.findall(prompt)}


# ---------------------------------------------------------------- 平面
PLANES = (
    [f"hand>={c}" for c in (1, 2, 3, 4)]
    + ["hand_aka", "wait", "dora_indicator", "dora"]
    + [f"seen>={c}" for c in (1, 2, 3, 4)]
    + ["ankan_cand", "kakan_cand"]
    + [f"kawa{p}_{x}" for p in range(4) for x in ("any", ">=2", "early6", "last3", "last")]
    + [f"fuuro{p}_{x}" for p in range(4) for x in ("any", ">=2")]
    + [f"ankan{p}" for p in range(4)]
    + [f"tedashi{p}" for p in range(4)]
    + [f"riichi_tile{p}" for p in range(4)]
    + ["last_tsumo", "last_kawa", "legal_dahai", "legal_dahai_aka"]
)
P = {n: i for i, n in enumerate(PLANES)}
N_PLANES = len(PLANES)

# ---------------------------------------------------------------- 标量
SCALARS = (
    [f"seat{i}" for i in range(4)] + [f"bakaze_{w}" for w in WINDS] + [f"jikaze_{w}" for w in WINDS]
    + [f"kyoku{i}" for i in range(4)] + ["honba", "kyotaku", "oya"]
    + [f"score{p}" for p in range(4)] + [f"diff{p}" for p in (1, 2, 3)] + [f"rank{i}" for i in range(4)]
    + ["turn", "tiles_left"] + [f"shanten{i}" for i in range(7)]
    + [f"dora_owned{p}" for p in range(4)] + ["aka_hand_m", "aka_hand_p", "aka_hand_s", "doras_seen"]
    + ["aka_seen_m", "aka_seen_p", "aka_seen_s"]
    + [f"riichi_declared{p}" for p in range(4)] + [f"riichi_accepted{p}" for p in range(4)]
    + ["can_riichi", "is_riichi", "furiten", "menzen"]
    + [f"legal_{c}" for c in CALLS]
)
S = {n: i for i, n in enumerate(SCALARS)}
N_SCALARS = len(SCALARS)

USED_FIELDS = {
    "Your identifier", "bakaze", "jikaze", "kyoku", "honba", "kyotaku", "oya", "Scores", "Your rank", "at turn",
    "title left", "shanten", "my hands", "wait tiles", "dora indicators", "dora owned", "akas in your hand",
    "doras seen", "akas seen", "tiles seen", "ankan candidates", "kakan candidates", "kawa overview",
    "fuuro overview", "ankan overview", "last tedashis", "riichi sutehais", "last self tsumo", "last kawa tile",
    "riichi declared", "riichi accepted", "can riichi", "is riichi", "at furiten", "is menzen", "Legal actions"}


def _counts(tiles):
    a = np.zeros(34, np.int16)
    for t in tiles:
        if t:
            a[k34(t)] += 1
    return a


def encode(f):
    """f = fields(prompt) → (planes uint8 [N_PLANES, 34], scalars float32 [N_SCALARS])。"""
    missing = USED_FIELDS - set(f)
    assert not missing, f"prompt 缺字段 {missing}"
    pl = np.zeros((N_PLANES, 34), np.uint8)
    sc = np.zeros(N_SCALARS, np.float32)

    def thresholds(name, cnt):
        for c in (1, 2, 3, 4):
            pl[P[f"{name}>={c}"]] = cnt >= c

    hand = _counts(f["my hands"])
    thresholds("hand", hand)
    for t in f["my hands"]:
        if t.endswith("r"):
            pl[P["hand_aka"], k34(t)] = 1
    pl[P["wait"]] = _counts(f["wait tiles"]) > 0
    for d in f["dora indicators"]:
        pl[P["dora_indicator"], k34(d)] += 1
        pl[P["dora"], IDX[DORA_NEXT[base(d)]]] += 1
    thresholds("seen", _counts(f["tiles seen"]))
    pl[P["ankan_cand"]] = _counts(f["ankan candidates"]) > 0
    pl[P["kakan_cand"]] = _counts(f["kakan candidates"]) > 0

    for p, kawa in enumerate(f["kawa overview"]):
        kawa = [t for t in kawa if t]
        cnt = _counts(kawa)
        pl[P[f"kawa{p}_any"]] = cnt >= 1
        pl[P[f"kawa{p}_>=2"]] = cnt >= 2
        pl[P[f"kawa{p}_early6"]] = _counts(kawa[:6]) > 0
        pl[P[f"kawa{p}_last3"]] = _counts(kawa[-3:]) > 0
        if kawa:
            pl[P[f"kawa{p}_last"], k34(kawa[-1])] = 1
    for p, melds in enumerate(f["fuuro overview"]):
        cnt = _counts([t for m in melds for t in m])
        pl[P[f"fuuro{p}_any"]] = cnt >= 1
        pl[P[f"fuuro{p}_>=2"]] = cnt >= 2
    for p, kans in enumerate(f["ankan overview"]):
        pl[P[f"ankan{p}"]] = _counts(kans) > 0
    for p, t in enumerate(f["last tedashis"]):
        if t:
            pl[P[f"tedashi{p}"], k34(t)] = 1
    for p, t in enumerate(f["riichi sutehais"]):
        if t:
            pl[P[f"riichi_tile{p}"], k34(t)] = 1
    if f["last self tsumo"]:
        pl[P["last_tsumo"], k34(f["last self tsumo"])] = 1
    if f["last kawa tile"]:
        pl[P["last_kawa"], k34(f["last kawa tile"])] = 1
    for a in f["Legal actions"]:
        if a.startswith("dahai:"):
            t = a.split(":", 1)[1].strip()
            pl[P["legal_dahai_aka" if t.endswith("r") else "legal_dahai"], k34(t)] = 1
        else:
            sc[S[f"legal_{a}"]] = 1

    sc[S[f"seat{f['Your identifier']}"]] = 1
    sc[S[f"bakaze_{f['bakaze']}"]] = 1
    sc[S[f"jikaze_{f['jikaze']}"]] = 1
    sc[S[f"kyoku{min(f['kyoku'], 3)}"]] = 1
    sc[S["honba"]] = min(f["honba"], 8) / 4
    sc[S["kyotaku"]] = min(f["kyotaku"], 8) / 4
    sc[S["oya"]] = f["oya"]
    scores = f["Scores"]
    for p in range(4):
        sc[S[f"score{p}"]] = scores[p] / 50000
    for p in (1, 2, 3):
        sc[S[f"diff{p}"]] = np.clip((scores[0] - scores[p]) / 20000, -2, 2)
    sc[S[f"rank{min(f['Your rank'], 3)}"]] = 1
    sc[S["turn"]] = f["at turn"] / 20
    sc[S["tiles_left"]] = f["title left"] / 70
    sc[S[f"shanten{min(max(f['shanten'], 0), 6)}"]] = 1
    for p in range(4):
        sc[S[f"dora_owned{p}"]] = f["dora owned"][p] / 4
    for i, s in enumerate("mps"):
        sc[S[f"aka_hand_{s}"]] = f["akas in your hand"][i]
        sc[S[f"aka_seen_{s}"]] = f["akas seen"][i]
    sc[S["doras_seen"]] = f["doras seen"] / 4
    for p in range(4):
        sc[S[f"riichi_declared{p}"]] = f["riichi declared"][p]
        sc[S[f"riichi_accepted{p}"]] = f["riichi accepted"][p]
    sc[S["can_riichi"]] = f["can riichi"]
    sc[S["is_riichi"]] = f["is riichi"]
    sc[S["furiten"]] = f["at furiten"]
    sc[S["menzen"]] = f["is menzen"]
    return pl, sc


def short_instruction(f):
    """文字侧只剩一句指令 + 合法动作（局面本身由局面 token 提供）。"""
    return ("以下の局面で最善の行動を選び、理由とともにJSONで答えてください。合法手: "
            + json.dumps(f["Legal actions"], ensure_ascii=False))


def check(f, pl, sc):
    """自检：从张量反推关键信息，和 prompt 原文逐项对照。返回不一致的项目列表。"""
    bad = []
    hand = _counts(f["my hands"])
    rec = sum((pl[P[f"hand>={c}"]].astype(int) for c in (1, 2, 3, 4)))
    if not np.array_equal(rec, hand):
        bad.append("hand")
    seen = _counts(f["tiles seen"])
    if not np.array_equal(sum(pl[P[f"seen>={c}"]].astype(int) for c in (1, 2, 3, 4)), np.minimum(seen, 4)):
        bad.append("seen")
    for p, kawa in enumerate(f["kawa overview"]):
        if not np.array_equal(pl[P[f"kawa{p}_any"]] > 0, _counts(kawa) > 0):
            bad.append(f"kawa{p}")
    legal = {a.split(":", 1)[1].strip() for a in f["Legal actions"] if a.startswith("dahai:")}
    rec_legal = {KINDS[i] for i in np.flatnonzero(pl[P["legal_dahai"]])} | \
                {KINDS[i] + "r" for i in np.flatnonzero(pl[P["legal_dahai_aka"]])}
    if rec_legal != legal:
        bad.append("legal_dahai")
    calls = {a for a in f["Legal actions"] if not a.startswith("dahai:")}
    if calls != {c for c in CALLS if sc[S[f"legal_{c}"]]}:
        bad.append("legal_calls")
    if sc[S[f"seat{f['Your identifier']}"]] != 1 or sc[S["menzen"]] != f["is menzen"]:
        bad.append("scalars")
    return bad
