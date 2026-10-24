"""对齐阶段（第 2 步 A）的训练目标：把局面张量"念"成一句日语描述。

LLM 冻住、只训投影层时，让它看着 35 个局面 token 说出手牌、ドラ、立直者、場風/自風、順位、向聴、直前の捨て牌 ——
说得出来，就说明投影层已经把局面翻译成 LLM 读得懂的东西。描述只从张量本身解码（不看 prompt），
合法动作已经写在文字指令里，所以不放进描述（否则 LLM 抄指令就能答对，学不到读局面）。
"""
import numpy as np

from state_features import KINDS, P, S

HONOR = dict(zip("ESWNPFC", "東南西北白發中"))
SEAT = {1: "下家", 2: "対面", 3: "上家"}
WIND = {"E": "東", "S": "南", "W": "西", "N": "北"}


def tname(i, red=False):
    k = KINDS[i]
    return ("赤" + k) if red else HONOR.get(k, k)


def hand_tiles(pl):
    cnt = sum(pl[P[f"hand>={c}"]].astype(int) for c in (1, 2, 3, 4))
    out = []
    for i in range(34):
        for j in range(cnt[i]):
            out.append(tname(i, red=bool(pl[P["hand_aka"], i]) and j == 0))
    return out


def caption(pl, sc):
    bakaze = next(w for w in "ESWN" if sc[S[f"bakaze_{w}"]])
    jikaze = next(w for w in "ESWN" if sc[S[f"jikaze_{w}"]])
    rank = 1 + next(i for i in range(4) if sc[S[f"rank{i}"]])
    shanten = next(i for i in range(7) if sc[S[f"shanten{i}"]])
    dora = [tname(i) for i in np.flatnonzero(pl[P["dora"]]) for _ in range(int(pl[P["dora"], i]))]
    riichi = [SEAT[p] for p in (1, 2, 3) if sc[S[f"riichi_declared{p}"]]]
    last = np.flatnonzero(pl[P["last_kawa"]])
    parts = [
        f"{WIND[bakaze]}場・{WIND[jikaze]}家、{rank}位",
        "手牌 " + " ".join(hand_tiles(pl)),
        "ドラ " + " ".join(dora),
        ("リーチ者 " + "・".join(riichi)) if riichi else "リーチ者なし",
        f"{shanten}向聴" if shanten else "テンパイ",
    ]
    if len(last):
        parts.append(f"直前の捨て牌 {tname(int(last[0]))}")
    return "。".join(parts) + "。"


CAPTION_INSTRUCTION = "この局面を説明してください。"
