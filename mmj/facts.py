#!/usr/bin/env python3
"""程序算好的局面事实（与动作无关），作为文字附在指令后面：数字由程序给，模型只负责取舍和表达。

和 tools/build_explanations.py 的参考理由用同一套计算（向聴・受け入れ、安全度、順位と点差、鳴き前後の向聴），
但列出所有候选，不透露选了哪个——推理时模型还没选动作，事实里不能有答案。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mmj/facts.py     # 生成 work/mmj/{train,test}_facts.jsonl
"""
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, "/mnt/d/mahjong-llm/tools")
from build_explanations import (CALL_USE, DORA_NEXT, KINDS, act_name, after_discard, fields, k34, name,  # noqa: E402
                                safety, shanten, situation, to34, ukeire_tiles)

SFT = os.path.expanduser("~/mahjong-llm/work/sft")
OUT = os.path.expanduser("~/mahjong-llm/work/mmj")
SPLITS = {"train": "riichi_100k_mortal_reason.jsonl", "test": "riichi_test5k.jsonl"}   # 与 build_state_dataset.py 行对齐
CALL_NAME = {"pon": "ポン", "chi_low": "チー(下)", "chi_mid": "チー(嵌)", "chi_high": "チー(上)"}


def call_after(f, hand, menzen, call):
    """鳴いた後の向聴（build_explanations.explain 的鸣牌分支，逐行照搬）。不能鸣返回 None。"""
    t = k34(f["last kawa tile"])
    a = hand[:]
    if call == "pon":
        ok = a[t] >= 2
        a[t] -= 2
    else:
        use = [t + d for d in CALL_USE[call]]
        ok = t < 27 and all(0 <= u < 27 and u // 9 == t // 9 and a[u] >= 1 for u in use)
        for u in use:
            a[u] -= 1
    if not ok:
        return None
    after = min(shanten([x - (j == i) for j, x in enumerate(a)], False) for i in range(34) if a[i])
    yakuhai = call == "pon" and KINDS[t] in ("P", "F", "C", f["jikaze"], f["bakaze"])
    tanyao = all((k % 9) not in (0, 8) and k < 27 for k in range(34) for _ in range(hand[k])) and \
        all(len(m) and all(k34(x) < 27 and k34(x) % 9 not in (0, 8) for x in m)
            for m in f["fuuro overview"][0]) and t < 27 and t % 9 not in (0, 8)
    return after, yakuhai, tanyao


def facts_text(prompt):
    f = fields(prompt)
    menzen, seen = f["is menzen"], to34(f["tiles seen"])
    hand = to34(f["my hands"])
    legal = f["Legal actions"]
    sit, riichi, _ = situation(f)
    dora = sorted({DORA_NEXT[d.rstrip("r")[:2] if d[0].isdigit() else d] for d in f["dora indicators"]})
    s = list(sit) + ["ドラ" + "・".join(name(d) for d in dora), "門前" if menzen else "副露あり"]

    discards = [a.split(":")[1] for a in legal if a.startswith("dahai:")]
    if discards:
        opts = sorted(((after_discard(hand, t, seen, menzen), t) for t in discards), key=lambda x: (x[0][0], -x[0][1]))
        items = []
        for (sh, uk), t in opts:          # 牌効率順：先頭が効率最大
            it = f"{name(t)}切り{'テンパイ' if sh == 0 else f'{sh}向聴'}{uk}枚"
            if riichi:
                it += "・" + safety(t, f, riichi)[1]
            items.append(it)
        s.append("打牌候補（効率順）：" + "／".join(items))
    if "reach" in legal:
        tenpai = []
        for t in discards:
            a = hand[:]
            a[k34(t)] -= 1
            if shanten(a, True) == 0:
                w = ukeire_tiles(a, True, seen)
                tenpai.append((sum(n for _, n in w), t, w))
        if tenpai:
            n, t, w = max(tenpai)
            s.append(f"リーチ候補：{name(t)}切りで{''.join(name(x) for x, _ in w)}待ち残り{n}枚")
    calls = [c for c in ("pon", "chi_low", "chi_mid", "chi_high") if c in legal]
    if calls:
        before = shanten(hand, menzen)
        items = []
        for c in calls:
            r = call_after(f, hand, menzen, c)
            if r:
                after, yakuhai, tanyao = r
                items.append(f"{CALL_NAME[c]}で{before}→{after}向聴" + ("（役牌）" if yakuhai else "（断么九形）" if tanyao else ""))
        if items:
            s.append(f"鳴き候補（{name(f['last kawa tile'])}）：" + "／".join(items))
    return "。".join(s) + "。"


def work(line):
    return facts_text(json.loads(line)["instruction"])


def main():
    for split, fn in SPLITS.items():
        lines = open(f"{SFT}/{fn}", encoding="utf-8").readlines()
        with mp.Pool(8) as pool:
            res = pool.map(work, lines, chunksize=500)
        with open(f"{OUT}/{split}_facts.jsonl", "w", encoding="utf-8") as fo:
            for r in res:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        lens = sorted(len(r) for r in res)
        print(f"[{split}] {len(res)} 条；事实长度中位数 {lens[len(lens) // 2]} 字，95% 分位 {lens[int(len(lens) * .95)]} 字")
        print("  例：", res[0])
        print("  例：", res[3])


if __name__ == "__main__":
    main()
