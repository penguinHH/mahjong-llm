#!/usr/bin/env python3
"""打错的牌是"两张都合理"还是"真的打错"？—— 对 dahai 样本做牌效对比。

    ~/mahjong-llm/venv-train/bin/python /mnt/d/mahjong-llm/train_small/analyze_discards.py \
        ~/mahjong-llm/work/pred/qwen35-0.8b/generated_predictions.jsonl

对每条"玩家实际打牌"的样本，分别算玩家打的牌和模型打的牌打出后的向听数和有效进张
（有效进张 = 能让向听数下降的牌种，按 4 - tiles seen 计剩余枚数）。
另外按"是否有对手立直"分开统计，并看立直局面下打的是不是现物（立直者牌河里有的牌）。
需要 `mahjong` 库（uv pip install mahjong）。prompt 里的列表都以自己为下标 0（相对座位）。
"""
import collections
import json
import os
import re
import sys
from pathlib import Path

from mahjong.shanten import Shanten

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_acc import parse  # noqa: E402

KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + list("ESWNPFC")
IDX = {k: i for i, k in enumerate(KINDS)}
FIELD = re.compile(r"\n\n\d+\. ([^\n]+):\n(.*?)(?=\n\n\d+\. |\n\nPlease tell me)", re.S)
SH = Shanten()


def k34(tile):
    return IDX[tile[:2] if tile.endswith("r") else tile]      # 5mr → 5m


def fields(prompt):
    return {k: json.loads(v) for k, v in FIELD.findall(prompt)}


def to34(tiles):
    a = [0] * 34
    for t in tiles:
        a[k34(t)] += 1
    return a


def shanten(a, menzen):
    return SH.calculate_shanten(a, use_chiitoitsu=menzen, use_kokushi=menzen)


def after_discard(hand, tile, seen, menzen):
    """打出 tile 后的 (向听数, 有效进张枚数)。"""
    a = hand[:]
    a[k34(tile)] -= 1
    sh = shanten(a, menzen)
    uke = 0
    for t in range(34):
        if a[t] >= 4:
            continue
        a[t] += 1
        if shanten(a, menzen) < sh:
            uke += max(0, 4 - seen[t])
        a[t] -= 1
    return sh, uke


def bucket(p, m):
    """玩家/模型打牌后的 (向听, 进张) → 类别。"""
    (ps, pu), (ms, mu) = p, m
    if ms > ps:
        return "模型向听更差（退了向听）"
    if ms < ps:
        return "玩家主动退向听（模型没退）"
    if mu >= pu:
        return "同向听，模型进张 ≥ 玩家"
    if mu >= 0.75 * pu:
        return "同向听，模型进张 75-100%"
    return "同向听，模型进张 < 75%"


def pct(a, b):
    return f"{a / b:6.1%}" if b else "   -  "


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    stat = collections.Counter()
    cats = {k: collections.Counter() for k in ("all", "riichi", "calm")}
    by_sh = collections.defaultdict(lambda: [0, 0])
    eff = collections.Counter()
    genbutsu = collections.Counter()
    worst = []
    # 多模态 / VLM 的预测里 prompt 没有文本局面：按行号回到 riichi_test5k 的原 prompt（两者逐行对齐）
    full = None
    if not fields(rows[0]["prompt"]) and len(rows) == 5000:
        src = os.path.expanduser("~/mahjong-llm/work/sft/riichi_test5k.jsonl")
        full = [json.loads(line)["instruction"] for line in open(src, encoding="utf-8")]
    for i, r in enumerate(rows):
        gold, pred = parse(r["label"]), parse(r["predict"])
        if not gold or not gold.startswith("dahai:"):
            continue
        f = fields(full[i] if full else r["prompt"])
        hand, seen = to34(f["my hands"]), to34(f["tiles seen"])
        menzen = f["is menzen"]
        legal = [a.split(":")[1] for a in f["Legal actions"] if a.startswith("dahai:")]
        riichi_opp = [i for i in (1, 2, 3) if f["riichi declared"][i]]
        sit = "riichi" if riichi_opp else "calm"
        g = gold.split(":")[1]
        cur_sh = shanten(hand, menzen)
        stat["dahai"] += 1
        stat[sit] += 1
        ok = pred == gold
        by_sh[min(cur_sh, 3)][0] += ok
        by_sh[min(cur_sh, 3)][1] += 1
        stat[sit + "_ok"] += ok

        # 打出后牌效最好的选择（先比向听，再比进张）
        opts = {t: after_discard(hand, t, seen, menzen) for t in legal}
        best = min(opts.values(), key=lambda v: (v[0], -v[1]))
        p = opts.get(g) or after_discard(hand, g, seen, menzen)
        eff["player_best"] += p == best
        if riichi_opp:
            safe = lambda t: all(any(k34(x.rstrip("|")) == k34(t) for x in f["kawa overview"][i] if x)
                                 for i in riichi_opp)
            genbutsu["player"] += safe(g)

        if pred is None or not pred.startswith("dahai:"):
            stat["non_dahai_pred"] += 1
            continue
        m_tile = pred.split(":")[1]
        if m_tile not in legal:
            stat["illegal"] += 1
            continue
        m = opts[m_tile]
        eff["model_best"] += m == best
        eff["model_dahai"] += 1
        if riichi_opp:
            genbutsu["model"] += safe(m_tile)
            genbutsu["n"] += 1
        if ok:
            continue
        if k34(m_tile) == k34(g):
            c = "同一种牌，只差赤/非赤"
        else:
            c = bucket(p, m)
        cats["all"][c] += 1
        cats[sit][c] += 1
        if c.startswith("模型向听更差") and sit == "calm":
            worst.append((f["my hands"], g, p, m_tile, m))

    n = stat["dahai"]
    wrong = sum(cats["all"].values())
    print(f"玩家打牌样本 {n} 条；模型也输出打牌且合法、但打的牌不同：{wrong} 条"
          f"（模型改选立直等非打牌 {stat['non_dahai_pred']}，不合法 {stat['illegal']}）\n")
    print(f"{'打错的牌按牌效分类':<24}{'全部':>12}{'无人立直':>12}{'有对手立直':>12}")
    order = ["同一种牌，只差赤/非赤", "同向听，模型进张 ≥ 玩家", "同向听，模型进张 75-100%",
             "同向听，模型进张 < 75%", "模型向听更差（退了向听）", "玩家主动退向听（模型没退）"]
    tot = {k: sum(v.values()) for k, v in cats.items()}
    for c in order:
        print(f"  {c:<22}" + "".join(f"{cats[k][c]:>6} {pct(cats[k][c], tot[k])}" for k in ("all", "calm", "riichi")))
    print(f"  {'合计':<22}" + "".join(f"{tot[k]:>6} {'':>6} " for k in ("all", "calm", "riichi")))

    print(f"\n一致率：无人立直 {pct(stat['calm_ok'], stat['calm'])}（{stat['calm']} 条），"
          f"有对手立直 {pct(stat['riichi_ok'], stat['riichi'])}（{stat['riichi']} 条）")
    print("按打牌前向听数：" + "，".join(
        f"{'3+' if s == 3 else s} 向听 {pct(h, c)}（{c}）" for s, (h, c) in sorted(by_sh.items())))
    print(f"打出牌效最优（最低向听 + 最多进张）的比例：玩家 {pct(eff['player_best'], n)}，"
          f"模型 {pct(eff['model_best'], eff['model_dahai'])}")
    if genbutsu["n"]:
        print(f"有对手立直时打现物的比例：玩家 {pct(genbutsu['player'], stat['riichi'])}，"
              f"模型 {pct(genbutsu['model'], genbutsu['n'])}")
    if worst:
        print("\n无人立直、模型退了向听的例子：")
        for hand, g, p, mt, m in worst[:5]:
            print(f"  手牌 {' '.join(hand)} | 玩家打 {g} → {p[0]}向听 {p[1]}枚 | 模型打 {mt} → {m[0]}向听 {m[1]}枚")


if __name__ == "__main__":
    main(sys.argv[1])
