#!/usr/bin/env python3
"""给 Mortal 标签的 SFT 样本配上"有事实依据"的日语理由（程序算事实，模型学取舍）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/tools/build_explanations.py [条数] [--show N]

输入 sft/riichi_100k_mortal.jsonl + riichi_100k_mortal_meta.jsonl（行对齐，make_distill_sets.py 生成）；
输出 sft/riichi_100k_mortal_reason.jsonl：prompt 不变，答案变成 {"reason": "...", "action": "..."}（先理由后动作）；
另存 riichi_100k_mortal_reason_facts.jsonl：每条理由用到的结构化事实，之后用来自动核对模型写出的理由。

理由里的每一句都来自确定性的计算，不让模型（也不让我）编：
  向聴・受け入れ（mahjong 库，受け入れ枚数 = 4 - tiles seen）、对立直者的安全度（現物 / 筋 / 片筋 / 字牌見え枚数 / 無筋）、
  ドラ、順位と点差（Scores，自己是下标 0）、Mortal 对各合法动作的 Q 值差。
不判断"有没有役"之类需要完整和牌判定的事（只写确定的：役牌ポン、全部中张 = 断么九形）。
"""
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, "/mnt/d/mahjong-llm/train_small")
from analyze_discards import IDX, KINDS, after_discard, fields, k34, shanten, to34  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work/sft")
SEAT = {1: "下家", 2: "対面", 3: "上家"}
HONOR = dict(zip("ESWNPFC", "東南西北白發中"))
DORA_NEXT = {**{f"{n}{s}": f"{n % 9 + 1}{s}" for s in "mps" for n in range(1, 10)},
             "E": "S", "S": "W", "W": "N", "N": "E", "P": "F", "F": "C", "C": "P"}
CALL_USE = {"chi_low": (1, 2), "chi_mid": (-1, 1), "chi_high": (-2, -1)}


def name(t):
    """5pr → 赤5p，N → 北，8p → 8p。"""
    if t.endswith("r"):
        return "赤" + t[:2]
    return HONOR.get(t, t)


def act_name(a):
    if a.startswith("dahai:"):
        return name(a.split(":")[1]) + "切り"
    return {"reach": "リーチ", "pon": "ポン", "chi_low": "チー", "chi_mid": "チー", "chi_high": "チー", "kan": "カン",
            "hora": "和了", "pass": "スルー", "ryukyoku": "流局"}.get(a, a)


def safety(t, f, riichi):
    """t 对每个立直者的安全度，返回最危险的那一档（0 現物 … 4 無筋）及说明。"""
    k = k34(t)
    worst = (-1, "")
    for i in riichi:
        kawa = {k34(x.rstrip("|")) for x in f["kawa overview"][i] if x}
        if k in kawa:
            lv, s = 0, "現物"
        elif k >= 27:
            seen = to34(f["tiles seen"])[k]
            lv, s = (1, f"字牌{seen}枚見え") if seen >= 3 else (3, f"字牌{seen}枚見え")
        else:
            n, base = k % 9 + 1, k - k % 9
            lo = n - 3 >= 1 and (base + n - 4) in kawa
            hi = n + 3 <= 9 and (base + n + 2) in kawa
            need_two = 4 <= n <= 6
            if (lo and hi) or (not need_two and (lo or hi)):
                lv, s = 1, "筋"
            elif lo or hi:
                lv, s = 2, "片筋"
            else:
                lv, s = 4, "無筋"
        if lv > worst[0]:
            worst = (lv, f"{SEAT[i]}に{s}")
    return worst


def ukeire_tiles(a, menzen, seen):
    sh = shanten(a, menzen)
    out = []
    for t in range(34):
        if a[t] >= 4:
            continue
        a[t] += 1
        if shanten(a, menzen) < sh:
            out.append((KINDS[t], max(0, 4 - seen[t])))
        a[t] -= 1
    return out


def mortal_gap(action, q):
    others = {a: v for a, v in q.items() if a != action}
    if not others:
        return "", {}
    second = max(others, key=others.get)
    gap = q[action] - others[second]
    # 不写具体分差：推理时模型看不到 Mortal，具体数字只能靠猜。按 10 万条的分布分三档
    # （打牌：<0.5 占 22%，0.5～3 占 54%，≥3 占 24%）
    level = 0 if gap < 0.5 else 1 if gap < 3 else 2
    alt = ("ダマの" if action == "reach" and second.startswith("dahai:") else "") + act_name(second)
    s = [f"次点の{alt}と僅差", f"次点の{alt}より良い", f"次点の{alt}より明確に良い"][level]
    return s, {"second": second, "gap": round(gap, 3), "level": level}


def situation(f):
    sc = f["Scores"]
    rank = 1 + sum(s > sc[0] for s in sc[1:])
    riichi = [i for i in (1, 2, 3) if f["riichi declared"][i]]
    parts = []
    if riichi:
        parts.append("・".join(SEAT[i] for i in riichi) + "リーチ")
    if f["bakaze"] != "E":                       # 南場以降は点差に触れる
        top = max(sc[1:])
        parts.append(f"{rank}位（トップと{sc[0] - top:+d}点）" if rank > 1 else f"トップ（2位と{sc[0] - top:+d}点）")
    return parts, riichi, rank


def explain(prompt, action, q):
    f = fields(prompt)
    menzen, seen = f["is menzen"], to34(f["tiles seen"])
    hand = to34(f["my hands"])
    dora = {DORA_NEXT[d.rstrip("r")[:2] if d[0].isdigit() else d] for d in f["dora indicators"]}
    sit, riichi, rank = situation(f)
    facts = {"action": action, "riichi": riichi, "rank": rank}
    s = list(sit)
    kind = action.split(":")[0]

    if kind == "dahai":
        legal = [a.split(":")[1] for a in f["Legal actions"] if a.startswith("dahai:")]
        opts = {t: after_discard(hand, t, seen, menzen) for t in legal}
        c = action.split(":")[1]
        csh, cuk = opts[c]
        best = min(opts, key=lambda t: (opts[t][0], -opts[t][1]))
        bsh, buk = opts[best]
        facts.update(chosen=[c, csh, cuk], best=[best, bsh, buk])
        state = "テンパイ" if csh == 0 else f"{csh}向聴"
        if (csh, cuk) == (bsh, buk):
            s.append(f"{name(c)}切りで{state}・受け入れ{cuk}枚（牌効率最大）")
        else:
            s.append(f"{name(c)}切りで{state}・受け入れ{cuk}枚（効率最大は{name(best)}切りの{bsh}向聴{buk}枚）")
            if KINDS[k34(best)] in dora and k34(best) != k34(c):      # 牌効率ならドラを切るところ、残した
                s.append(f"ドラの{name(best)}は残す")
        if riichi:
            lv, desc = safety(c, f, riichi)
            facts["safety"] = [lv, desc]
            s.append(f"{name(c)}は{desc}" + ("で安全度優先" if (csh, cuk) != (bsh, buk) and lv <= 1 else ""))
        if "reach" in f["Legal actions"]:
            facts["dama"] = True
            s.append("リーチせずダマ")
    elif kind == "reach":
        legal = [a.split(":")[1] for a in f["Legal actions"] if a.startswith("dahai:")]
        tenpai = []
        for t in legal:
            a = hand[:]
            a[k34(t)] -= 1
            if shanten(a, True) == 0:
                w = ukeire_tiles(a, True, seen)
                tenpai.append((sum(n for _, n in w), t, w))
        if tenpai:
            n, t, w = max(tenpai)
            facts["tenpai"] = [t, [x for x, _ in w], n]
            s.append(f"{name(t)}切りでテンパイ、{''.join(name(x) for x, _ in w)}待ち残り{n}枚。門前なのでリーチ")
        else:
            s.append("テンパイ、リーチ")
    elif kind in ("pon", "chi_low", "chi_mid", "chi_high") or (kind == "pass" and any(
            a in q for a in ("pon", "chi_low", "chi_mid", "chi_high"))):
        call = kind if kind != "pass" else next(a for a in ("pon", "chi_low", "chi_mid", "chi_high") if a in q)
        t = k34(f["last kawa tile"])
        before = shanten(hand, menzen)
        a = hand[:]
        if call == "pon":
            ok = a[t] >= 2
            a[t] -= 2
        else:                                   # チー：同じ色の数牌だけ
            use = [t + d for d in CALL_USE[call]]
            ok = t < 27 and all(0 <= u < 27 and u // 9 == t // 9 and a[u] >= 1 for u in use)
            for u in use:
                a[u] -= 1
        if ok:
            after = min(shanten([x - (j == i) for j, x in enumerate(a)], False) for i in range(34) if a[i])
            facts.update(call=call, before=before, after=after)
            tile = name(f["last kawa tile"])
            yakuhai = call == "pon" and KINDS[t] in ("P", "F", "C", f["jikaze"], f["bakaze"])
            tanyao = all((k % 9) not in (0, 8) and k < 27 for k in range(34) for _ in range(hand[k])) and \
                all(len(m) and all(k34(x)  < 27 and k34(x) % 9 not in (0, 8) for x in m)
                    for m in f["fuuro overview"][0]) and t < 27 and t % 9 not in (0, 8)
            what = f"{tile}を{act_name(call)}"
            if kind != "pass":
                s.append(f"{what}して{before}→{after}向聴" + ("（役牌）" if yakuhai else "（断么九形）" if tanyao else ""))
            elif after >= before:
                s.append(f"{what}しても{after}向聴で前進しないのでスルー")
            else:
                s.append(f"{what}すれば{after}向聴に進むが" + ("門前を維持してスルー" if menzen else "スルー"))
    elif kind == "hora":
        s.append("和了")
    elif kind == "kan":
        s.append("カン")
    elif kind == "ryukyoku":
        s.append("九種九牌で流局")
    elif kind == "pass":
        s.append("見送り")

    g, gf = mortal_gap(action, q)
    if g and kind != "hora":
        s.append(g)
        facts.update(gf)
    return "。".join(s) + "。", facts


def work(args):
    row, meta = args
    try:
        reason, facts = explain(row["instruction"], meta["mortal"], meta["q"])
    except Exception as e:          # 解析不了的少数样本：只给动作
        reason, facts = "", {"error": repr(e)[:200]}
    out = {"instruction": row["instruction"],
           "output": json.dumps({"reason": reason, "action": meta["mortal"]}, ensure_ascii=False)}
    return out, {"md5": meta["md5"], **facts}


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(argv[0]) if argv and argv[0].isdigit() else None
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
    base = "riichi_test5k_mortal" if "--test" in sys.argv else "riichi_100k_mortal"   # 测试集：评测时核对模型写的理由
    rows = [json.loads(l) for l in open(f"{W}/{base}.jsonl", encoding="utf-8")]
    meta = [json.loads(l) for l in open(f"{W}/{base}_meta.jsonl", encoding="utf-8")]
    assert len(rows) == len(meta)
    pairs = list(zip(rows, meta))[:limit]
    with mp.Pool(8) as pool:
        res = pool.map(work, pairs, chunksize=200)
    suffix = "" if limit is None else f"_first{limit}"
    with open(f"{W}/{base}_reason{suffix}.jsonl", "w", encoding="utf-8") as fo, \
            open(f"{W}/{base}_reason{suffix}_facts.jsonl", "w", encoding="utf-8") as ff:
        for out, facts in res:
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            ff.write(json.dumps(facts, ensure_ascii=False) + "\n")
    err = sum("error" in fa for _, fa in res)
    lens = sorted(len(json.loads(o["output"])["reason"]) for o, _ in res)
    print(f"{len(res)} 条，解析失败 {err} 条；理由长度 中位数 {lens[len(lens) // 2]} 字，95% 分位 {lens[int(len(lens) * .95)]} 字")
    for o, fa in res[:show]:
        print(o["output"])


if __name__ == "__main__":
    main()
