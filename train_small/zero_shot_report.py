#!/usr/bin/env python3
"""后训练到底带来了多少：基座模型零样本 vs 规则基线 vs 后训练模型，同一批 5000 条留出样本。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/train_small/zero_shot_report.py \
        名字=~/mahjong-llm/work/pred/qwen35-0.8b-base/generated_predictions.jsonl [名字=更多预测文件...]

eval_vs_mortal.py 只在"解析出来且合法"的预测上算一致率；零样本时格式和合法性本身就是问题，所以这里分开报：
  格式正确（能解析出 {"action": ...}）、合法（在 Legal actions 里）、与牌谱 / Mortal 一致（格式错或不合法都算错）、
  Q 差（只在合法的预测上算，越小越好）、预测的动作类型分布、"总是选列表第一个"的比例（判断是否退化成固定输出）。
不需要模型的参照基线（直接按规则从 Legal actions 里选）：
  随机       合法动作里均匀随机（报期望值，不抽样）
  第一个     总选 Legal actions 的第一个
  跳过/摸切  能 pass 就 pass，否则打刚摸的牌，否则第一个
  牌效贪心   能和就和；能立直就立直；有打牌选项时打"打出后向听最小、进张最多"的牌；只能鸣牌时 pass
--lenient：严格解析失败时，再从文本里找第一个 "action": "..." 字段（JSON 没闭合、外面套了数组、后面跟长 reasoning 都算）
  或第一个用「」括起来的动作串（VLM 基座常写「最善の行動は「dahai:2m」です」），
  用来区分"不会按格式答"和"不会打"。
"""
import collections
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_discards import after_discard, fields, to34  # noqa: E402
from eval_acc import parse  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work/sft")
norm = lambda a: a.replace(" ", "") if a else a  # noqa: E731


def load():
    src = [json.loads(l) for l in open(f"{W}/riichi_test5k.jsonl", encoding="utf-8")]
    mor = {r["md5"]: r for r in map(json.loads, open(f"{W}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    rows = []
    for r in src:
        lab = mor.get(hashlib.md5(r["instruction"].encode()).hexdigest())
        f = fields(r["instruction"])
        rows.append(dict(lab=lab, f=f, legal=[norm(a) for a in f["Legal actions"]],
                         q={norm(k): v for k, v in lab["q"].items()} if lab else None))
    return rows


def kind(a):
    t = a.split(":")[0]
    return "chi" if t.startswith("chi") else t


def score(rows, pick):
    """pick(i, row) → 动作字符串或 None。返回各项指标。"""
    n = fmt = leg = h = m = first = 0
    gaps, types = [], collections.Counter()
    for i, r in enumerate(rows):
        if r["lab"] is None:
            continue
        a = norm(pick(i, r))
        n += 1
        fmt += a is not None
        ok = a in r["legal"]
        leg += ok
        h += a == norm(r["lab"]["human"])
        m += a == norm(r["lab"]["mortal"])
        first += a == r["legal"][0]
        if ok and a in r["q"]:
            gaps.append(max(r["q"].values()) - r["q"][a])
        types[kind(a) if ok else ("格式错" if a is None else "不合法")] += 1
    return dict(n=n, 格式正确=fmt / n, 合法=leg / n, 与牌谱一致=h / n, 与Mortal一致=m / n,
                Q差=sum(gaps) / max(1, len(gaps)), 选第一个=first / n,
                类型={k: round(v / n, 3) for k, v in types.most_common(6)})


def random_expect(rows):
    """均匀随机选合法动作的期望值（不抽样）。"""
    n = h = m = 0
    gaps = []
    for r in rows:
        if r["lab"] is None:
            continue
        L = r["legal"]
        n += 1
        h += (norm(r["lab"]["human"]) in L) / len(L)
        m += (norm(r["lab"]["mortal"]) in L) / len(L)
        gq = [max(r["q"].values()) - r["q"][a] for a in L if a in r["q"]]
        if gq:
            gaps.append(sum(gq) / len(gq))
    return dict(n=n, 格式正确=1.0, 合法=1.0, 与牌谱一致=h / n, 与Mortal一致=m / n, Q差=sum(gaps) / len(gaps))


def pass_or_tsumogiri(i, r):
    L = r["legal"]
    if "pass" in L:
        return "pass"
    t = f"dahai:{r['f'].get('last self tsumo')}"
    return t if t in L else L[0]


def greedy(i, r):
    L, f = r["legal"], r["f"]
    if "hora" in L:
        return "hora"
    if "reach" in L:
        return "reach"
    dahai = [a.split(":")[1] for a in L if a.startswith("dahai:")]
    if not dahai:
        return "pass" if "pass" in L else L[0]
    hand, seen = to34(f["my hands"]), to34(f["tiles seen"])
    best = min(dahai, key=lambda t: (lambda s, u: (s, -u))(*after_discard(hand, t, seen, f["is menzen"])))
    return f"dahai:{best}"


LENIENT = re.compile(r'"action"\s*:\s*"([^"]+)"|「([a-z_]+(?::[0-9a-z]+)?)」')


def parse_lenient(text):
    a = parse(text)
    if a is None:
        m = LENIENT.search(text)
        a = (m.group(1) or m.group(2)).replace(" ", "") if m else None
    return a


def main(args):
    lenient = "--lenient" in args
    args = [a for a in args if a != "--lenient"]
    pp = parse_lenient if lenient else parse
    rows = load()
    res = {"随机（期望）": random_expect(rows),
           "第一个": score(rows, lambda i, r: r["legal"][0]),
           "跳过/摸切": score(rows, pass_or_tsumogiri),
           "牌效贪心": score(rows, greedy)}
    for a in args:
        name, path = a.split("=", 1) if "=" in a else (os.path.basename(os.path.dirname(a)), a)
        preds = [json.loads(l) for l in open(os.path.expanduser(path), encoding="utf-8")]
        assert len(preds) == len(rows), f"{path}: {len(preds)} 行，应为 {len(rows)}"
        res[name] = score(rows, lambda i, r: pp(preds[i]["predict"]))
        res[name]["例"] = preds[0]["predict"][:120]
    keys = ["格式正确", "合法", "与牌谱一致", "与Mortal一致", "Q差", "选第一个"]
    print("| 方法 | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for k, v in res.items():
        print(f"| {k} | " + " | ".join(
            ("—" if x not in v else f"{v[x]:.3f}" if x == "Q差" else f"{v[x]:.2%}") for x in keys) + " |")
    for k, v in res.items():
        if "类型" in v:
            print(f"{k} 动作分布：{v['类型']}" + (f"  例：{v['例']!r}" if "例" in v else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
