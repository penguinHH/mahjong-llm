#!/usr/bin/env python3
"""用 Mortal 标签替换 SFT 子集的答案（蒸馏数据）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/tools/make_distill_sets.py

riichi_100k.jsonl  + distill/mortal_full7000.jsonl → riichi_100k_mortal.jsonl（prompt 不变，答案换成 Mortal 首选）
riichi_test5k.jsonl + distill/mortal_test1000.jsonl → riichi_test5k_mortal.jsonl（行顺序与 riichi_test5k 相同，评测用）
按 instruction 的 md5 拼接。训练文件只含 instruction/output（q 的键每行不同，会让 datasets 的 Arrow 推断失败）；
human（牌谱动作）和 q（合法动作的 Q 值）另存同名 *_meta.jsonl，行顺序一致。
"""
import hashlib
import json
import os

W = os.path.expanduser("~/mahjong-llm/work")
JOBS = [("sft/riichi_100k.jsonl", "distill/mortal_full7000.jsonl", "sft/riichi_100k_mortal.jsonl"),
        ("sft/riichi_test5k.jsonl", "distill/mortal_test1000.jsonl", "sft/riichi_test5k_mortal.jsonl")]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


for src, labels, dst in JOBS:
    rows = [json.loads(l) for l in open(f"{W}/{src}", encoding="utf-8")]
    need = {md5(r["instruction"]) for r in rows}
    lab = {}
    for line in open(f"{W}/{labels}", encoding="utf-8"):
        d = json.loads(line)
        if d["md5"] in need:
            lab[d["md5"]] = d
    n = same = 0
    by = {}
    meta = dst.replace(".jsonl", "_meta.jsonl")
    with open(f"{W}/{dst}.part", "w", encoding="utf-8") as fo, open(f"{W}/{meta}.part", "w", encoding="utf-8") as fm:
        for r in rows:
            d = lab.get(md5(r["instruction"]))
            if d is None:
                continue
            assert json.loads(r["output"])["action"] == d["human"], "牌谱动作对不上"
            fo.write(json.dumps({"instruction": r["instruction"], "output": json.dumps({"action": d["mortal"]})},
                                ensure_ascii=False) + "\n")
            fm.write(json.dumps({"md5": d["md5"], "human": d["human"], "mortal": d["mortal"], "q": d["q"]}) + "\n")
            n += 1
            same += d["mortal"] == d["human"]
            t = d["human"].split(":")[0]
            t = "chi" if t.startswith("chi") else t
            c = by.setdefault(t, [0, 0])
            c[0] += 1
            c[1] += d["mortal"] == d["human"]
    os.replace(f"{W}/{dst}.part", f"{W}/{dst}")
    os.replace(f"{W}/{meta}.part", f"{W}/{meta}")
    print(f"{dst}: {n}/{len(rows)} 条（覆盖 {n / len(rows):.2%}），Mortal 与牌谱一致 {same / n:.2%}，"
          f"即 {1 - same / n:.2%} 的答案被换掉")
    print("  按牌谱动作：" + "，".join(f"{t} {h / c:.1%}（{c}）" for t, (c, h) in sorted(by.items(), key=lambda kv: -kv[1][0])))
