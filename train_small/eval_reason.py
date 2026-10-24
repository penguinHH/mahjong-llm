#!/usr/bin/env python3
"""理由付き模型的评测：动作一致率 + 理由的事实正确率。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/train_small/eval_reason.py \
        ~/mahjong-llm/work/pred/qwen35-0.8b-mortal-reason/generated_predictions.jsonl

做法：对模型选的那个动作，用 tools/build_explanations.py 的同一套计算重新生成"参考理由"
（用测试集的 Mortal Q 值），再把模型写的理由按"。"切成句子逐句对照：
  - 完全一致率：整段理由和参考理由一字不差
  - 句子精确率：模型写的句子里有多少条出现在参考理由里（= 写出的事实有多少是对的）
  - 句子召回率：参考理由的句子里有多少条模型写出来了
  - 按句子类型（牌効率 / 安全度 / 状況 / Mortal 評価 / 鳴き / リーチ）分别统计精确率
模型写的理由和它自己选的动作对不上（比如理由说切 3p、动作却是 5m），会体现在精确率里。
"""
import collections
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, "/mnt/d/mahjong-llm/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_explanations import explain  # noqa: E402
from eval_acc import parse  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work/sft")


def kind(sent):
    if "次点" in sent:
        return "Mortal評価"
    if re.search(r"に(現物|筋|片筋|無筋|字牌)", sent):
        return "安全度"
    if "向聴" in sent and ("切り" in sent) or "テンパイ・" in sent:
        return "牌効率"
    if "リーチ" in sent and "待ち" in sent or "ダマ" in sent:
        return "リーチ"
    if "ポン" in sent or "チー" in sent:
        return "鳴き"
    if "位" in sent or sent.endswith("リーチ"):
        return "状況"
    return "その他"


def split(reason):
    return [s for s in reason.split("。") if s]


def main(path):
    src = [json.loads(l) for l in open(f"{W}/riichi_test5k.jsonl", encoding="utf-8")]
    meta = {m["md5"]: m for m in map(json.loads, open(f"{W}/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    preds = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(preds) == len(src)
    st = collections.Counter()
    by = collections.defaultdict(lambda: [0, 0])
    bad = []
    for r, p in zip(src, preds):
        m = meta.get(hashlib.md5(r["instruction"].encode()).hexdigest())
        act = parse(p["predict"])
        st["n"] += 1
        st["fmt"] += act is not None
        if m is None or act is None:
            continue
        st["human"] += act == m["human"]
        st["mortal"] += act == m["mortal"]
        mm = re.search(r"\{.*?\}", p["predict"], re.S)
        try:
            reason = json.loads(mm.group(0)).get("reason", "")
        except Exception:
            reason = ""
        if act not in m["q"]:
            st["illegal"] += 1
            continue
        ref, _ = explain(r["instruction"], act, m["q"])
        st["with_reason"] += bool(reason)
        st["exact"] += reason == ref
        ms, rs = split(reason), set(split(ref))
        st["sent_pred"] += len(ms)
        st["sent_ok"] += sum(s in rs for s in ms)
        st["sent_ref"] += len(rs)
        st["sent_hit"] += sum(s in set(ms) for s in rs)
        for s in ms:
            k = kind(s)
            by[k][0] += s in rs
            by[k][1] += 1
            if s not in rs and len(bad) < 8:
                bad.append((s, ref))
    n = st["n"]
    print(f"样本 {n}  格式正确 {st['fmt'] / n:.2%}  与牌谱一致 {st['human'] / n:.2%}  与 Mortal 一致 {st['mortal'] / n:.2%}"
          f"  非法动作 {st['illegal']}")
    k = st["n"] - st["illegal"]
    print(f"有理由 {st['with_reason'] / k:.2%}  理由完全一致 {st['exact'] / k:.2%}  "
          f"句子精确率 {st['sent_ok'] / max(1, st['sent_pred']):.2%}（{st['sent_pred']} 句）  "
          f"句子召回率 {st['sent_hit'] / max(1, st['sent_ref']):.2%}")
    print("按句子类型的精确率：" + "，".join(f"{t} {ok / c:.1%}（{c}）" for t, (ok, c) in sorted(by.items(), key=lambda kv: -kv[1][1])))
    print("写错的句子举例（模型句 → 参考理由）：")
    for s, ref in bad:
        print(f"  {s}\n      → {ref}")


if __name__ == "__main__":
    main(sys.argv[1])
