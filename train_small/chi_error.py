#!/usr/bin/env python3
"""e2e_v7 的 chi 差距拆解：感知读错了，还是决策错了。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/train_small/chi_error.py [感知目录名]

读五样东西（全部只读）：
  W/mmj/test_planes.npy                      真值平面（与 test_text.jsonl 行对齐）
  W/mm3/<感知目录>/test_pred.npz              感知预测平面（默认 perception_v7）
  W/mmj/test_text.jsonl                      指令 + 牌谱动作
  W/sft/riichi_test5k_mortal_meta.jsonl      Mortal 标签（按 instruction md5 对照）
  W/mm3/e2e_<版本>/generated_predictions.jsonl  e2e 动作（默认 e2e_v7）

对"牌谱动作 = chi"（71 条）与"合法动作含 chi"两个子集分别统计：
  last_kawa 平面全对、手牌平面全对、两者都对（鸣牌上下文完整）
  再做 e2e 动作对错 × 上下文对错的 2×2 归因表。
last_kawa 为空但能鸣牌时，按 e2e_eval.guess_last_kawa 的同款规则从各家牌河最后一张推。
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/d/mahjong-llm/mmj")
from state_features import KINDS, P, PLANES, S, fields  # noqa: E402
from actions import ACTIONS, legal_mask  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work")
PERC = sys.argv[1] if len(sys.argv) > 1 else "perception_v7"
E2E = sys.argv[2] if len(sys.argv) > 2 else "e2e_v7"
E2E_TRUE = sys.argv[3] if len(sys.argv) > 3 else "e2e_true"
norm = lambda a: a.replace(" ", "") if a else a  # noqa: E731

HAND_IDX = [P[f"hand>={c}"] for c in (1, 2, 3, 4)] + [P["hand_aka"]]


def guess_last_kawa(pl, legal):
    """与 e2e_eval.guess_last_kawa 相同：能吃只看上家(3)，否则上家→対面(2)→下家(1)。"""
    lasts = {p: np.flatnonzero(pl[P[f"kawa{p}_last"]]) for p in (3, 2, 1)}
    for p in ([3] if any(a.startswith("chi") for a in legal) else [3, 2, 1]):
        if len(lasts[p]):
            return KINDS[lasts[p][0]]
    return ""


def kawa_tile(pl, legal):
    idx = np.flatnonzero(pl[P["last_kawa"]])
    if len(idx):
        return KINDS[idx[0]]
    if any(a in ("chi_low", "chi_mid", "chi_high", "pon") for a in legal):
        return guess_last_kawa(pl, legal)
    return ""


def act(s):
    try:
        return norm(json.loads(s[s.index("{"):s.rindex("}") + 1])["action"])
    except Exception:
        return None


def main():
    tp = np.load(f"{W}/mmj/test_planes.npy")
    tsc = np.load(f"{W}/mmj/test_scalars.npy")
    z = np.load(f"{W}/mm3/{PERC}/test_pred.npz")
    pp, psc = z["planes"].astype(np.uint8), z["scalars"].astype(np.float32)
    src = [json.loads(l) for l in open(f"{W}/sft/riichi_test5k.jsonl", encoding="utf-8")]  # 完整 prompt，有 Legal actions
    text = [json.loads(l) for l in open(f"{W}/mmj/test_text.jsonl", encoding="utf-8")]
    mor = {r["md5"]: r for r in map(json.loads, open(f"{W}/sft/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    e2e = [act(json.loads(l)["predict"]) for l in open(f"{W}/mm3/{E2E}/generated_predictions.jsonl", encoding="utf-8")]
    assert len(text) == len(tp) == len(pp) == len(e2e) == len(src)
    assert all(act(s["output"]) == act(t["output"]) for s, t in zip(src, text)), "riichi_test5k 与 test_text 行不对齐"

    rows = []
    for i, s in enumerate(src):
        f = fields(s["instruction"])
        legal = [norm(a) for a in f["Legal actions"]]
        m = mor.get(hashlib.md5(s["instruction"].encode()).hexdigest())
        rows.append(dict(i=i, legal=legal, human=act(s["output"]),
                         mortal=norm(m["mortal"]) if m else None,
                         t_kawa=kawa_tile(tp[i], legal), p_kawa=kawa_tile(pp[i], legal),
                         hand_ok=all((tp[i, h] == pp[i, h]).all() for h in HAND_IDX),
                         e2e=e2e[i]))

    for label, sel in (("全部", lambda r: True),
                       ("合法动作含 chi", lambda r: any(a.startswith("chi") for a in r["legal"])),
                       ("牌谱动作 = chi", lambda r: str(r["human"]).startswith("chi"))):
        sub = [r for r in rows if sel(r) and r["mortal"]]
        n = len(sub)
        if not n:
            continue
        kawa_ok = [r["t_kawa"] == r["p_kawa"] for r in sub]
        ctx_ok = [k and r["hand_ok"] for k, r in zip(kawa_ok, sub)]
        e2e_ok = [r["e2e"] == r["human"] for r in sub]
        print(f"\n== {label}（n={n}）==")
        print(f"  last_kawa 对 {sum(kawa_ok) / n:.1%}   手牌全对 {sum(r['hand_ok'] for r in sub) / n:.1%}   "
              f"上下文全对（两者都对） {sum(ctx_ok) / n:.1%}")
        print(f"  e2e 动作 = 牌谱 {sum(e2e_ok) / n:.1%}   e2e 动作 = Mortal {sum(r['e2e'] == r['mortal'] for r in sub) / n:.1%}")
        # 归因表：动作错 × 上下文错
        both = sum(1 for k, e in zip(ctx_ok, e2e_ok) if k and e)
        only_ctx = sum(1 for k, e in zip(ctx_ok, e2e_ok) if k and not e)
        only_act = sum(1 for k, e in zip(ctx_ok, e2e_ok) if not k and e)
        neither = sum(1 for k, e in zip(ctx_ok, e2e_ok) if not k and not e)
        print(f"  2×2（上下文对/错 × 动作对/错）：都对 {both}｜上下文对但动作错（决策错） {only_ctx}｜"
              f"上下文错但动作碰对 {only_act}｜上下文错且动作错 {neither}")
        # 上下文对/错时的动作准确率
        for name, flag in (("上下文全对", True), ("上下文有错", False)):
            s = [e for k, e in zip(ctx_ok, e2e_ok) if k == flag]
            if s:
                print(f"  {name}时 e2e = 牌谱：{sum(s)}/{len(s)} = {sum(s) / len(s):.1%}")
    # 牌谱动作 = chi 的深挖：错误是预测局面引入的，还是真实局面也会错？chi 还在合法手里吗？
    try:
        et = [act(json.loads(l)["predict"]) for l in open(f"{W}/mm3/{E2E_TRUE}/generated_predictions.jsonl", encoding="utf-8")]
    except FileNotFoundError:
        et = None
    if et is not None:
        chi = [r for r in rows if str(r["human"]).startswith("chi") and r["mortal"]]
        v7_ok = [r["e2e"] == r["human"] for r in chi]
        t_ok = [et[r["i"]] == r["human"] for r in chi]
        tt = {(a, b): 0 for a in (True, False) for b in (True, False)}
        for a, b in zip(t_ok, v7_ok):
            tt[(a, b)] += 1
        print(f"\n== 牌谱动作 = chi（n={len(chi)}）：{E2E_TRUE} 对/错 × {E2E} 对/错 ==")
        print(f"  真对·v7对 {tt[(True, True)]}｜真对·v7错（预测局面引入） {tt[(True, False)]}｜"
              f"真错·v7对 {tt[(False, True)]}｜真错·v7错（两者都错） {tt[(False, False)]}")
        # 合法手列表：预测平面（图片上看不见的清零，同 e2e_eval）推导 vs 指令原文
        unseen = [i for i, n in enumerate(PLANES) if n.startswith("tedashi")]
        pz, sz = pp.copy(), psc.copy()
        pz[:, unseen] = 0
        sz[:, S["furiten"]] = 0
        pmask = legal_mask(pz, sz.round())
        tmask = legal_mask(tp, tsc.round())
        inlegal = drop = 0
        for r in chi:
            legal_from_pred = {norm(ACTIONS[k]) for k in np.flatnonzero(pmask[r["i"]])}
            legal_true = set(r["legal"])
            if r["human"] in legal_from_pred:
                inlegal += 1
            if legal_from_pred != legal_true:
                drop += 1
        print(f"  牌谱 chi 动作仍在预测局面的合法手里：{inlegal}/{len(chi)}；预测合法手 ≠ 原文合法手：{drop}/{len(chi)}")
        # v7 答错时它答了什么
        wrong = [r for r, ok in zip(chi, v7_ok) if not ok]
        cnt = {}
        for r in wrong:
            a = r["e2e"] or "格式错"
            cnt[a.split(":")[0] if not a.startswith("chi") else "chi"] = cnt.get(a.split(":")[0] if not a.startswith("chi") else "chi", 0) + 1
        print(f"  v7 答错时选的动作类型：{cnt}；对错与 Mortal 一致 {sum(r['e2e'] == r['mortal'] for r in wrong)}/{len(wrong)}")

    # 举几个读错的例子
    bad = [r for r in rows if str(r["human"]).startswith("chi") and r["mortal"] and r["t_kawa"] != r["p_kawa"]][:5]
    print("\n牌谱动作=chi 且 last_kawa 读错的例子（真值 → 预测｜e2e 动作 vs 牌谱/Mortal）：")
    for r in bad:
        print(f"  #{r['i']}: last_kawa {r['t_kawa'] or '空'} → {r['p_kawa'] or '空'}｜e2e {r['e2e']}｜"
              f"牌谱 {r['human']}｜Mortal {r['mortal']}")


if __name__ == "__main__":
    main()
