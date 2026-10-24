#!/usr/bin/env python3
"""感知结果的规则后处理：能由已读准的部分推出来的量，不靠网络回归，按规则算。

  1. last_kawa（最后打出的那张）：轮到自己时取上家牌河最后一张；要决定鸣不鸣时取各家牌河的最后一张
     （能吃只看上家；碰优先选手里有 2 张的那家；否则上家 → 対面 → 下家）
  2. 吃的种类：画面上只有一个"チー"按钮，分不出吃下 / 吃中 / 吃上，网络只能平均猜（每个约 0.07）。
     改成：三个吃的概率之和 > CHI_TH 视为"能吃"，具体哪种由手牌 + 被吃的牌按规则判断
  3. 向听数：由手牌（已 100% 读准）用 mahjong 库算，门前才计七对 / 国士；轮到自己时先去掉刚摸的牌
  4. 位次、点差：由四家点数算
  5. 门前：自家没有副露

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/mm3/postprocess.py \
        --pred ~/mahjong-llm/work/mm3/perception_v5/test_pred.npz      # 先在真实局面上验证规则，再修正预测
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mmj"))
sys.path.append("/mnt/d/mahjong-llm/train_small")
from actions import legal_mask  # noqa: E402
from analyze_discards import shanten  # noqa: E402
from state_features import P, S  # noqa: E402

CHI = ["legal_chi_low", "legal_chi_mid", "legal_chi_high"]
CALLS = CHI + ["legal_pon", "legal_kan", "legal_hora"]


def hand_counts(pl):
    return sum(pl[P[f"hand>={c}"]].astype(int) for c in (1, 2, 3, 4))


def chi_shapes(h, t):
    """被吃的牌 t（0-33）+ 手牌计数 h → (能吃下, 能吃中, 能吃上)。"""
    if t >= 27:
        return False, False, False
    n = t % 9
    return (n <= 6 and h[t + 1] > 0 and h[t + 2] > 0,
            1 <= n <= 7 and h[t - 1] > 0 and h[t + 1] > 0,
            n >= 2 and h[t - 2] > 0 and h[t - 1] > 0)


def guess_last(pl, chi, h):
    lasts = {p: np.flatnonzero(pl[P[f"kawa{p}_last"]]) for p in (3, 2, 1)}
    if chi:
        return int(lasts[3][0]) if len(lasts[3]) else None
    for p in (3, 2, 1):                      # 碰：手里有 2 张的那家
        if len(lasts[p]) and h[lasts[p][0]] >= 2:
            return int(lasts[p][0])
    for p in (3, 2, 1):
        if len(lasts[p]):
            return int(lasts[p][0])
    return None


CHI_TH = 0.1   # 能吃的局面里三者之和集中在 0.2 左右、不能吃的 99% 在 0.03 以下；用测试集前 1000 条定的门槛，其余 4000 条准确 99.95%


def refine(pl, sc):
    """单个局面：pl uint8 [58,34]、sc float [71] → 修正后的副本。"""
    pl, sc = pl.copy(), sc.copy()
    h = hand_counts(pl)
    own = h.sum() % 3 == 2                  # 轮到自己（摸牌后 / 鸣牌后要打牌）
    menzen = not pl[P["fuuro0_any"]].any()
    sc[S["menzen"]] = float(menzen)
    chi_any = sum(sc[S[c]] for c in CHI) > CHI_TH
    call_any = chi_any or any(sc[S[c]] > 0.5 for c in CALLS[3:])
    # last_kawa：轮到自己时 98% 就是上家牌河的最后一张；要决定鸣不鸣时按 guess_last 推；网络对这个平面几乎没学会
    k3 = np.flatnonzero(pl[P["kawa3_last"]])
    if own and len(k3):
        pl[P["last_kawa"]] = 0
        pl[P["last_kawa"], k3[0]] = 1
    elif call_any and not own:
        t = guess_last(pl, chi_any, h)
        if t is not None:
            pl[P["last_kawa"]] = 0
            pl[P["last_kawa"], t] = 1
    # 吃：先判断"能吃"，再按规则定种类
    last = np.flatnonzero(pl[P["last_kawa"]])
    shapes = chi_shapes(h, int(last[0])) if chi_any and len(last) else (False, False, False)
    for c, ok in zip(CHI, shapes):
        sc[S[c]] = float(ok)
    # 向听数：引擎算的是"摸牌之前"的 13 张（轮到自己时不一致的 21.5% 全是规则比引擎小 1）→ 先去掉刚摸的牌
    if h.sum():
        a = list(h)
        ts_ = np.flatnonzero(pl[P["last_tsumo"]])
        if own and len(ts_) and a[ts_[0]] > 0:
            a[ts_[0]] -= 1
        sh = min(max(shanten(a, menzen), 0), 6)
        for i in range(7):
            sc[S[f"shanten{i}"]] = float(i == sh)
    # 位次、点差
    s = [sc[S[f"score{p}"]] * 50000 for p in range(4)]
    rank = sum(x > s[0] for x in s[1:])
    for i in range(4):
        sc[S[f"rank{i}"]] = float(i == min(rank, 3))
    for p in (1, 2, 3):
        sc[S[f"diff{p}"]] = np.clip((s[0] - s[p]) / 20000, -2, 2)
    return pl, sc


def refine_all(planes, scalars):
    out = [refine(p, s) for p, s in zip(planes, scalars)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def report(name, pp, ps, tp, ts):
    pm, tm = legal_mask(pp, ps.round()), legal_mask(tp, ts)
    keys = ["shanten0", "shanten1", "shanten2", "rank0", "rank1", "diff1", "legal_chi_low", "legal_chi_mid", "legal_chi_high"]
    err = {k: np.abs(ps[:, S[k]] - ts[:, S[k]]).mean() for k in keys}
    sh_p = ps[:, [S[f"shanten{i}"] for i in range(7)]].argmax(1)
    sh_t = ts[:, [S[f"shanten{i}"] for i in range(7)]].argmax(1)
    lk = (pp[:, P["last_kawa"]].clip(max=1) == tp[:, P["last_kawa"]].clip(max=1)).all(1)
    print(f"[{name}] 合法手完全一致 {np.mean((pm == tm).all(1)):.1%}；向听数一致 {np.mean(sh_p == sh_t):.1%}；"
          f"last_kawa 一致 {np.mean(lk):.1%}")
    print("   标量平均误差：", "，".join(f"{k} {v:.3f}" for k, v in err.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="perception.py 的 test_pred.npz")
    ap.add_argument("--out", default=None, help="修正后的输出（默认同目录 test_pred_refined.npz）")
    args = ap.parse_args()
    W = os.path.expanduser("~/mahjong-llm/work")
    z = np.load(args.pred)
    n = len(z["planes"])
    tp, ts = np.load(f"{W}/mmj/test_planes.npy")[:n], np.load(f"{W}/mmj/test_scalars.npy")[:n]
    # 1) 规则本身的上限：真实局面里去掉 last_kawa 和吃的种类、只留"能不能吃"，再按规则补回来
    tp0, ts0 = tp.copy(), ts.copy()
    call_any = ts0[:, [S[c] for c in CALLS]].max(1) > 0
    tp0[:, P["last_kawa"]] = 0
    chi_any = ts0[:, [S[c] for c in CHI]].max(1) > 0
    ts0[:, [S[c] for c in CHI]] = 0
    ts0[chi_any, S["legal_chi_low"]] = 1.0          # 只保留"能吃"这个信息（三者之和 > 门槛）
    report("规则上限（真实局面）", *refine_all(tp0, ts0), tp, ts)
    # 2) 感知预测：修正前后
    pp, ps = z["planes"].astype(np.uint8), z["scalars"].astype(np.float32)
    report("感知 修正前", pp, ps, tp, ts)
    rp, rs = refine_all(pp, ps)
    report("感知 修正后", rp, rs, tp, ts)
    out = args.out or os.path.join(os.path.dirname(args.pred), "test_pred_refined.npz")
    np.savez_compressed(out, planes=rp, scalars=rs, rows=z["rows"])
    print("已保存", out)


if __name__ == "__main__":
    main()
