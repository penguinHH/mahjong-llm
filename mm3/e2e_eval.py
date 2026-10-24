#!/usr/bin/env python3
"""三合一端到端评测：卓面图片 → 感知模块（perception.py 的 test_pred.npz）→ 局面编码器 + LLM → 打牌和理由。

LLM（mmj 训好的理由模型）的输入只有三样：35 个局面 token、指令里的合法手列表、程序算的事实文本。
这里全部从"预测的局面张量"重建，不碰牌谱原文：
  合法手      ← legal_dahai / legal_dahai_aka 平面 + legal_* 标量
  事实文本    ← 把平面和标量还原成 facts.py 要的字段（手牌、見え枚数、ドラ表示牌、点数、立直者、各家牌河……），
                再调用同一个 facts_text
  评价顺序    ← 编码器策略头的第 1 名 + 价值头排的另外两名（与 policy_facts.py 相同）；
                --gap 时另外两名改由 mmj/gap_head.py 的次点 / 档位头给出（分差写成各档代表值）
预测局面里看不见的特征（手切、振听）清零。--state true 用真实局面走同一条路，检查还原得准不准（应与原事实文件几乎一致）。

    PY=~/mahjong-llm/venv-train/bin/python
    $PY -I /mnt/d/mahjong-llm/mm3/e2e_eval.py --state true --out ~/mahjong-llm/work/mm3/e2e_true
    $PY -I /mnt/d/mahjong-llm/mm3/e2e_eval.py --state ~/mahjong-llm/work/mm3/perception_tfx/test_pred.npz \
        --out ~/mahjong-llm/work/mm3/e2e_perception
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MMJ = os.path.join(HERE, "..", "mmj")
sys.path.insert(0, HERE)
sys.path.append(MMJ)                                   # 放最后：mmj 也有 model.py
sys.path.append("/mnt/d/mahjong-llm/tools")
import facts as F  # noqa: E402
from perception import load_encoder  # noqa: E402
from actions import ACTIONS, legal_mask  # noqa: E402
from build_explanations import act_name, name  # noqa: E402
from state_features import KINDS, P, PLANES, S, SCALARS, WINDS, short_instruction  # noqa: E402
from gap_head import CANON, load_gap, load_gap_encoder  # noqa: E402   （放在最后：它会把 mmj 插到 sys.path 最前面）

_spec = importlib.util.spec_from_file_location("mmj_model", os.path.join(MMJ, "model.py"))
mmj_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mmj_model)

W = os.path.expanduser("~/mahjong-llm/work")
D = f"{W}/mmj"
UNSEEN_PLANES = [i for i, n in enumerate(PLANES) if n.startswith("tedashi")]
AKA = {"5m": "5mr", "5p": "5pr", "5s": "5sr"}
F.fields = lambda f: f          # facts_text 直接吃还原好的字段字典


# ---------------------------------------------------------------- 张量 → facts.py 要的字段
def counts(pl, prefix):
    return sum(pl[P[f"{prefix}>={c}"]].astype(int) for c in (1, 2, 3, 4))


def tiles_from(cnt, aka_row=None):
    out = []
    for k, n in enumerate(cnt):
        t = KINDS[k]
        for j in range(int(n)):
            out.append(AKA[t] if aka_row is not None and j == 0 and t in AKA and aka_row[k] else t)
    return out


def kinds_with_dup(pl, p_any, p_two):
    out = []
    for k in np.flatnonzero(pl[P[p_any]]):
        out += [KINDS[k]] * (2 if pl[P[p_two], k] else 1)
    return out


CALLS = ("chi_low", "chi_mid", "chi_high", "pon")


def guess_last_kawa(pl, legal):
    """last_kawa 平面为空但能鸣牌时：从各家牌河的最后一张里推（同 render_board 的高亮规则：能吃只看上家，否则上家→対面→下家）。"""
    lasts = {p: np.flatnonzero(pl[P[f"kawa{p}_last"]]) for p in (3, 2, 1)}
    for p in ([3] if any(a.startswith("chi") for a in legal) else [3, 2, 1]):
        if len(lasts[p]):
            return KINDS[lasts[p][0]]
    return ""


def to_fields(pl, sc):
    legal = [ACTIONS[i] for i in np.flatnonzero(legal_mask(pl[None], sc[None].round())[0])]
    last = np.flatnonzero(pl[P["last_kawa"]])
    if not len(last) and any(a in CALLS for a in legal):
        g = guess_last_kawa(pl, legal)
        last = [KINDS.index(g)] if g else []
    return {
        "my hands": tiles_from(counts(pl, "hand"), pl[P["hand_aka"]]),
        "tiles seen": tiles_from(counts(pl, "seen")),
        "Legal actions": legal,
        "last kawa tile": KINDS[last[0]] if len(last) else "",
        "bakaze": WINDS[int(np.argmax([sc[S[f"bakaze_{w}"]] for w in WINDS]))],
        "jikaze": WINDS[int(np.argmax([sc[S[f"jikaze_{w}"]] for w in WINDS]))],
        "is menzen": bool(round(float(sc[S["menzen"]]))),
        "fuuro overview": [[kinds_with_dup(pl, f"fuuro{p}_any", f"fuuro{p}_>=2")] if pl[P[f"fuuro{p}_any"]].any() else []
                           for p in range(4)],
        "dora indicators": [KINDS[k] for k in np.flatnonzero(pl[P["dora_indicator"]])],
        "Scores": [int(round(float(sc[S[f"score{p}"]]) * 50000 / 100)) * 100 for p in range(4)],
        "riichi declared": [bool(round(float(sc[S[f"riichi_declared{p}"]]))) for p in range(4)],
        "kawa overview": [kinds_with_dup(pl, f"kawa{p}_any", f"kawa{p}_>=2") for p in range(4)],
    }


def show(a):
    if a.startswith("dahai:"):
        return name(a.split(":")[1]) + "切り"
    return F.CALL_NAME.get(a, act_name(a))


def eval_order(enc, planes, scalars, mask, device, bs=1024):
    """policy_facts.py 的评价顺序：第 1 名按策略头，其余两名按价值头排，标与第 1 名的 Q 差。"""
    out = []
    with torch.no_grad():
        for i in range(0, len(planes), bs):
            lg, v = enc(torch.from_numpy(planes[i:i + bs]).to(device), torch.from_numpy(scalars[i:i + bs]).to(device),
                        torch.from_numpy(mask[i:i + bs]).to(device), with_value=True)
            for lg_, v_, m in zip(lg.float().cpu().numpy(), v.float().cpu().numpy(), mask[i:i + bs]):
                legal = np.flatnonzero(m)
                if not len(legal):
                    out.append("")
                    continue
                first = legal[np.argmax(lg_[legal])]
                rest = [k for k in legal[np.argsort(-v_[legal])] if k != first][:2]
                parts = [show(ACTIONS[first])] + [f"{show(ACTIONS[k])}（{v_[k] - v_[first]:+.1f}）" for k in rest]
                out.append("評価順：" + " > ".join(parts) + "。")
    return out


def eval_order_gap(enc, gap, planes, scalars, mask, device, bs=1024):
    """--gap：第 1 名仍按策略头；次点和第三名按次点头排，分差写成该档的代表值（mmj/gap_head.py 的 CANON），
    这样 LLM 读到的数字离 0.5 / 3 两个阈值都很远，照抄就是头预测的档位。第三名的分差不小于次点的。"""
    out = []
    with torch.no_grad():
        for i in range(0, len(planes), bs):
            m = torch.from_numpy(mask[i:i + bs]).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                tiles, glob = enc.enc(torch.from_numpy(planes[i:i + bs]).to(device),
                                      torch.from_numpy(scalars[i:i + bs]).to(device))
                first = enc.head(tiles, glob).float().masked_fill(~m, -1e4).argmax(1)
                o = gap(tiles.float(), glob.float(), first).float()
            for f, o_, m_ in zip(first.cpu().numpy(), o.cpu().numpy(), mask[i:i + bs]):
                legal = np.flatnonzero(m_)
                if not len(legal):
                    out.append("")
                    continue
                rest = [k for k in legal[np.argsort(-o_[legal, 0])] if k != f][:2]
                parts, g = [show(ACTIONS[f])], 0.0
                for k in rest:
                    g = max(g, CANON[int(o_[k, 1:].argmax())])
                    parts.append(f"{show(ACTIONS[k])}（{-g:+.1f}）")
                out.append("評価順：" + " > ".join(parts) + "。")
    return out


def act(s):
    try:
        return json.loads(s[s.index("{"):s.rindex("}") + 1])["action"].replace(" ", "")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="true = 真实局面；否则为 perception.py 的 test_pred.npz")
    ap.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--enc", default=f"{D}/encoder_tf256b6_v_x.pt")
    ap.add_argument("--model", default=f"{D}/mm_q35_08b_reason_tf256b6_v_x/final.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--gap", default=None, help="mmj/gap_head.py 训出的次点 / 档位头；给了就用它生成評価順")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    tp, ts = np.load(f"{D}/test_planes.npy"), np.load(f"{D}/test_scalars.npy")
    if args.state == "true":
        pl, sc = tp.copy(), ts.copy()
    else:
        z = np.load(args.state)
        assert (z["rows"] == np.arange(len(z["rows"]))).all(), "test_pred.npz 的行号应从 0 连续"
        pl, sc = z["planes"].astype(np.uint8), z["scalars"].astype(np.float32)
    n = min(args.n, len(pl))
    pl, sc = pl[:n], sc[:n]
    if args.state != "true":          # 图片上看不见的特征清零；真实局面保留，好让还原的事实能与原文件逐字对比
        pl[:, UNSEEN_PLANES] = 0
        sc[:, S["furiten"]] = 0
    text = [json.loads(l) for l in open(f"{D}/test_text.jsonl", encoding="utf-8")][:n]
    ref_facts = [json.loads(l) for l in open(f"{D}/test_facts_val.jsonl", encoding="utf-8")][:n]

    # 指令 + 事实（从张量还原）
    fl = [to_fields(pl[i], sc[i]) for i in range(n)]
    ins = [short_instruction(f) for f in fl]
    mask = legal_mask(pl, sc.round())
    model = mmj_model.MahjongMM(args.llm, args.enc)
    model.add_lora(r=8, alpha=16)
    sd = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(dev).eval()
    enc_full = load_encoder(args.enc, dev)       # 带策略头 + 价值头的完整 EncoderPolicy（MahjongMM 里只有编码器主干）
    if args.gap:                                  # 微调过编码器的次点头用它自带的编码器排序（LLM 的 35 个 token 仍来自 --enc）
        order = eval_order_gap(load_gap_encoder(args.gap, dev) or enc_full, load_gap(args.gap, dev), pl, sc, mask, dev)
    else:
        order = eval_order(enc_full, pl, sc, mask, dev)
    # 推不出被鸣的牌时，事实里不写鸣牌那部分（facts.call_after 需要 last kawa tile）；指令里的合法手不变
    safe = [f if f["last kawa tile"] or not any(a in CALLS for a in f["Legal actions"])
            else dict(f, **{"Legal actions": [a for a in f["Legal actions"] if a not in CALLS]}) for f in fl]
    facts = [F.facts_text(f) + o for f, o in zip(safe, order)]
    print(f"指令与原文一致：{np.mean([a == t['instruction'] for a, t in zip(ins, text)]):.1%}；"
          f"事实与原事实文件一致：{np.mean([a == b for a, b in zip(facts, ref_facts)]):.1%}", flush=True)
    for i in range(3):
        if facts[i] != ref_facts[i]:
            print("  还原:", facts[i][:200], "\n  原文:", ref_facts[i][:200], flush=True)
    full = [f"{i}\n事実：{x}" for i, x in zip(ins, facts)]

    preds = []
    for i in range(0, n, args.bs):
        with torch.autocast("cuda", torch.bfloat16):
            preds += model.generate(torch.from_numpy(pl[i:i + args.bs]).to(dev), torch.from_numpy(sc[i:i + args.bs]).to(dev),
                                    full[i:i + args.bs], max_new_tokens=128)
    gold = [t["output"] for t in text]
    with open(f"{args.out}/generated_predictions.jsonl", "w", encoding="utf-8") as fo:
        for p, g in zip(preds, gold):
            fo.write(json.dumps({"prompt": "", "predict": p, "label": g}, ensure_ascii=False) + "\n")
    agree = np.mean([act(p) == act(g) for p, g in zip(preds, gold)])
    fmt = np.mean([act(p) is not None for p in preds])
    res = {"n": n, "与牌谱一致": float(agree), "格式正确": float(fmt), "例": preds[0][:200]}
    print("端到端结果", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{args.out}/eval.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
