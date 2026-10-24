#!/usr/bin/env python3
"""第 2 步训练：局面编码器（冻结）+ 投影层 + LLM。

    PY=~/mahjong-llm/venv-train/bin/python; M=/mnt/d/mahjong-llm/mmj/train_mm.py
    # A. 对齐：LLM 冻住、只训投影层，目标 = 局面描述（captions.py）
    $PY -I $M --stage align --out ~/mahjong-llm/work/mmj/mm_q35_08b_align
    # B. 微调：加 LoRA，投影层继续训，目标 = 理由 + Mortal 动作（或 --target action 只要动作）
    $PY -I $M --stage sft --target reason --init ~/mahjong-llm/work/mmj/mm_q35_08b_align/final.pt \
        --out ~/mahjong-llm/work/mmj/mm_q35_08b_reason
    # C. 同 B，指令后附上程序算好的事实（先跑 facts.py）：加 --facts，--out ~/mahjong-llm/work/mmj/mm_q35_08b_reason_facts
    # 冒烟测试：--max_steps 20 --eval_n 32

超参默认与之前的文字版 LoRA 一致（每步 16 条、lr 1e-4、cosine、warmup 10%、LoRA r8/α16、全部线性层、1 epoch）。
评测：在测试集上生成，写成与 LLaMA-Factory 相同格式的 generated_predictions.jsonl（与 riichi_test5k 行对齐），
可直接用 train_small/eval_vs_mortal.py、eval_reason.py 评。
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from captions import CAPTION_INSTRUCTION, caption  # noqa: E402
from model import MahjongMM  # noqa: E402

D = os.path.expanduser("~/mahjong-llm/work/mmj")


def load_split(split, target, facts=False, facts_name="facts"):
    planes = np.load(f"{D}/{split}_planes.npy")
    scalars = np.load(f"{D}/{split}_scalars.npy")
    text = [json.loads(l) for l in open(f"{D}/{split}_text.jsonl", encoding="utf-8")]
    if target == "caption":
        ins = [CAPTION_INSTRUCTION] * len(text)
        ans = [caption(p, s) for p, s in zip(planes, scalars)]
    else:
        ins = [t["instruction"] for t in text]
        if facts:      # facts.py 生成、与 text 行对齐：程序算好的数字附在指令后面
            fa = [json.loads(l) for l in open(f"{D}/{split}_{facts_name}.jsonl", encoding="utf-8")]
            assert len(fa) == len(ins)
            ins = [f"{i}\n事実：{x}" for i, x in zip(ins, fa)]
        outs = [json.loads(t["output"]) for t in text]
        ans = [json.dumps(o if target == "reason" else {"action": o["action"]}, ensure_ascii=False) for o in outs]
    return planes, scalars, ins, ans


def evaluate(model, n, out_dir, target, bs=32, max_new=128, facts=False, facts_name="facts"):
    """测试集前 n 条生成；动作类目标写 generated_predictions.jsonl 并报一致率。"""
    planes, scalars, ins, ans = load_split("test", "caption" if target == "caption" else "action", facts, facts_name)
    n = min(n, len(ins))
    dev = next(model.proj.parameters()).device
    preds = []
    model.eval()
    for i in range(0, n, bs):
        P = torch.from_numpy(planes[i:i + bs]).to(dev)
        S = torch.from_numpy(scalars[i:i + bs]).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            preds += model.generate(P, S, ins[i:i + bs], max_new_tokens=max_new)
    model.train()
    model.enc.eval()
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/generated_predictions{'' if n == len(planes) else f'_first{n}'}.jsonl", "w",
              encoding="utf-8") as fo:
        for p, a in zip(preds, ans[:n]):
            fo.write(json.dumps({"prompt": "", "predict": p, "label": a}, ensure_ascii=False) + "\n")
    if target == "caption":
        # 逐项准确率：场况 / 手牌 / ドラ / リーチ者 / 向聴 / 直前の捨て牌（按句首关键词对齐）
        keys = {"場況": "場・", "手牌": "手牌", "ドラ": "ドラ", "リーチ": "リーチ者", "向聴": ("向聴", "テンパイ"), "直前": "直前"}
        hit, tot = {k: 0 for k in keys}, {k: 0 for k in keys}
        for p, a in zip(preds, ans[:n]):
            ps, as_ = p.strip().split("。"), a.split("。")
            for k, kw in keys.items():
                kws = kw if isinstance(kw, tuple) else (kw,)
                ref = next((s for s in as_ if any(w in s for w in kws)), None)
                if ref is None:
                    continue
                tot[k] += 1
                hit[k] += ref in ps
        exact = np.mean([p.strip() == a for p, a in zip(preds, ans[:n])])
        return {"caption_exact": float(exact), **{k: round(hit[k] / max(1, tot[k]), 4) for k in keys},
                "example": preds[0][:200]}

    def act(s):
        m = re.search(r"\{.*?\}", s, re.S)
        try:
            return json.loads(m.group(0))["action"].replace(" ", "") if m else None
        except Exception:
            return None
    ok = [act(p) == act(a) for p, a in zip(preds, ans[:n])]
    fmt = [act(p) is not None for p in preds]
    return {"human": float(np.mean(ok)), "format": float(np.mean(fmt)), "example": preds[0][:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--enc", default=f"{D}/encoder_full_soft_e16.pt")
    ap.add_argument("--stage", choices=["align", "sft"], required=True)
    ap.add_argument("--target", choices=["caption", "reason", "action"], default=None)
    ap.add_argument("--init", default=None, help="上一阶段的 final.pt（投影层 / LoRA）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=0, help="每次前向的条数（梯度累积到 bs）；0 = 不拆")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--eval_n", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--facts", action="store_true", help="指令后附上 facts.py 算好的事实（向聴・受け入れ・点差・安全度）")
    ap.add_argument("--facts_name", default="facts", help="事实文件 {split}_<名>.jsonl；facts_pol = 再加编码器评价顺序（policy_facts.py）")
    args = ap.parse_args()
    args.target = args.target or ("caption" if args.stage == "align" else "reason")
    args.lr = args.lr or (1e-3 if args.stage == "align" else 1e-4)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda"

    model = MahjongMM(args.llm, args.enc)
    for p in model.enc.parameters():
        p.requires_grad_(False)
    for p in model.llm.parameters():
        p.requires_grad_(False)
    if args.stage == "sft":
        model.add_lora(r=args.lora_r, alpha=2 * args.lora_r)
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=False)
        missing = model.load_state_dict(sd, strict=False)
        print(f"载入 {args.init}：{len(sd)} 个张量（未匹配 {len(missing.unexpected_keys)}）", flush=True)
    model.to(dev)
    model.enc.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:                       # 可训练参数用 fp32（与 LLaMA-Factory 的 upcast 一致），前向在 bf16 autocast 里
        p.data = p.data.float()
    n_tr = sum(p.numel() for p in trainable)
    print(f"LLM {args.llm} hidden={model.llm.config.hidden_size}；可训练参数 {n_tr / 1e6:.2f}M "
          f"（投影层 {sum(p.numel() for p in model.proj.parameters()) / 1e6:.2f}M）", flush=True)
    print("模板前半段:", repr(model.prefix), "| 后半段:", repr(model.suffix), flush=True)

    planes, scalars, ins, ans = load_split("train", args.target, args.facts, args.facts_name)
    n = len(ins)
    total = args.max_steps or int(math.ceil(n * args.epochs / args.bs))
    warm = max(1, int(total * args.warmup))
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    print(f"训练 {n} 条，目标 {args.target}，共 {total} 步（每步 {args.bs} 条），lr {args.lr}", flush=True)
    print("目标示例:", ans[0][:160], flush=True)

    mb = args.micro_bs or args.bs
    log = open(f"{args.out}/log.jsonl", "a")
    rng = np.random.default_rng(0)
    order = np.concatenate([rng.permutation(n) for _ in range(int(math.ceil(total * args.bs / n)) + 1)])
    t0, run = time.time(), 0.0
    for step in range(total):
        idx = order[step * args.bs:(step + 1) * args.bs]
        opt.zero_grad(set_to_none=True)
        for j in range(0, len(idx), mb):       # 微批次累积梯度：每步仍是 bs 条
            sub = idx[j:j + mb]
            P = torch.from_numpy(planes[sub]).to(dev)
            S = torch.from_numpy(scalars[sub]).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(P, S, [ins[i] for i in sub], [ans[i] for i in sub]) * len(sub) / len(idx)
            loss.backward()
            run += loss.item()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 20 == 0 or step == total - 1:
            el = time.time() - t0
            rec = {"step": step + 1, "loss": run / min(20, (step % 20) + 1), "lr": sched.get_last_lr()[0],
                   "s_per_step": el / (step + 1), "eta_min": el / (step + 1) * (total - step - 1) / 60}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run = 0.0
        if args.eval_every and (step + 1) % args.eval_every == 0:
            r = evaluate(model, min(args.eval_n, 256), args.out, args.target, facts=args.facts, facts_name=args.facts_name)
            print("eval", json.dumps(r, ensure_ascii=False), flush=True)
    torch.save(model.trainable_state(), f"{args.out}/final.pt")
    print(f"训练完成，{(time.time() - t0) / 60:.1f} 分钟；保存 {args.out}/final.pt", flush=True)
    r = evaluate(model, args.eval_n, args.out, args.target, facts=args.facts, facts_name=args.facts_name)
    print("最终评测", json.dumps(r, ensure_ascii=False), flush=True)
    log.write(json.dumps({"final_eval": r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
