#!/usr/bin/env python3
"""三合一联合训练：每步混合三种输入模式（只局面 / 只图片 / 图片 + 局面）和两种任务（决策 / 看图念局面）。

    PY=~/mahjong-llm/venv-train/bin/python; T=/mnt/d/mahjong-llm/mm3/train.py
    # 冒烟（CPU 也能跑）：$PY -I $T --max_steps 2 --bs 1 --ga 1 --eval_n 2 --device cpu --out /tmp/mm3_smoke
    $PY -I $T --init_proj ~/mahjong-llm/work/mmj/mm_q35_08b_align/final.pt --out ~/mahjong-llm/work/mm3/tri_q35_08b

冻结：局面编码器、视觉塔（默认；--lora_visual 时给视觉塔也挂 LoRA）、LLM 本体。训练：局面投影层、LLM 的 LoRA、
视觉投影层 merger（--train_merger）。超参默认与之前一致：每步 16 条（bs × ga）、lr 1e-4、cosine、warmup 10%、LoRA r8/α16。
评测：测试集前 eval_n 条，三种模式分别生成，报与牌谱的一致率；eval_n = 5000 时每种模式写一份
generated_predictions_<mode>.jsonl（与 riichi_test5k 行对齐，可用 train_small/eval_vs_mortal.py、eval_reason.py 评）。
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
from data import TriData  # noqa: E402
from model import MahjongTri  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work")


def act(s):
    m = re.search(r"\{.*?\}", s, re.S)
    try:
        return json.loads(m.group(0))["action"].replace(" ", "") if m else None
    except Exception:
        return None


def evaluate(model, test, n, out, device, bs=8):
    n = min(n, len(test))
    res = {}
    model.eval()
    for mode in ("state", "image", "both"):
        preds = []
        for s in range(0, n, bs):
            items = [test.item(i, mode=mode, task="decide") for i in range(s, min(n, s + bs))]
            with torch.autocast(device, torch.bfloat16):
                preds += model.generate(items, max_new_tokens=128, device=device)
        gold = [test.human[i] for i in range(n)]
        res[mode] = float(np.mean([act(p) == act(g) for p, g in zip(preds, gold)]))
        res[mode + "_format"] = float(np.mean([act(p) is not None for p in preds]))
        res[mode + "_example"] = preds[0][:160]
        if n == len(test):
            with open(f"{out}/generated_predictions_{mode}.jsonl", "w", encoding="utf-8") as fo:
                for p, g in zip(preds, gold):
                    fo.write(json.dumps({"prompt": "", "predict": p, "label": g}, ensure_ascii=False) + "\n")
    model.train()
    model.enc.eval()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--enc", default=f"{W}/mmj/encoder_full_soft_e16.pt")
    ap.add_argument("--init_proj", default=None, help="mmj 对齐 / 微调阶段的 final.pt（只取 proj.*）")
    ap.add_argument("--label", choices=["mortal", "human"], default="mortal")
    ap.add_argument("--modes", default="0.3,0.4,0.3", help="state,image,both 的抽样概率")
    ap.add_argument("--p_caption", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_visual", action="store_true")
    ap.add_argument("--train_merger", action="store_true")
    ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="关掉梯度检查点（默认开）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    dev = args.device

    model = MahjongTri(args.llm, args.enc)
    for p in model.parameters():
        p.requires_grad_(False)
    model.add_lora(r=args.lora_r, alpha=2 * args.lora_r, visual=args.lora_visual)
    if not args.no_grad_ckpt:   # 带图时激活值很大（视觉塔 3072 图块 + LLM 约 900 token），用梯度检查点换显存
        model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.proj.parameters():
        p.requires_grad_(True)
    if args.train_merger:
        for n, p in model.vlm.named_parameters():
            if "visual.merger" in n:
                p.requires_grad_(True)
    if args.init_proj:
        sd = {k: v for k, v in torch.load(args.init_proj, map_location="cpu", weights_only=False).items()
              if k.startswith("proj.")}
        model.load_state_dict(sd, strict=False)
        print(f"局面投影层载入 {args.init_proj}（{len(sd)} 个张量）", flush=True)
    model.to(dev)
    model.enc.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    groups = {"proj": 0, "lora": 0, "merger": 0}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups["proj" if n.startswith("proj.") else "merger" if "merger" in n else "lora"] += p.numel()
    print(f"可训练参数：{ {k: f'{v / 1e6:.2f}M' for k, v in groups.items()} }", flush=True)

    modes = tuple(float(x) for x in args.modes.split(","))
    train = TriData("train", args.label, modes, args.p_caption)
    test = TriData("test", args.label)
    total = args.max_steps or int(math.ceil(len(train) * args.epochs / (args.bs * args.ga)))
    warm = max(1, int(total * 0.1))
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    print(f"训练 {len(train)} 条，标签 {args.label}，模式 {modes}，caption {args.p_caption}；"
          f"共 {total} 步（每步 {args.bs}×{args.ga} 条）", flush=True)

    order = np.concatenate([np.random.default_rng(0).permutation(len(train))
                            for _ in range(int(math.ceil(total * args.bs * args.ga / len(train))) + 1)])
    log = open(f"{args.out}/log.jsonl", "a")
    # from_pretrained 返回 eval 模式，HF 只在 training=True 时做梯度检查点；不切回训练模式 --no_grad_ckpt 形同虚设
    # （队列 30 / R53 就是这样跑完的：检查点没生效，显存约 11GB；LoRA 与模型 dropout 都是 0，结果不受影响）
    model.train()
    model.enc.eval()
    t0, run, k = time.time(), 0.0, 0
    for step in range(total):
        for g in range(args.ga):
            idx = order[k:k + args.bs]
            k += args.bs
            items = [train.item(int(i)) for i in idx]
            with torch.autocast(dev, torch.bfloat16):
                loss = model(items, device=dev) / args.ga
            loss.backward()
            run += loss.item()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step == total - 1:
            el = time.time() - t0
            rec = {"step": step + 1, "loss": run / min(10, step % 10 + 1), "lr": sched.get_last_lr()[0],
                   "s_per_step": round(el / (step + 1), 2), "eta_min": round(el / (step + 1) * (total - step - 1) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run = 0.0
        if (step + 1) % 1000 == 0:
            torch.save(model.trainable_state(), f"{args.out}/step{step + 1}.pt")
    torch.save(model.trainable_state(), f"{args.out}/final.pt")
    print(f"训练完成 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    r = evaluate(model, test, args.eval_n, args.out, dev)
    print("最终评测", json.dumps(r, ensure_ascii=False), flush=True)
    log.write(json.dumps({"final_eval": r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
