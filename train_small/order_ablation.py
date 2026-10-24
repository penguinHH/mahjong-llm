#!/usr/bin/env python3
"""R35（mm reason_tfx + facts_val）推理时評価順消融：LLM 到底有没有自己决策？

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/train_small/order_ablation.py

三个变体，除評価順外输入完全相同（真实局面 token + 同一份 facts_val 文本）：
  orig     原样。应逐条复现 R35 的动作（管线校验；生成参数与 train_mm.py evaluate 相同：贪心、128 token、bs 32）
  strip    去掉"評価順：…。"句。训练时从来没见过没有評価順的输入，数字只能当"下界"看
  reverse  評価順名次倒序。若动作跟着第 1 名换 → 在照抄；若动作基本不变 → 没有依赖評価順的次序
每个变体写 W/mmj/order_ablation_<变体>.jsonl（LLaMA-Factory 格式，行对齐），打印一致率与"动作与 R35 相同"比例。
"""
import hashlib
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MMJ = "/mnt/d/mahjong-llm/mmj"
sys.path.insert(0, HERE)
sys.path.append(MMJ)
sys.path.append("/mnt/d/mahjong-llm/tools")
from eval_acc import parse  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work")
D = f"{W}/mmj"
ORDER = re.compile(r"評価順：([^。]+)。")


def strip_order(f):
    return ORDER.sub("", f)


def reverse_order(f):
    return ORDER.sub(lambda m: "評価順：" + " > ".join(p.strip() for p in reversed(m.group(1).split(">"))) + "。", f)


def main():
    pl, sc = np.load(f"{D}/test_planes.npy"), np.load(f"{D}/test_scalars.npy")
    text = [json.loads(l) for l in open(f"{D}/test_text.jsonl", encoding="utf-8")]
    facts = [l.rstrip("\n") for l in open(f"{D}/test_facts_val.jsonl", encoding="utf-8")]
    src = [json.loads(l) for l in open(f"{W}/sft/riichi_test5k.jsonl", encoding="utf-8")]
    mor = {r["md5"]: r for r in map(json.loads, open(f"{W}/sft/riichi_test5k_mortal_meta.jsonl", encoding="utf-8"))}
    labels = [mor.get(hashlib.md5(t["instruction"].encode()).hexdigest()) for t in src]
    r35 = [parse(json.loads(l)["predict"]) for l in
           open(f"{D}/mm_q35_08b_reason_tf256b6_v_x/generated_predictions.jsonl", encoding="utf-8")]
    n = 5000

    _spec = importlib.util.spec_from_file_location("mmj_model", os.path.join(MMJ, "model.py"))
    mmj_model = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mmj_model)
    model = mmj_model.MahjongMM("Qwen/Qwen3.5-0.8B", f"{D}/encoder_tf256b6_v_x.pt")
    model.add_lora(r=8, alpha=16)
    sd = torch.load(f"{D}/mm_q35_08b_reason_tf256b6_v_x/final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to("cuda").eval()

    for name, fn in (("orig", lambda x: x), ("strip", strip_order), ("reverse", reverse_order)):
        ins = [f"{t['instruction']}\n事実：{fn(f)}" for t, f in zip(text, facts)]
        preds = []
        with torch.no_grad():
            for i in range(0, n, 32):
                with torch.autocast("cuda", torch.bfloat16):
                    preds += model.generate(torch.from_numpy(pl[i:i + 32]).to("cuda"),
                                            torch.from_numpy(sc[i:i + 32]).to("cuda"),
                                            ins[i:i + 32], max_new_tokens=128)
        with open(f"{D}/order_ablation_{name}.jsonl", "w", encoding="utf-8") as fo:
            for p, t in zip(preds, text):
                fo.write(json.dumps({"prompt": "", "predict": p, "label": t["output"]}, ensure_ascii=False) + "\n")
        acts = [parse(p) for p in preds]
        nl = sum(l is not None for l in labels)
        h = sum(a == l["human"] for a, l in zip(acts, labels) if l)
        m = sum(a == l["mortal"] for a, l in zip(acts, labels) if l)
        print(f"{name}: 与牌谱 {h / nl:.2%}  与 Mortal {m / nl:.2%}  动作与 R35 相同 {sum(a == b for a, b in zip(acts, r35)) / n:.2%}  "
              f"格式 {sum(a is not None for a in acts) / n:.2%}", flush=True)


if __name__ == "__main__":
    main()
