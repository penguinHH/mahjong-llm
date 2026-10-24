#!/usr/bin/env python3
"""遗忘检查 + 推理显存实测（GPU，约 10 分钟）。

1) 遗忘：同一组通用问题（日 / 中 / 英：算术、常识、翻译、代码、指令遵循），分别问
   不挂 LoRA 的 Qwen3.5-0.8B、挂文字版麻将 LoRA（LLaMA-Factory，Mortal 标签）、挂编码器 + LLM 版 LoRA（mmj，加事实），
   按答案里是否含标准答案关键词打分；另算"漂移"：基座自己的贪心回答在挂 LoRA 后的平均每 token 负对数似然（越接近基座越没变）。
   挂 / 摘 LoRA 用 peft 的 disable_adapter()，同一个模型对象，对照最干净。
2) 显存：权重常驻、单条短问答、8 条并发、1300 token 文字 prompt、看图（768 个图像 token）、编码器 + LLM（带事实），
   各自的峰值 torch.cuda.max_memory_allocated（不含 CUDA 上下文约 0.3～0.5GB，nvidia-smi 会多显示这部分）。

    HF_HUB_OFFLINE=1 ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/train_small/forgetting_check.py
"""
import json
import os
import sys

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

W = os.path.expanduser("~/mahjong-llm/work")
BASE = "Qwen/Qwen3.5-0.8B"
TEXT_LORA = f"{W}/ckpt/qwen35-0.8b-riichi_100k-mortal"
MM_LORA = f"{W}/mmj/mm_q35_08b_reason_facts/final.pt"

QA = [  # (问题, 任一关键词出现即算对)
    ("17×23はいくつですか？数字だけ答えてください。", ["391"]),
    ("123+456は？", ["579"]),
    ("2の10乗はいくつ？", ["1024"]),
    ("1から10までの整数の和は？", ["55"]),
    ("日本の首都はどこですか？", ["東京"]),
    ("富士山の標高は約何メートルですか？", ["3776", "3,776"]),
    ("水の化学式は？", ["H2O", "H₂O"]),
    ("太陽系で一番大きい惑星は？", ["木星"]),
    ("「ありがとう」を英語に訳してください。", ["thank"]),
    ("一年は何日ですか（うるう年でない場合）？", ["365"]),
    ("中国的首都是哪里？", ["北京"]),
    ("中国最长的河流是哪条？", ["长江", "長江"]),
    ("一周有几天？", ["7", "七"]),
    ("水在标准大气压下的沸点是多少摄氏度？", ["100"]),
    ("What is the largest planet in the solar system?", ["jupiter"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("Translate 'cat' into Japanese.", ["猫", "ねこ", "ネコ"]),
    ("What is 9 times 8?", ["72"]),
    ("Pythonでリストxを逆順にした新しいリストを作る式を1行で書いてください。", ["[::-1]", "reversed("]),
    ("「はい」か「いいえ」だけで答えてください：太陽は東から昇りますか？", ["はい"]),
    ("麻雀の役満「国士無双」は何種類の么九牌を集める役ですか？", ["13", "十三"]),
    ("麻雀で、リーチした後に自分の捨て牌で和了れる牌を見逃すと何と呼ばれる状態になりますか？", ["フリテン", "振聴", "振り聴"]),
]


def chat_ids(tok, qs):
    texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for q in qs]
    tok.padding_side = "left"
    return tok(texts, return_tensors="pt", padding=True).to("cuda")


@torch.no_grad()
def answer(model, tok, qs, max_new=64):
    enc = chat_ids(tok, qs)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


@torch.no_grad()
def nll(model, tok, qs, ans):
    """每条回答在 model 下的平均每 token 负对数似然。"""
    res = []
    for q, a in zip(qs, ans):
        p = chat_ids(tok, [q])["input_ids"][0]
        full = torch.cat([p, tok(a + tok.eos_token, return_tensors="pt", add_special_tokens=False)["input_ids"][0].cuda()])
        logits = model(input_ids=full[None]).logits[0, len(p) - 1:-1].float()
        res.append(torch.nn.functional.cross_entropy(logits, full[len(p):]).item())
    return float(np.mean(res))


def score(ans):
    return [any(k.lower() in a.lower().replace(" ", "") for k in keys) for a, (_, keys) in zip(ans, QA)]


def peak(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    qs = [q for q, _ in QA]
    proc = AutoProcessor.from_pretrained(BASE)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    mem = {"权重常驻（含视觉塔）": torch.cuda.memory_allocated() / 2**30}

    # ---- 显存（基座）
    mem["单条短问答"] = peak(lambda: answer(model, tok, qs[:1]))
    mem["8 条并发短问答"] = peak(lambda: answer(model, tok, qs[:8]))
    long_prompt = json.loads(open(f"{W}/sft/riichi_test5k.jsonl", encoding="utf-8").readline())["instruction"]
    mem["文字版麻将 prompt（约 1300 token）"] = peak(lambda: answer(model, tok, [long_prompt], 16))
    img = Image.open(f"{W}/vlm/img/test/000000.png").convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "以下の局面で最善の行動を選び、JSONで答えてください。"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    vin = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    mem["看图（1024×768 → 768 图像 token）"] = peak(lambda: model.generate(**vin, max_new_tokens=16, do_sample=False))

    # ---- 遗忘：文字版麻将 LoRA
    base_ans = answer(model, tok, qs)
    base_nll = nll(model, tok, qs, base_ans)
    model = PeftModel.from_pretrained(model, TEXT_LORA).eval()
    text_ans = answer(model, tok, qs)
    text_nll = nll(model, tok, qs, base_ans)
    with model.disable_adapter():
        assert answer(model, tok, qs[:2]) == base_ans[:2], "摘下 LoRA 后应与基座完全一致"
    del model
    torch.cuda.empty_cache()

    # ---- 遗忘：编码器 + LLM 版 LoRA（mmj，AutoModelForCausalLM 结构）+ 这条管线的推理显存
    sys.path.insert(0, "/mnt/d/mahjong-llm/mmj")
    from model import MahjongMM
    from train_mm import load_split
    mm = MahjongMM(BASE, f"{W}/mmj/encoder_full_soft_e16.pt")
    mm.add_lora(r=8, alpha=16)
    mm.load_state_dict(torch.load(MM_LORA, map_location="cpu", weights_only=False), strict=False)
    mm.cuda().eval()
    mm_ans = answer(mm.llm, mm.tok, qs)
    mm_nll = nll(mm.llm, mm.tok, qs, base_ans)
    pl, sc, ins, _ = load_split("test", "action", True)
    P, S = torch.from_numpy(pl[:1]).cuda(), torch.from_numpy(sc[:1]).cuda()

    def run_mm():
        with torch.autocast("cuda", torch.bfloat16):
            mm.generate(P, S, ins[:1], max_new_tokens=96)
    mem["编码器 + LLM（带事实，生成理由）"] = peak(run_mm)

    sb, st, sm = score(base_ans), score(text_ans), score(mm_ans)
    print("\n== 遗忘检查（22 题，答案含关键词算对）")
    print(f"  基座            {sum(sb):2d}/22   漂移（基座回答的 NLL）{base_nll:.3f}")
    print(f"  + 文字版 LoRA   {sum(st):2d}/22   {text_nll:.3f}")
    print(f"  + 编码器版 LoRA {sum(sm):2d}/22   {mm_nll:.3f}")
    for (q, _), b, t, m, ab, at, am in zip(QA, sb, st, sm, base_ans, text_ans, mm_ans):
        flag = "" if b == t == m else "  <-- 不同"
        print(f"\n[{'○' if b else '×'}{'○' if t else '×'}{'○' if m else '×'}] {q}{flag}")
        if flag or not b:
            print(f"   基座: {ab[:90]!r}\n   文字: {at[:90]!r}\n   编码: {am[:90]!r}")
    print("\n== 推理显存峰值（bf16，torch 实际分配）")
    for k, v in mem.items():
        print(f"  {k:<28} {v:.2f} GB")


if __name__ == "__main__":
    main()
