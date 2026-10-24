"""把答案部分的 loss 拆到每个 token 上：格式 token vs 决策 token（动作值）。
用法：python token_loss.py <adapter 目录> [样本数]
CPU 上跑，不碰正在训练的 GPU。"""
import json
import math
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

torch.set_num_threads(6)
CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
# 必须用训练时同一个类（多模态版）：LoRA 挂在 model.language_model.layers.* 上，
# 用 AutoModelForCausalLM 会得到纯文本类 Qwen3_5ForCausalLM，层名对不上，补丁不会加载
base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
model = PeftModel.from_pretrained(base, CKPT).eval()
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen3_5_nothink"))

rows = [json.loads(l) for l in open(os.path.expanduser("~/mahjong-llm/work/sft/riichi_test5k.jsonl"))]
random.seed(0)
rows = random.sample(rows, N)

fmt_sum = dec_sum = 0.0
fmt_n = dec_n = 0
correct_first = 0
shown = 0
for r in rows:
    # 用 LLaMA-Factory 训练时同一个模板编码，保证逐 token 一致（transformers 自带模板会多一个空 <think> 块）
    p_ids, a_ids = template.encode_oneturn(tok, [{"role": "user", "content": r["instruction"]},
                                                 {"role": "assistant", "content": r["output"]}])
    ids = p_ids + a_ids
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x).logits[0]
    logp = torch.log_softmax(logits[:-1].float(), -1)
    resp = list(range(len(p_ids), len(ids)))
    # 答案里 "action": " 之后到 " 之前的 token 算"决策"，其余算"格式"
    text_so_far, in_value, parts = "", False, []
    for i in resp:
        piece = tok.decode([ids[i]])
        before = text_so_far
        text_so_far += piece
        loss = -logp[i - 1, ids[i]].item()
        start = before.count('"action": "') == 0 and text_so_far.count('"action": "') == 1
        is_dec = '"action": "' in before and before.count('"') < 4
        parts.append((piece, loss, is_dec))
        if is_dec:
            dec_sum += loss
            dec_n += 1
        else:
            fmt_sum += loss
            fmt_n += 1
    dec_tokens = [p for p in parts if p[2]]
    # 决策的第一个 token 模型是否预测对（贪心）
    i0 = next(i for i, p in zip(resp, parts) if p[2])
    correct_first += int(logp[i0 - 1].argmax().item() == ids[i0])
    if shown < 4:
        print("答案:", r["output"], " 每个 token 的 loss:",
              " ".join(f"[{p[0]!r}{'*' if p[2] else ''} {p[1]:.3f}]" for p in parts))
        shown += 1

n_tok = fmt_n + dec_n
print(f"\n{N} 条样本，答案部分共 {n_tok} 个 token（平均每条 {n_tok / N:.1f} 个），其中决策 token {dec_n} 个")
print(f"全部答案 token 的平均 loss（≈训练日志里的 loss）: {(fmt_sum + dec_sum) / n_tok:.4f}")
print(f"  格式 token 平均 loss: {fmt_sum / fmt_n:.4f}，占总 loss 的 {fmt_sum / (fmt_sum + dec_sum):.1%}")
print(f"  决策 token 平均 loss: {dec_sum / dec_n:.4f}，占总 loss 的 {dec_sum / (fmt_sum + dec_sum):.1%}"
      f"（对应正确 token 的几何平均概率 {math.exp(-dec_sum / dec_n):.1%}）")
print(f"  决策第一个 token 贪心预测正确率: {correct_first / N:.1%}")
