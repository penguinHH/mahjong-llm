import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

torch.set_num_threads(6)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen3_5_nothink"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
model = PeftModel.from_pretrained(base, sys.argv[1]).eval()
print("model class:", type(base).__name__)

r = json.loads(open(os.path.expanduser("~/mahjong-llm/work/sft/riichi_test5k.jsonl")).readline())
p_ids, a_ids = template.encode_oneturn(tok, [{"role": "user", "content": r["instruction"]},
                                             {"role": "assistant", "content": r["output"]}])
print("prompt 末尾:", repr(tok.decode(p_ids[-8:])), "| 答案开头:", repr(tok.decode(a_ids[:4])))
print("prompt 开头:", repr(tok.decode(p_ids[:6])))
ids = p_ids + a_ids
with torch.no_grad():
    out = model(torch.tensor([ids]))
logits = out.logits[0]
print("logits 形状:", tuple(logits.shape), "输入长度:", len(ids))
pos = len(p_ids) - 1                      # 预测答案第一个 token 的位置
top = torch.topk(torch.log_softmax(logits[pos].float(), -1), 5)
print("答案第一个位置 top5:", [(repr(tok.decode([i])), round(v.item(), 2)) for v, i in zip(top.values, top.indices)])
# 再看看纯生成：模型自己会接什么
gen = model.generate(torch.tensor([p_ids]), max_new_tokens=12, do_sample=False)
print("贪心生成:", repr(tok.decode(gen[0][len(p_ids):])))
