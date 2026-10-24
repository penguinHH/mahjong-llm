"""生成 Colab notebook：LoRA SFT（免费版 T4，fp16，Drive 断点续训）。

    python make_notebook.py            # Qwen2.5-0.5B-Instruct，10 万条
    python make_notebook.py tinyswallow  # SakanaAI/TinySwallow-1.5B-Instruct，前 2 万条
"""
import json
import sys

CFGS = {
    "qwen25": dict(model="Qwen/Qwen2.5-0.5B-Instruct", title="Qwen2.5-0.5B", tag="qwen25-0.5b", extra="",
                   note="10 万条", max_samples="", val_size=1000, ipynb="qwen25_0_5b_colab.ipynb"),
    "tinyswallow": dict(model="SakanaAI/TinySwallow-1.5B-Instruct", title="TinySwallow-1.5B（Sakana AI）",
                        tag="tinyswallow-1.5b", note="前 2 万条（T4 上全量约 100 小时，先看 2 万条）",
                        extra="default_system: あなたは、Sakana AI株式会社が開発したTinySwallowです。小型ながら、誠実で優秀なアシスタントです。\n",
                        max_samples="max_samples: 20000\n", val_size=500, ipynb="tinyswallow_1_5b_colab.ipynb"),
}
C = CFGS[sys.argv[1] if len(sys.argv) > 1 else "qwen25"]

CELLS = []
T = C['tag']


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# Qwen2.5-0.5B 立直麻将 LoRA SFT（Colab 免费版）

和本地 `train_small/qwen25_0_5b_lora.yaml` 同一套配置（10 万条、每步 16 条、lr 1e-4、1 epoch、LoRA rank 8），只改了：
fp16（T4 不支持 bf16）、每 200 步存完整 checkpoint 到 Drive、同一条命令自动续训。

**用法**：运行时 → 更改运行时类型 → T4 GPU，然后"全部运行"。断线后重新打开，再"全部运行"一次即可从最新 checkpoint 继续。
训练数据来自天凤牌谱，**不要公开分享这个 notebook 和 Drive 文件夹**。""")

code("""import subprocess, sys
out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(out or "没有 GPU")
assert out, "没有分到 GPU：运行时 → 更改运行时类型 → T4 GPU"
GPU = out.split(",")[0]""")

code("""from google.colab import drive
drive.mount("/content/drive")
import os
ROOT = "/content/drive/MyDrive/mahjong_colab"          # 数据压缩包放在这里
for d in ["ckpt", "pred", "tokenized"]:
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
assert os.path.exists(f"{ROOT}/colab_data.tar.gz"), f"请把 colab_data.tar.gz 上传到 {ROOT}/"
print("Drive 就绪:", os.listdir(ROOT))""")

code("""%%bash
# LLaMA-Factory 固定到和本地相同的提交；不重装 Colab 自带的 torch
set -e
if [ ! -d /content/LLaMA-Factory ]; then
  git clone -q https://github.com/hiyouga/LLaMA-Factory.git /content/LLaMA-Factory
  git -C /content/LLaMA-Factory checkout -q 1fded918ac4723ac9966363f51ed4500866c5d1d
fi
cd /content/LLaMA-Factory
pip install -q -e . 2>&1 | grep -viE "already satisfied|warning" | tail -5 || true
pip install -q liger-kernel jieba nltk rouge-chinese 2>&1 | tail -1 || true
python -c "import llamafactory, transformers, peft, torch; print('llamafactory ok | transformers', transformers.__version__, '| peft', peft.__version__, '| torch', torch.__version__)"
""")

code("""%%bash
set -e
mkdir -p /content/data && cd /content/data
[ -f riichi_100k.jsonl ] || tar -xzf /content/drive/MyDrive/mahjong_colab/colab_data.tar.gz
wc -l riichi_100k.jsonl riichi_test5k.jsonl
cat > dataset_info.json <<'EOF'
{
  "riichi_100k":  {"file_name": "riichi_100k.jsonl",  "columns": {"prompt": "instruction", "response": "output"}},
  "riichi_test5k": {"file_name": "riichi_test5k.jsonl", "columns": {"prompt": "instruction", "response": "output"}}
}
EOF""")

code("""train_yaml = f'''
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
dataset_dir: /content/data
dataset: riichi_100k
template: qwen
cutoff_len: 4096
preprocessing_num_workers: 2
tokenized_path: {ROOT}/tokenized/riichi_100k_qwen25     # 分词结果存 Drive，重开会话不用再分词
output_dir: {ROOT}/ckpt/qwen25-0.5b-riichi_100k
logging_steps: 10
save_strategy: steps
save_steps: 200
save_only_model: false
plot_loss: true
overwrite_output_dir: false      # 目录里有 checkpoint 就自动续训
report_to: none
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true                       # T4 没有 bf16
enable_liger_kernel: true        # 融合输出层，省下 15 万词表的 logits 显存
torch_empty_cache_steps: 50
val_size: 1000
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 400
'''
open("/content/train.yaml", "w").write(train_yaml)
print(train_yaml)""")

code("""%%bash
# 训练（可反复运行：有 checkpoint 就续训；跑完后再运行会直接退出）
cd /content/LLaMA-Factory
if [ -f /content/drive/MyDrive/mahjong_colab/ckpt/qwen25-0.5b-riichi_100k/all_results.json ]; then
  echo "训练已完成"; cat /content/drive/MyDrive/mahjong_colab/ckpt/qwen25-0.5b-riichi_100k/all_results.json; exit 0
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
llamafactory-cli train /content/train.yaml 2>&1 | tee -a /content/drive/MyDrive/mahjong_colab/train.log | grep -vE "it/s\\]|s/it\\]"
""")

code("""predict_yaml = f'''
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
adapter_name_or_path: {ROOT}/ckpt/qwen25-0.5b-riichi_100k
trust_remote_code: true
stage: sft
do_predict: true
finetuning_type: lora
dataset_dir: /content/data
eval_dataset: riichi_test5k
template: qwen
cutoff_len: 4096
output_dir: {ROOT}/pred/qwen25-0.5b
overwrite_output_dir: true
per_device_eval_batch_size: 8
predict_with_generate: true
max_new_tokens: 32
do_sample: false
fp16: true
report_to: none
'''
open("/content/predict.yaml", "w").write(predict_yaml)""")

code("""%%bash
# 留出集 5000 条生成动作（训练完成后才会跑）
cd /content/LLaMA-Factory
R=/content/drive/MyDrive/mahjong_colab
[ -f $R/ckpt/qwen25-0.5b-riichi_100k/all_results.json ] || { echo "训练还没完成，先别预测"; exit 0; }
[ -f $R/pred/qwen25-0.5b/generated_predictions.jsonl ] && { echo "已有预测结果"; exit 0; }
llamafactory-cli train /content/predict.yaml 2>&1 | grep -vE "it/s\\]|s/it\\]" | tail -5
wc -l $R/pred/qwen25-0.5b/generated_predictions.jsonl""")

code("""# 一致率（与本地 train_small/eval_acc.py 相同的统计）
import collections, json, os, re
P = f"{ROOT}/pred/qwen25-0.5b/generated_predictions.jsonl"
if not os.path.exists(P):
    print("还没有预测结果")
else:
    def parse(text):
        m = re.search(r'\\{.*?\\}', text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))["action"].replace(" ", "")
        except Exception:
            return None
    n = fmt = hit = 0
    by = collections.defaultdict(lambda: [0, 0])
    for line in open(P, encoding="utf-8"):
        r = json.loads(line)
        gold, pred = parse(r["label"]), parse(r["predict"])
        n += 1; fmt += pred is not None
        t = gold.split(":")[0]; ok = pred == gold
        hit += ok; by[t][0] += ok; by[t][1] += 1
    print(f"样本 {n}  格式正确 {fmt / n:.2%}  整体一致率 {hit / n:.2%}")
    for t, (h, c) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {t:<10} {c:>6} 条  一致率 {h / c:.2%}")
    print("预测文件在 Drive:", P, "—— 下载回本地可做更细的分析")""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
REPL = [("Qwen/Qwen2.5-0.5B-Instruct", C["model"]),
        ("# Qwen2.5-0.5B 立直麻将", f"# {C['title']} 立直麻将"),
        ("（10 万条、", f"（{C['note']}、"),
        ("riichi_100k_qwen25", f"riichi_100k_{T}"),
        ("qwen25-0.5b", T),
        ("mahjong_colab/train.log", f"mahjong_colab/train_{T}.log"),
        ("val_size: 1000", f"val_size: {C['val_size']}"),
        ("template: qwen\n", "template: qwen\n" + C["extra"]),
        ("dataset: riichi_100k\n", "dataset: riichi_100k\n" + C["max_samples"])]
for c in nb["cells"]:
    for a, b in REPL:
        c["source"] = c["source"].replace(a, b)
    c["source"] = c["source"].splitlines(keepends=True)
out = __file__.replace("make_notebook.py", C["ipynb"])
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written", out, len(CELLS), "cells")
