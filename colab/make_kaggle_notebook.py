"""生成 Kaggle notebook：Qwen2.5-0.5B-Instruct LoRA SFT（T4 ×2 双卡并行，fp16，跨会话接力续训）。

和 Colab 版 / 本地 qwen25_0_5b_lora.yaml 同一套超参；每步仍是 16 条（2 卡 × 每卡 2 条 × 累积 4 次）。
Kaggle 的 "Save & Run All" 必须 12 小时内跑完，否则这个版本作废、输出也不保存 →
训练用 timeout 11 小时截断，checkpoint 留在 /kaggle/working；下次把上一版的输出加为输入就会接着训。

    python make_kaggle_notebook.py          # 人类标签（riichi_100k）→ qwen25_0_5b_kaggle.ipynb
    python make_kaggle_notebook.py mortal   # Mortal 标签（riichi_100k_mortal，硬标签蒸馏）→ qwen25_0_5b_mortal_kaggle.ipynb
"""
import json
import sys

MORTAL = len(sys.argv) > 1 and sys.argv[1] == "mortal"

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# Qwen2.5-0.5B 立直麻将 LoRA SFT（Kaggle，T4 ×2）

**设置**：Accelerator = GPU T4 x2，Internet = On；Input 加私有数据集 `riichi-sft-data`
（riichi_100k_a/b.jsonl.gz + riichi_test5k.jsonl.gz）。
**运行**：Save Version → Save & Run All（后台跑，可关浏览器）。一次最多 11 小时训练，没训完就把这一版的 Output
加为下一版的 Input，再 Save & Run All，会自动从最新 checkpoint 接着训。
训练数据来自天凤牌谱，**notebook 和数据集都保持私有**。""")

code("""import glob, os, shutil, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
W = "/kaggle/working"
CKPT = f"{W}/ckpt/qwen25-0.5b-riichi_100k"
os.makedirs(f"{W}/ckpt", exist_ok=True)

# 数据：上传时拆成 3 个 gz（单次上传上限 10MB）；Kaggle 可能已自动解压，也可能没有
import gzip
os.makedirs("/kaggle/tmp/data", exist_ok=True)
def part(name):
    for p in glob.glob(f"/kaggle/input/**/{name}*", recursive=True):
        if os.path.isfile(p):
            return gzip.open(p, "rb") if p.endswith(".gz") else open(p, "rb")
    raise FileNotFoundError(f"Input 里找不到 {name}：请把私有数据集 riichi-sft-data 加为 Input")
with open("/kaggle/tmp/data/riichi_100k.jsonl", "wb") as fo:      # a + b 按原顺序拼回
    for n in ("riichi_100k_a.jsonl", "riichi_100k_b.jsonl"):
        shutil.copyfileobj(part(n), fo)
with open("/kaggle/tmp/data/riichi_test5k.jsonl", "wb") as fo:
    shutil.copyfileobj(part("riichi_test5k.jsonl"), fo)
import hashlib
md5 = hashlib.md5(open("/kaggle/tmp/data/riichi_100k.jsonl", "rb").read()).hexdigest()
assert md5 == "4c838b3168f3eb205d297d5c1b3d4ceb", f"拼回的训练集和本地不一致：{md5}"
json.dump({n: {"file_name": f"{n}.jsonl", "columns": {"prompt": "instruction", "response": "output"}}
           for n in ("riichi_100k", "riichi_test5k")}, open("/kaggle/tmp/data/dataset_info.json", "w"))
print(subprocess.run("wc -l /kaggle/tmp/data/*.jsonl", shell=True, capture_output=True, text=True).stdout)

# 接力：上一版输出（作为 Input 挂进来）里有 checkpoint 就拷到 working 继续
if not glob.glob(f"{CKPT}/checkpoint-*") and not os.path.exists(f"{CKPT}/all_results.json"):
    prev = sorted(glob.glob("/kaggle/input/**/ckpt/qwen25-0.5b-riichi_100k", recursive=True))
    if prev:
        shutil.copytree(prev[0], CKPT, dirs_exist_ok=True)
        print("从上一版接续：", sorted(os.listdir(CKPT)))
print("当前 checkpoint：", sorted(glob.glob(f"{CKPT}/checkpoint-*")))""".replace("import glob, os, shutil, subprocess", "import glob, json, os, shutil, subprocess"))

code("""%%bash
# LLaMA-Factory 固定到和本地相同的提交（需要 Internet = On）
set -e
if [ ! -d /kaggle/tmp/LLaMA-Factory ]; then
  git clone -q https://github.com/hiyouga/LLaMA-Factory.git /kaggle/tmp/LLaMA-Factory
  git -C /kaggle/tmp/LLaMA-Factory checkout -q 1fded918ac4723ac9966363f51ed4500866c5d1d
fi
cd /kaggle/tmp/LLaMA-Factory
pip install -q -e . 2>&1 | grep -viE "already satisfied|warning|requires|incompatible" | tail -3 || true
pip install -q liger-kernel jieba nltk rouge-chinese 2>&1 | tail -1 || true
python -c "import llamafactory, transformers, peft, torch; print('llamafactory ok | transformers', transformers.__version__, '| peft', peft.__version__, '| torch', torch.__version__, '| GPUs', torch.cuda.device_count())"
""")

code("""open("/kaggle/tmp/train.yaml", "w").write(f'''
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
dataset_dir: /kaggle/tmp/data
dataset: riichi_100k
template: qwen
cutoff_len: 4096
preprocessing_num_workers: 4
output_dir: {CKPT}
logging_steps: 10
save_strategy: steps
save_steps: 200
save_only_model: false
save_total_limit: 3              # 只留最近 3 个 checkpoint，控制输出大小
plot_loss: true
overwrite_output_dir: false      # 有 checkpoint 就自动续训
report_to: none
per_device_train_batch_size: 2
gradient_accumulation_steps: 4   # 2 卡 × 2 × 4 = 每步 16 条，与单卡 2 × 8 相同
ddp_find_unused_parameters: false
learning_rate: 1.0e-4
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
enable_liger_kernel: true
torch_empty_cache_steps: 50
val_size: 1000
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 400
''')""")

code("""# 训练：双卡 torchrun；11 小时后自动停（Kaggle 版本必须 12 小时内结束才能保存输出）
# 用 Python 读子进程输出：%%bash 单元的输出要等整个单元结束才进 Kaggle 日志页，Python 的 print 是实时的
import re, time
if os.path.exists(f"{CKPT}/all_results.json"):
    print("训练已完成", open(f"{CKPT}/all_results.json").read())
else:
    env = dict(os.environ, FORCE_TORCHRUN="1", PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = ["timeout", "--signal=INT", "--kill-after=10m", "11h", "llamafactory-cli", "train", "/kaggle/tmp/train.yaml"]
    keep = re.compile(r"'loss'|'eval_loss'|Traceback|Error|out of memory|Saving model checkpoint|train_runtime|trainable params")
    bar = re.compile(r"\\d+/\\d+ \\[")
    last_bar = 0.0
    with open(f"{W}/train.log", "ab") as log:
        p = subprocess.Popen(cmd, cwd="/kaggle/tmp/LLaMA-Factory", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        buf = b""
        while True:
            chunk = p.stdout.read1(65536)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            parts = re.split(rb"[\\r\\n]", buf + chunk)
            buf = parts.pop()
            for s in parts:
                t = s.decode("utf-8", "replace").strip()
                if keep.search(t):
                    print(t[:300], flush=True)
                elif bar.search(t) and time.time() - last_bar > 600:     # 进度条每 10 分钟打一次
                    print(t[:200], flush=True)
                    last_bar = time.time()
        rc = p.wait()
    print(f"训练进程退出码 {rc}（124/130 = 到 11 小时被截断，下一版接着训）")
    print("checkpoint：", sorted(glob.glob(f"{CKPT}/checkpoint-*"), key=lambda s: int(s.rsplit("-", 1)[1]))[-3:])""")

code("""%%bash
# 训练完成后才预测留出集 5000 条（单卡即可）
CKPT=/kaggle/working/ckpt/qwen25-0.5b-riichi_100k
[ -f $CKPT/all_results.json ] || { echo "训练还没完成：把这一版 Output 加为下一版 Input 再 Save & Run All"; exit 0; }
cat > /kaggle/tmp/predict.yaml <<EOF
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
adapter_name_or_path: $CKPT
trust_remote_code: true
stage: sft
do_predict: true
finetuning_type: lora
dataset_dir: /kaggle/tmp/data
eval_dataset: riichi_test5k
template: qwen
cutoff_len: 4096
output_dir: /kaggle/working/pred/qwen25-0.5b
overwrite_output_dir: true
per_device_eval_batch_size: 8
predict_with_generate: true
max_new_tokens: 32
do_sample: false
fp16: true
report_to: none
EOF
cd /kaggle/tmp/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train /kaggle/tmp/predict.yaml > /kaggle/working/predict.log 2>&1
wc -l /kaggle/working/pred/qwen25-0.5b/generated_predictions.jsonl""")

code("""# 一致率（与本地 train_small/eval_acc.py 相同的统计）
import collections, re
P = "/kaggle/working/pred/qwen25-0.5b/generated_predictions.jsonl"
if os.path.exists(P):
    def parse(text):
        m = re.search(r'\\{.*?\\}', text, re.S)
        try:
            return json.loads(m.group(0))["action"].replace(" ", "") if m else None
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
else:
    print("还没有预测结果")""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
REPL = [("立直麻将 LoRA SFT（Kaggle", "立直麻将 Mortal 蒸馏（硬标签）LoRA SFT（Kaggle"),
        ("Input 加私有数据集 `riichi-sft-data`\n（riichi_100k_a/b.jsonl.gz + riichi_test5k.jsonl.gz）",
         "Input 加私有数据集 `riichi-sft-mortal`（riichi_100k_mortal_a/b.jsonl.gz，答案是 Mortal 首选）\n"
         "和 `riichi-sft-data`（只用其中的 riichi_test5k.jsonl.gz，标签是牌谱动作；和 Mortal 的一致率在本地用 eval_vs_mortal.py 算）"),
        ("请把私有数据集 riichi-sft-data 加为 Input", "请把私有数据集 riichi-sft-mortal 和 riichi-sft-data 加为 Input"),
        ("qwen25-0.5b-riichi_100k", "qwen25-0.5b-riichi_100k-mortal"),
        ('("riichi_100k_a.jsonl", "riichi_100k_b.jsonl")', '("riichi_100k_mortal_a.jsonl", "riichi_100k_mortal_b.jsonl")'),
        ("/kaggle/tmp/data/riichi_100k.jsonl", "/kaggle/tmp/data/riichi_100k_mortal.jsonl"),
        ("4c838b3168f3eb205d297d5c1b3d4ceb", "8404247173728127d08ba11aff2a5638"),
        ('for n in ("riichi_100k", "riichi_test5k")', 'for n in ("riichi_100k_mortal", "riichi_test5k")'),
        ("dataset: riichi_100k\n", "dataset: riichi_100k_mortal\n"),
        ("pred/qwen25-0.5b", "pred/qwen25-0.5b-mortal")] if MORTAL else []
for c in nb["cells"]:
    for a, b in REPL:
        c["source"] = c["source"].replace(a, b)
    c["source"] = c["source"].splitlines(keepends=True)
out = __file__.replace("make_kaggle_notebook.py", "qwen25_0_5b_mortal_kaggle.ipynb" if MORTAL else "qwen25_0_5b_kaggle.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written", out, len(CELLS), "cells")
