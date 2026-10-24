"""在 Modal 上跑 LLaMA-Factory（训练 / 预测），配置直接用 train_small/ 里的 yaml。

    # 一次性：上传数据（只放私有 Volume，不公开）
    modal volume put mahjong-data ~/mahjong-llm/work/sft/riichi_100k_mortal.jsonl /sft/riichi_100k_mortal.jsonl
    # 测速（30 步）
    modal run modal_train.py --cfg ../train_small/qwen35_2b_distill_bench.yaml --gpu A100-40GB
    # 正式训练（--detach：本地断开也继续跑；checkpoint 每 5 分钟提交到 Volume，中断后同一条命令续训）
    modal run --detach modal_train.py --cfg ../train_small/qwen35_2b_distill.yaml --gpu A100-40GB
    # 取回预测
    modal volume get mahjong-ckpt pred/qwen35-2b-mortal/generated_predictions.jsonl .

环境与本地 venv-train 相同（requirements-modal.txt 由 make_requirements.py 从本地 freeze 生成；LLaMA-Factory 同一提交）。
本地路径 /home/user/mahjong-llm/work/{sft,ckpt,pred} 在容器里换成 /data/sft、/ckpt/ckpt、/ckpt/pred。
训练数据来自天凤牌谱：Volume 保持私有，训出来的权重不公开、不上传到任何公开位置。
"""
import os
import subprocess
import threading
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
LF_COMMIT = "1fded918ac4723ac9966363f51ed4500866c5d1d"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install_from_requirements(f"{HERE}/requirements-modal.txt", extra_index_url="https://download.pytorch.org/whl/cu130")
    # LLaMA-Factory 的 mm_plugin 会 import torchaudio；本地也是 2.11.0+cu130 配 torch 2.14（能导入），不带依赖装，免得拉低 torch
    .pip_install("torchaudio==2.11.0+cu130", extra_index_url="https://download.pytorch.org/whl/cu130", extra_options="--no-deps")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0;9.0", "MAX_JOBS": "8", "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++"})
    # Qwen3.5 线性注意力层的短卷积用 causal-conv1d 的 CUDA 核；装不上时 transformers 会退回纯 torch 实现（慢）
    .run_commands("pip install --no-build-isolation causal-conv1d==1.7.0 || echo 'causal-conv1d 未装上，退回 torch 实现'")
    .run_commands(f"git clone -q https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory"
                  f" && git -C /opt/LLaMA-Factory checkout -q {LF_COMMIT}"
                  f" && pip install --no-deps -e /opt/LLaMA-Factory")
    .env({"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"})
)
data = modal.Volume.from_name("mahjong-data", create_if_missing=True)
ckpt = modal.Volume.from_name("mahjong-ckpt", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("mahjong-llm", image=image)
VOLS = {"/data": data, "/ckpt": ckpt, "/hf": hf}


def _run(yaml_text: str, name: str) -> int:
    os.makedirs("/ckpt/logs", exist_ok=True)
    cfg = f"/tmp/{name}.yaml"
    open(cfg, "w").write(yaml_text)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout, flush=True)
    stop = threading.Event()

    def committer():          # 定期把 checkpoint / 模型缓存提交到 Volume，容器意外退出也不丢
        while not stop.wait(300):
            for v in (ckpt, hf):
                try:
                    v.commit()
                except Exception as e:
                    print("commit 失败：", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    with open(f"/ckpt/logs/{name}.log", "ab") as log:
        p = subprocess.Popen(["llamafactory-cli", "train", cfg], cwd="/opt/LLaMA-Factory",
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(p.stdout.readline, b""):
            log.write(line)
            log.flush()
            print(line.decode("utf-8", "replace"), end="", flush=True)
        rc = p.wait()
    stop.set()
    ckpt.commit()
    hf.commit()
    print(f"llamafactory-cli 退出码 {rc}", flush=True)
    return rc


# CPU / 内存另外按秒计费（8 核 + 64GiB 每小时约 $0.89）；4 核 + 32GiB 足够（分词多花几分钟），每小时约 $0.45
KW = dict(volumes=VOLS, timeout=20 * 3600, cpu=4, memory=32768)


@app.function(gpu="A100-40GB", **KW)
def run_a100_40(yaml_text: str, name: str) -> int:
    return _run(yaml_text, name)


@app.function(gpu="A100-80GB", **KW)
def run_a100_80(yaml_text: str, name: str) -> int:
    return _run(yaml_text, name)


@app.function(gpu="H100", **KW)
def run_h100(yaml_text: str, name: str) -> int:
    return _run(yaml_text, name)


@app.local_entrypoint()
def main(cfg: str, gpu: str = "A100-40GB"):
    text = open(cfg, encoding="utf-8").read()
    for a, b in (("/home/user/mahjong-llm/work/sft", "/data/sft"),
                 ("/home/user/mahjong-llm/work/ckpt", "/ckpt/ckpt"),
                 ("/home/user/mahjong-llm/work/pred", "/ckpt/pred")):
        text = text.replace(a, b)
    assert "/home/user" not in text, "配置里还有本地路径"
    name = os.path.splitext(os.path.basename(cfg))[0]
    fn = {"A100-40GB": run_a100_40, "A100-80GB": run_a100_80, "H100": run_h100}[gpu]
    t = time.time()
    rc = fn.remote(text, name)
    print(f"{name} on {gpu}: 退出码 {rc}，用时 {(time.time() - t) / 60:.1f} 分钟")
