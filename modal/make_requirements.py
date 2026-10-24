"""从本地训练环境的 uv pip freeze（requirements-train.lock）生成 Modal 镜像用的 requirements-modal.txt。

去掉：本地可编辑安装的 LLaMA-Factory（镜像里按同一提交单独装）、causal-conv1d（要编译，单独装）、
和训练无关或版本互相冲突的包（torchaudio 2.11 钉死 torch 2.11，和 torch 2.14 冲突）、
nvidia-*（由 torch 的 cu130 轮子自己拉对应版本）。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = ("-e ", "causal-conv1d==", "torchaudio==", "torchdata==", "mahjong==", "nvidia-")
lines = [l.strip() for l in open(f"{HERE}/requirements-train.lock", encoding="utf-8")]
keep = [l for l in lines if l and not l.startswith(SKIP)]
open(f"{HERE}/requirements-modal.txt", "w", encoding="utf-8").write("\n".join(keep) + "\n")
print(f"{len(keep)} / {len(lines)} 个包写入 requirements-modal.txt")
