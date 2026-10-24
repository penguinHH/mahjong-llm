"""把训练/预测任务提交给已部署的 Modal 函数，提交完立刻返回（任务在云端独立运行，本地关机、断网都不影响）。

    modal deploy modal_train.py                                   # 代码或镜像改动后重新部署一次
    python launch.py ../train_small/qwen35_2b_distill.yaml A100-40GB
    modal app logs mahjong-llm                                    # 看实时日志；或 modal volume get mahjong-ckpt logs/<名字>.log

`modal run --detach` 的本地进程如果被杀（2026-10-19 13:37 电脑意外重启），云端那次调用也跟着停了，所以改用部署 + spawn。
训练配置里 overwrite_output_dir: false → 输出目录已有 checkpoint 就自动续训，重新提交同一个配置即可接着跑。
"""
import os
import sys

import modal

FN = {"A100-40GB": "run_a100_40", "A100-80GB": "run_a100_80", "H100": "run_h100"}


def main(cfg: str, gpu: str = "A100-40GB"):
    text = open(cfg, encoding="utf-8").read()
    for a, b in (("/home/user/mahjong-llm/work/sft", "/data/sft"),
                 ("/home/user/mahjong-llm/work/ckpt", "/ckpt/ckpt"),
                 ("/home/user/mahjong-llm/work/pred", "/ckpt/pred")):
        text = text.replace(a, b)
    assert "/home/user" not in text, "配置里还有本地路径"
    name = os.path.splitext(os.path.basename(cfg))[0]
    call = modal.Function.from_name("mahjong-llm", FN[gpu]).spawn(text, name)
    print(f"已提交 {name} on {gpu}，call id {call.object_id}")


if __name__ == "__main__":
    main(*sys.argv[1:])
