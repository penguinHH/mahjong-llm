"""三合一的训练数据：同一批 10 万条局面，每条样本随机选输入模式和任务。

输入模式：state（只局面）/ image（只图片）/ both（图片 + 局面），概率由 --modes 指定，训完缺哪种输入都能用（模态丢弃）。
任务：decide（理由 + Mortal 动作，或 --label human 时只要牌谱动作）/ caption（"看图念局面"，答案 = captions.caption）。
caption 任务只在有图片时出（训练视觉塔读准牌桌；只给局面时念局面是 mmj 对齐阶段的事）。
数据来源：局面张量 work/mmj/train_*（与 riichi_100k 行对齐）、图片 work/vlm/img/train/<行号>.png、牌谱动作 sft/riichi_100k.jsonl。
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mmj"))   # 放最后：mmj 也有 model.py
from captions import CAPTION_INSTRUCTION, caption  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work")
INS_REASON = "以下の局面で最善の行動を選び、理由とともにJSONで答えてください。合法手: "
INS_ACTION = "以下の局面で最善の行動を選び、JSONで答えてください。合法手: "


class TriData:
    def __init__(self, split="train", label="mortal", modes=(0.3, 0.4, 0.3), p_caption=0.2, seed=0):
        self.split, self.label = split, label
        self.planes = np.load(f"{W}/mmj/{split}_planes.npy")
        self.scalars = np.load(f"{W}/mmj/{split}_scalars.npy")
        text = [json.loads(l) for l in open(f"{W}/mmj/{split}_text.jsonl", encoding="utf-8")]
        self.legal = [t["instruction"].split("合法手: ", 1)[1] for t in text]
        if split == "train":
            self.reason = [t["output"] for t in text]                                   # 理由 + Mortal 动作
            self.human = [json.loads(l)["output"] for l in open(f"{W}/sft/riichi_100k.jsonl", encoding="utf-8")]
        else:
            self.reason = None
            self.human = [t["output"] for t in text]                                    # 测试集：牌谱动作
        self.modes = np.array(modes) / np.sum(modes)
        self.p_caption = p_caption
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.legal)

    def image(self, i):
        return Image.open(f"{W}/vlm/img/{self.split}/{i:06d}.png").convert("RGB")

    def item(self, i, mode=None, task=None):
        mode = mode or self.rng.choice(["state", "image", "both"], p=self.modes)
        if task is None:
            task = "caption" if mode != "state" and self.rng.random() < self.p_caption else "decide"
        has_img, has_state = mode in ("image", "both"), mode in ("state", "both")
        if task == "caption":
            ins, ans = CAPTION_INSTRUCTION, caption(self.planes[i], self.scalars[i])
        elif self.label == "human":
            ins, ans = INS_ACTION + self.legal[i], self.human[i]
        else:   # Mortal 标签版：指令要求写理由；测试集没有理由，答案只用来对动作
            ins, ans = INS_REASON + self.legal[i], self.reason[i] if self.reason is not None else self.human[i]
        return {"instruction": ins, "answer": ans, "mode": mode, "task": task,
                "image": self.image(i) if has_img else None,
                "planes": self.planes[i] if has_state else None,
                "scalars": self.scalars[i] if has_state else None}
