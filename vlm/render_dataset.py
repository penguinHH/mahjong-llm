#!/usr/bin/env python3
"""批量渲染 VLM 训练 / 测试图片，并生成 LLaMA-Factory 的多模态数据集（sharegpt 格式）。

    ~/mahjong-llm/venv-train/bin/python -I /mnt/d/mahjong-llm/vlm/render_dataset.py

图片：work/vlm/img/{train,test}/<行号>.png（训练 = riichi_100k 的 10 万条，测试 = riichi_test5k 的 5000 条，行号对齐）。
同一批图片配两套标签（与文字版 / 编码器版同一批局面，可直接比较）：
  vlm_human_train.jsonl        牌谱里人的动作，只输出 {"action": ...}
  vlm_mortal_reason_train.jsonl Mortal 的动作 + 程序生成的事实理由（与 mmj 第 2 步同一套目标）
  vlm_test_action.jsonl / vlm_test_reason.jsonl  测试集（标签 = 牌谱动作；指令分别对应两种训练）
文字侧只有一句指令 + 合法动作（与 mmj 的短指令相同），局面全部在图片里。
"""
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_board import fields, render  # noqa: E402

SFT = os.path.expanduser("~/mahjong-llm/work/sft")
OUT = os.path.expanduser("~/mahjong-llm/work/vlm")
INS_ACTION = "<image>以下の局面で最善の行動を選び、JSONで答えてください。合法手: "
INS_REASON = "<image>以下の局面で最善の行動を選び、理由とともにJSONで答えてください。合法手: "


def work(args):
    split, i, prompt = args
    path = f"{OUT}/img/{split}/{i:06d}.png"
    f = fields(prompt)
    if not os.path.exists(path):
        render(f).save(path, optimize=False)
    return path, json.dumps(f["Legal actions"], ensure_ascii=False)


def sample(img, ins, ans):
    return json.dumps({"messages": [{"role": "user", "content": ins}, {"role": "assistant", "content": ans}],
                       "images": [img]}, ensure_ascii=False)


def main():
    for s in ("train", "test"):
        os.makedirs(f"{OUT}/img/{s}", exist_ok=True)
    train = [json.loads(l) for l in open(f"{SFT}/riichi_100k.jsonl", encoding="utf-8")]
    reason = [json.loads(l) for l in open(f"{SFT}/riichi_100k_mortal_reason.jsonl", encoding="utf-8")]
    test = [json.loads(l) for l in open(f"{SFT}/riichi_test5k.jsonl", encoding="utf-8")]
    assert len(train) == len(reason) and all(a["instruction"] == b["instruction"] for a, b in zip(train, reason)), \
        "riichi_100k 与理由数据行不对齐"
    jobs = [("test", i, r["instruction"]) for i, r in enumerate(test)] + \
           [("train", i, r["instruction"]) for i, r in enumerate(train)]
    with mp.Pool(6) as pool:
        res = []
        for k, x in enumerate(pool.imap(work, jobs, chunksize=200)):
            res.append(x)
            if (k + 1) % 10000 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)
    te, tr = res[:len(test)], res[len(test):]
    with open(f"{OUT}/vlm_human_train.jsonl", "w", encoding="utf-8") as fh, \
            open(f"{OUT}/vlm_mortal_reason_train.jsonl", "w", encoding="utf-8") as fm:
        for (img, legal), h, m in zip(tr, train, reason):
            fh.write(sample(img, INS_ACTION + legal, h["output"]) + "\n")
            fm.write(sample(img, INS_REASON + legal, m["output"]) + "\n")
    with open(f"{OUT}/vlm_test_action.jsonl", "w", encoding="utf-8") as fa, \
            open(f"{OUT}/vlm_test_reason.jsonl", "w", encoding="utf-8") as fr:
        for (img, legal), r in zip(te, test):
            fa.write(sample(img, INS_ACTION + legal, r["output"]) + "\n")
            fr.write(sample(img, INS_REASON + legal, r["output"]) + "\n")
    cols = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
    json.dump({n: {"file_name": f"{n}.jsonl", **cols} for n in
               ("vlm_human_train", "vlm_mortal_reason_train", "vlm_test_action", "vlm_test_reason")},
              open(f"{OUT}/dataset_info.json", "w"), indent=2)
    print(f"完成：训练 {len(tr)} 张、测试 {len(te)} 张 → {OUT}")


if __name__ == "__main__":
    main()
