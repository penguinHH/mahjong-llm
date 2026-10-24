"""三合一：Qwen3.5 视觉塔 + 局面编码器 + LLM（同一个模型，三种输入：只图片 / 只局面 / 图片 + 局面）。

  图片   → Qwen3.5 自带视觉塔 + 视觉投影层（merger）→ 768 个图片 token（Qwen 原生流程，位置编码不受影响）
  局面   → 局面编码器（mmj 第 0 步训好，冻结）→ 局面投影层 → 35 个局面 token
  文字   → 分词
局面 token 的放法：文字里放 35 个占位 token（<|quad_start|>，平时不会出现），在词嵌入层挂钩子，
把这些位置的向量换成局面投影层的输出。占位 token 在位置编码里就是普通文字 token。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mmj"))   # 放最后：mmj 也有 model.py
from encoder import make_encoder  # noqa: E402


class Projector(nn.Module):
    """与 mmj/model.py 的局面投影层同结构、同参数名（不能直接 import：两个文件都叫 model.py），可直接载入对齐好的权重。"""

    def __init__(self, ch, hidden, n_tokens=35):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ch, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.slot = nn.Parameter(torch.zeros(n_tokens, hidden))

    def forward(self, tiles, glob):
        return self.mlp(torch.cat([tiles, glob[:, None]], 1)) + self.slot


STATE_TOKEN = "<|quad_start|>"
N_STATE = 35
LM_LINEAR = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"


class MahjongTri(nn.Module):
    def __init__(self, llm_name, enc_ckpt, max_pixels=1024 * 768, dtype=torch.bfloat16):
        super().__init__()
        self.proc = AutoProcessor.from_pretrained(llm_name)
        self.tok = self.proc.tokenizer
        self.max_pixels = max_pixels
        self.vlm = AutoModelForImageTextToText.from_pretrained(llm_name, dtype=dtype)
        ck = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
        self.enc = make_encoder(ck["enc_cfg"])
        self.enc.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("enc.")})
        self.proj = Projector(ck["enc_cfg"]["ch"], self.vlm.config.text_config.hidden_size, N_STATE)
        self.state_id = self.tok.convert_tokens_to_ids(STATE_TOKEN)
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id
        self._state = None          # 当前 batch 的局面向量（钩子里用）
        self.vlm.get_input_embeddings().register_forward_hook(self._inject)

    # ---------------------------------------------------------------- 局面 token 注入
    def _inject(self, module, inputs, output):
        ids = inputs[0]
        mask = ids == self.state_id
        if self._state is None or not mask.any():
            return output
        out = output.clone()
        out[mask] = self._state.to(out.dtype).reshape(-1, out.shape[-1])   # 按样本顺序、每个 35 个
        return out

    def add_lora(self, r=8, alpha=16, visual=False):
        """LoRA 加在 LLM 的所有线性层；visual=True 时视觉塔的线性层也加。"""
        from peft import LoraConfig, get_peft_model
        pat = rf".*language_model.*\.{LM_LINEAR}"
        if visual:
            pat = rf"({pat})|(.*visual\.blocks.*\.(qkv|proj|linear_fc1|linear_fc2))"
        self.vlm = get_peft_model(self.vlm, LoraConfig(r=r, lora_alpha=alpha, target_modules=pat, lora_dropout=0.0))

    # ---------------------------------------------------------------- 组 batch
    def prompt_text(self, instruction, has_image, has_state):
        content = ([{"type": "image"}] if has_image else []) + \
                  [{"type": "text", "text": (STATE_TOKEN * N_STATE + "\n" if has_state else "") + instruction}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)

    def collate(self, items, with_answer=True, left_pad=False, device="cuda"):
        """items: dict(instruction, image(PIL 或 None), planes, scalars(或 None), answer)。"""
        rows, pix, thw = [], [], []
        for it in items:
            text = self.prompt_text(it["instruction"], it["image"] is not None, it["planes"] is not None)
            enc = self.proc(text=[text], images=[it["image"]] if it["image"] is not None else None,
                            return_tensors="pt", images_kwargs={"max_pixels": self.max_pixels})
            ids = enc["input_ids"][0].tolist()
            tt = enc["mm_token_type_ids"][0].tolist() if "mm_token_type_ids" in enc else [0] * len(ids)
            lab = [-100] * len(ids)
            if with_answer:
                ans = self.tok(it["answer"], add_special_tokens=False)["input_ids"] + [self.eos_id]
                ids, tt, lab = ids + ans, tt + [0] * len(ans), lab + ans
            rows.append((ids, tt, lab))
            if it["image"] is not None:
                pix.append(enc["pixel_values"])
                thw.append(enc["image_grid_thw"])
        L = max(len(r[0]) for r in rows)
        B = len(rows)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        types = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        for b, (ids, tt, lab) in enumerate(rows):
            n = len(ids)
            sl = slice(L - n, L) if left_pad else slice(0, n)
            input_ids[b, sl] = torch.tensor(ids)
            mask[b, sl] = 1
            types[b, sl] = torch.tensor(tt)
            labels[b, sl] = torch.tensor(lab)
        batch = {"input_ids": input_ids, "attention_mask": mask, "mm_token_type_ids": types}
        if pix:
            batch["pixel_values"] = torch.cat(pix)
            batch["image_grid_thw"] = torch.cat(thw)
        batch = {k: v.to(device) for k, v in batch.items()}
        st = [i for i, it in enumerate(items) if it["planes"] is not None]
        planes = torch.stack([torch.as_tensor(items[i]["planes"]) for i in st]).to(device) if st else None
        scalars = torch.stack([torch.as_tensor(items[i]["scalars"]) for i in st]).to(device) if st else None
        return batch, (labels.to(device) if with_answer else None), planes, scalars

    def set_state(self, planes, scalars):
        if planes is None:
            self._state = None
            return
        with torch.no_grad():
            tiles, glob = self.enc(planes, scalars)
        self._state = self.proj(tiles.float(), glob.float())                # [n_state_samples, 35, H]

    def forward(self, items, device="cuda"):
        """只在答案位置算 logits（与 mmj/model.py 相同的省显存写法）：带图时序列约 900 个 token，
        整段过 24.8 万词表的 lm_head 会把 16GB 显存撑爆（实测 CUDA "device not ready / Failed to create GPU mapping"）。"""
        batch, labels, planes, scalars = self.collate(items, True, device=device)
        self.set_state(planes, scalars)
        base = self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm
        h = base.model(**batch).last_hidden_state
        tgt = labels[:, 1:]
        sel = tgt != -100
        logits = base.lm_head(h[:, :-1][sel]).float()
        self._state = None
        return F.cross_entropy(logits, tgt[sel])

    @torch.no_grad()
    def generate(self, items, max_new_tokens=128, device="cuda"):
        batch, _, planes, scalars = self.collate(items, False, left_pad=True, device=device)
        self.set_state(planes, scalars)
        out = self.vlm.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                eos_token_id=self.eos_id, pad_token_id=self.pad_id)
        self._state = None
        return self.tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)

    def trainable_state(self):
        sd = {f"proj.{k}": v for k, v in self.proj.state_dict().items()}
        for n, p in self.vlm.named_parameters():
            if p.requires_grad:
                sd[f"vlm.{n}"] = p.detach().cpu()
        return sd
