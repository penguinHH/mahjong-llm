"""第 2 步：局面编码器 + 投影层 + LLM（可加 LoRA）。换 LLM 只改模型名：投影层输出维度从 LLM 配置里读。

序列 = [对话模板前半段][35 个局面 token][对话模板后半段（含文字指令）][答案 + 结束符]，用 inputs_embeds 送进 LLM，
只对答案部分算 loss。对话模板用 tokenizer.apply_chat_template 渲染一个带占位符的用户消息再切开，Qwen3.5 / Qwen2.5 /
TinySwallow 都通用（Qwen3.5 用 enable_thinking=False，与之前 LLaMA-Factory 的 qwen3_5_nothink 一致）。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from encoder import make_encoder

PH = "<<STATE>>"


class Projector(nn.Module):
    """编码器的 192 维向量 → LLM 词向量空间。外加每个位置一个可学习的"槽位"向量（区分 34 种牌 + 全局）。"""

    def __init__(self, ch, hidden, n_tokens=35):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ch, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.slot = nn.Parameter(torch.zeros(n_tokens, hidden))

    def forward(self, tiles, glob):
        return self.mlp(torch.cat([tiles, glob[:, None]], 1)) + self.slot


class MahjongMM(nn.Module):
    def __init__(self, llm_name, enc_ckpt, dtype=torch.bfloat16):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype)
        ck = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
        self.enc = make_encoder(ck["enc_cfg"])
        self.enc.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("enc.")})
        self.proj = Projector(ck["enc_cfg"]["ch"], self.llm.config.hidden_size)
        kw = {"enable_thinking": False} if "qwen3" in self.llm.config.model_type else {}
        tmpl = self.tok.apply_chat_template([{"role": "user", "content": PH + "\n{instruction}"}],
                                            tokenize=False, add_generation_prompt=True, **kw)
        self.prefix, self.suffix = tmpl.split(PH)
        self.prefix_ids = self.tok(self.prefix, add_special_tokens=False)["input_ids"]
        self.eos = self.tok.eos_token
        self.eos_id = self.tok.convert_tokens_to_ids(self.eos)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id

    def add_lora(self, r=8, alpha=16, dropout=0.0):
        from peft import LoraConfig, get_peft_model
        self.llm = get_peft_model(self.llm, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                                       target_modules="all-linear", task_type="CAUSAL_LM"))

    def state_embeds(self, planes, scalars):
        tiles, glob = self.enc(planes, scalars)
        return self.proj(tiles.float(), glob.float())

    def build(self, planes, scalars, instructions, answers=None, left_pad=False):
        """→ inputs_embeds [B, L, H]、attention_mask [B, L]、labels [B, L]（无答案时为 None）。"""
        dev = planes.device
        emb = self.llm.get_input_embeddings()
        st = self.state_embeds(planes, scalars).to(emb.weight.dtype)          # [B, 35, H]
        seqs, labs = [], []
        pre = emb(torch.tensor(self.prefix_ids, device=dev))
        for i, ins in enumerate(instructions):
            suf_ids = self.tok(self.suffix.replace("{instruction}", ins), add_special_tokens=False)["input_ids"]
            ans_ids = self.tok(answers[i] + self.eos, add_special_tokens=False)["input_ids"] if answers else []
            ids = torch.tensor(suf_ids + ans_ids, device=dev)
            seqs.append(torch.cat([pre, st[i], emb(ids)], 0))
            n_ctx = len(self.prefix_ids) + st.shape[1] + len(suf_ids)
            labs.append([-100] * n_ctx + ans_ids)
        L = max(s.shape[0] for s in seqs)
        L = -(-L // 32) * 32          # 补齐到 32 的倍数：长度种类少了，线性注意力的 Triton 核不用每批重新调优
        H = seqs[0].shape[1]
        x = torch.zeros(len(seqs), L, H, dtype=seqs[0].dtype, device=dev)
        mask = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
        lab = torch.full((len(seqs), L), -100, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            n = s.shape[0]
            sl = slice(L - n, L) if left_pad else slice(0, n)
            x[i, sl] = s
            mask[i, sl] = 1
            lab[i, sl] = torch.tensor(labs[i], device=dev)
        return x, mask, (lab if answers else None)

    def forward(self, planes, scalars, instructions, answers):
        """只在答案位置过 lm_head 算 loss（与 HF 的 labels 损失等价）：词表 24.8 万，整段 logits 转 fp32 要好几 GB，
        指令里加了事实文字（--facts）后序列翻倍，会爆 16GB 显存。"""
        x, mask, lab = self.build(planes, scalars, instructions, answers)
        base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        h = base.model(inputs_embeds=x, attention_mask=mask).last_hidden_state
        tgt = lab[:, 1:]
        sel = tgt != -100
        logits = base.get_output_embeddings()(h[:, :-1][sel]).float()
        return torch.nn.functional.cross_entropy(logits, tgt[sel])

    @torch.no_grad()
    def generate(self, planes, scalars, instructions, max_new_tokens=128):
        x, mask, _ = self.build(planes, scalars, instructions, None, left_pad=True)
        out = self.llm.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=max_new_tokens,
                                do_sample=False, eos_token_id=self.eos_id, pad_token_id=self.pad_id)
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def trainable_state(self):
        """只存训练过的部分：投影层 + LoRA（编码器和 LLM 本体不变，不用存）。"""
        sd = {f"proj.{k}": v for k, v in self.proj.state_dict().items()}
        for n, p in self.llm.named_parameters():
            if p.requires_grad:
                sd[f"llm.{n}"] = p.detach().cpu()
        return sd
