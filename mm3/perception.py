#!/usr/bin/env python3
"""感知模块：卓面图片 → 局面张量（58×34 平面 + 71 标量），再交给冻结的局面编码器做决策。

结构：Qwen3.5 视觉塔（基座冻结，挂 LoRA）→ merger 之前的 3072 个图块特征，按真实行列排回 48×64 网格
      → 卷积下采样到 24×32 → 展平 + MLP → 平面 logits / 标量（--head conv，默认；渲染画面布局固定）
  试过但不收敛的两版（保留以便对照）：
  --head dec --feat merged：merger 后 768 个 token + 35 个查询 token 的解码器，1000 步手牌整组全对 0%（DETR 式收敛慢）；
  --head pool：每个区域学一张位置权重图再汇聚，权重图初始近乎均匀、softmax 梯度极小，250 步 loss 不动。
损失：平面 BCE + 标量 SmoothL1（经冻结编码器的策略 KL 可选，默认关掉，见 --kl_w 的说明）
      + 逐图块监督（--dense）：每个 16×16 图块是哪张牌、属于哪个区域，标签由 vlm/render_labels.py 按渲染布局精确生成。
        只有汇总后的平面时，平面 BCE 卡在 0.158 左右（按频率猜 0.171），模型没真正学会认牌。
看不见的特征（手切平面、振听标量）不算损失、输入编码器时清零，与 mmj/ablate_unseen.py 一致。
评测：测试集逐格准确率、各组整组全对率、标量 MAE、编码器用预测局面 vs 真实局面的决策一致率；
      预测结果存成 test_pred.npz，供后面的端到端评测（图片 → 感知 → 编码器 → LLM）用。
数据：work/vlm/img/{split}/<行号>.png 与 work/mmj/{split}_planes.npy / _scalars.npy 按行号对齐。
中断后用同样的命令重跑会从 out/ckpt.pt 接着训。

    PY=~/mahjong-llm/venv-train/bin/python; P=/mnt/d/mahjong-llm/mm3/perception.py
    $PY -I $P --max_steps 20 --eval_n 64 --out /tmp/perc_smoke            # 冒烟
    $PY -I $P --out ~/mahjong-llm/work/mm3/perception_tfx                  # 正式
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                          # assemble.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mmj"))   # 放最后：mmj 也有 model.py
import assemble  # noqa: E402
import postprocess  # noqa: E402
from actions import legal_mask  # noqa: E402
from encoder import EncoderPolicy  # noqa: E402
from state_features import PLANES, SCALARS  # noqa: E402

W = os.path.expanduser("~/mahjong-llm/work")
N_TILE = 34
UNSEEN_PLANES = [i for i, n in enumerate(PLANES) if n.startswith("tedashi")]
UNSEEN_SCALARS = [i for i, n in enumerate(SCALARS) if n == "furiten"]
GROUPS = {
    "手牌": [i for i, n in enumerate(PLANES) if n.startswith("hand")],
    "牌河": [i for i, n in enumerate(PLANES) if n.startswith("kawa")],
    "副露": [i for i, n in enumerate(PLANES) if n.startswith("fuuro") or re.fullmatch(r"ankan\d", n)],
    "宝牌": [i for i, n in enumerate(PLANES) if n in ("dora_indicator", "dora")],
    "立直牌": [i for i, n in enumerate(PLANES) if n.startswith("riichi_tile")],
}
VISIBLE = [i for i in range(len(PLANES)) if i not in UNSEEN_PLANES]
# 逐图块标签（vlm/render_labels.py 生成）：kind 0-33 牌、34-36 赤五、37 背景；region 0 背景、1 手牌、2 摸牌、
# 3-6 四家牌河、7-10 四家副露、11 宝牌指示牌。背景格占大多数，损失里降权
N_KIND, N_REGION, BG_KIND = 38, 12, 37
N_TH = 121                         # 点数千位 0～120
SCORE_IDX = [SCALARS.index(f"score{p}") for p in range(4)]
# 四家点数文字所在的图块窗口（起始行, 起始列），每个 ROI_H×ROI_W 个图块，顺序 = 自家/下家/対面/上家。
# 由 render_board.py:103-113 的版面加 textbbox 算出：自家/対面的"風 点数"约 15px 高，占 2 行 × 6 列图块；
# 下家/上家的点数只有约 10px 高，挤在 1 行 × 3 列里。窗口四周各留 1 格余量
SCORE_ROI = [(25, 28), (23, 33), (13, 28), (23, 23)]
ROI_H, ROI_W = 3, 8


def score_targets(scalars):
    """标量里的点数（/50000）→ 千位、百位两组类别 [B, 4]。"""
    v = torch.round(scalars[:, SCORE_IDX].float() * 50000).long().clamp(0, N_TH * 1000 - 100)
    return v // 1000, (v % 1000) // 100


def score_loss(logit, scalars):
    th, hu = score_targets(scalars)
    return F.cross_entropy(logit[..., :N_TH].reshape(-1, N_TH).float(), th.reshape(-1)) + \
        F.cross_entropy(logit[..., N_TH:].reshape(-1, 10).float(), hu.reshape(-1))


def score_decode(logit):
    """[B, 4, N_TH+10] → 点数 /50000（与标量同尺度）[B, 4]。"""
    return (logit[..., :N_TH].argmax(-1) * 1000 + logit[..., N_TH:].argmax(-1) * 100).float() / 50000


def zero_unseen(planes, scalars):
    planes, scalars = planes.clone(), scalars.clone()
    planes[:, UNSEEN_PLANES] = 0
    scalars[:, UNSEEN_SCALARS] = 0
    return planes, scalars


# ---------------------------------------------------------------- 数据
class ImgState(Dataset):
    def __init__(self, split, proc, max_pixels, n=None, dense=False, dense_dir="dense2", center=False):
        self.split, self.proc, self.max_pixels = split, proc, max_pixels
        self.planes = np.load(f"{W}/mmj/{split}_planes.npy", mmap_mode="r")
        self.scalars = np.load(f"{W}/mmj/{split}_scalars.npy", mmap_mode="r")
        dd = f"{W}/vlm/{dense_dir}/{split}"
        self.kind = np.load(f"{dd}_kind.npy", mmap_mode="r") if dense else None
        self.region = np.load(f"{dd}_region.npy", mmap_mode="r") if dense else None
        self.center = np.load(f"{dd}_center.npy", mmap_mode="r") if center else None
        self.n = len(self.planes) if n is None else min(n, len(self.planes))
        self.text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}]}],
                                             tokenize=False, add_generation_prompt=False)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = Image.open(f"{W}/vlm/img/{self.split}/{i:06d}.png").convert("RGB")
        enc = self.proc(text=[self.text], images=[img], return_tensors="pt",
                        images_kwargs={"max_pixels": self.max_pixels})
        pl = np.array(self.planes[i])
        sc = np.array(self.scalars[i])
        mask = legal_mask(pl[None], sc[None])[0]
        kind = torch.from_numpy(np.array(self.kind[i])).long() if self.kind is not None else torch.zeros(0)
        region = torch.from_numpy(np.array(self.region[i])).long() if self.region is not None else torch.zeros(0)
        center = torch.from_numpy(np.array(self.center[i])).long() if self.center is not None else torch.zeros(0)
        return enc["pixel_values"], enc["image_grid_thw"], torch.from_numpy(pl), torch.from_numpy(sc), \
            torch.from_numpy(mask), kind, region, center, i


def collate(items):
    pix, thw, pl, sc, mask, kind, region, center, idx = zip(*items)
    return torch.cat(pix), torch.cat(thw), torch.stack(pl), torch.stack(sc), torch.stack(mask), \
        torch.stack(kind), torch.stack(region), torch.stack(center), list(idx)


# ---------------------------------------------------------------- 模型
class Head(nn.Module):
    def __init__(self, vis_dim, n_tok, d=512, layers=4, heads=8):
        super().__init__()
        self.mem = nn.Linear(vis_dim, d)
        self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
        self.q = nn.Parameter(torch.randn(1, N_TILE + 1, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.plane = nn.Linear(d, len(PLANES))
        self.scal = nn.Linear(d, len(SCALARS))

    def forward(self, vis):                                   # vis [B, n_tok, vis_dim]
        mem = self.mem(vis) + self.pos
        h = self.norm(self.dec(self.q.expand(len(vis), -1, -1), mem))
        return self.plane(h[:, :N_TILE]).transpose(1, 2), self.scal(h[:, N_TILE])   # [B,58,34], [B,71]


class PoolHead(nn.Module):
    """按位置汇聚：渲染画面的布局固定（手牌、四家牌河、副露、宝牌都在固定位置），
    每个"区域"学一张只看位置、不看内容的权重图，把 token 特征汇聚成 n_reg 个向量，再用 MLP 读出平面和标量。
    比查询 token 的交叉注意力（DETR 式，收敛很慢）好训得多。"""

    def __init__(self, vis_dim, n_tok, d=256, n_reg=48, hidden=2048):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.region = nn.Parameter(torch.randn(n_reg, n_tok) * 0.01)
        self.mlp = nn.Sequential(nn.Linear(n_reg * d, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, len(PLANES) * N_TILE + len(SCALARS)))

    def forward(self, vis):                                   # vis [B, n_tok, vis_dim]
        f = self.proj(vis)
        z = torch.einsum("rt,btd->brd", torch.softmax(self.region, -1), f)
        out = self.mlp(z.flatten(1))
        n = len(PLANES) * N_TILE
        return out[:, :n].reshape(-1, len(PLANES), N_TILE), out[:, n:]


class ConvHead(nn.Module):
    """图块特征按真实行列排回 48×64 的网格 → 卷积下采样到 24×32 → 展平 + MLP。
    Qwen 预处理器的图块顺序是 (行块, 列块, 块内行, 块内列)，每 2×2 个图块连续排列。
    布局固定，展平后的全连接层等于"每个位置有自己的读出权重"，梯度直接，收敛快。"""

    def __init__(self, vis_dim, n_tok, grid=(48, 64), c=128, c_out=48, hidden=1024, in_norm="bn", dense=False,
                 hist=False, center=False, score_cls=False, score_roi=False):
        super().__init__()
        self.gh, self.gw = grid
        assert self.gh * self.gw == n_tok, f"网格 {grid} 与 token 数 {n_tok} 不符"
        # 视觉塔最后一层有少数通道数值极大（整体 std≈51、均值≈8）。按 token 的 LayerNorm 会被这几个通道主导、
        # 把其余通道压得很小；按通道标准化（BatchNorm，统计量来自 batch 内所有图块）不受影响
        self.in_norm = in_norm
        self.norm = nn.BatchNorm1d(vis_dim) if in_norm == "bn" else nn.LayerNorm(vis_dim)
        self.inp = nn.Linear(vis_dim, c)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.GroupNorm(8, c),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GELU(), nn.GroupNorm(8, c),
            nn.Conv2d(c, c_out, 1))
        # 逐图块分类头（--dense）：接在第一层卷积之后（48×64 全分辨率），预测每格是哪张牌（37 类 + 背景）、属于哪个区域。
        # 有逐格标签时梯度直接告诉每个位置"这里是哪张牌"，比只看汇总后的平面强得多；后面的读出共用这层特征。
        # --center 时再加 38 个通道：这一格是不是某张牌的中心、是哪种牌。相邻的同种牌在 kind 图上连成一片，
        # 数不出张数；每张牌恰好一个中心格，按中心数就准了（assemble.py 用它组装牌局）
        self.dense = nn.Conv2d(c, N_KIND + N_REGION + (N_KIND if center else 0), 1) if dense else None
        # 直方图通路（--hist）：逐格的"区域 × 牌种"概率求和成 12×38 的直方图（例如手牌区里 5m 占了多少格），
        # 小 MLP 换算成平面，加到整图读出的平面上。手牌 / 牌河 / 副露 / 宝牌平面基本都是计数，这条路直接得多。
        # 最后一层初始化为 0：一开始不干扰整图读出
        self.hist = None
        if dense and hist:
            self.hist = nn.Sequential(nn.Linear(N_REGION * N_KIND, 512), nn.GELU(), nn.Linear(512, len(PLANES) * N_TILE))
            nn.init.zeros_(self.hist[-1].weight)
            nn.init.zeros_(self.hist[-1].bias)
        n_flat = c_out * (self.gh // 2) * (self.gw // 2)
        self.mlp = nn.Sequential(nn.Linear(n_flat, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, len(PLANES) * N_TILE + len(SCALARS)))
        # 点数分类头（--score_cls）：回归的点数平均差 400～480 点，误差 <50 点的只有约 11%，"几位、差几点"写不对。
        # 点数都是 100 的倍数、0～94000：四家各分"千位"（0～120）和"百位"（0～9）两个分类
        self.score = nn.Linear(hidden, 4 * (N_TH + 10)) if score_cls else None
        # 点数 ROI 头（--score_roi，替代上面的 score）：整图读出要先降到 24×32 再拉平，几像素高的数字基本丢了
        # （v6：四家全对 22.8%，21.6% 的读数错 1000 点以上）。这里直接取视觉塔在点数那几个图块上的特征
        # （归一化之后、降到 c 维之前），四家共用一个小 MLP，末尾拼座位 one-hot 区分版面
        self.roi = None
        if score_roi:
            self.roi = nn.Sequential(nn.Linear(vis_dim * ROI_H * ROI_W + 4, 512), nn.GELU(), nn.LayerNorm(512),
                                     nn.Linear(512, N_TH + 10))

    def forward(self, vis):
        b, n, d = vis.shape
        xn = self.norm(vis.reshape(b * n, d)).reshape(b, n, d) if self.in_norm == "bn" else self.norm(vis)
        x = self.inp(xn)
        x = self.grid(x)
        h1 = self.conv[:3](x)                                  # 第一层卷积（全分辨率）
        z = self.mlp[:-1](self.conv[3:](h1).flatten(1))
        out = self.mlp[-1](z)
        n = len(PLANES) * N_TILE
        planes = out[:, :n].reshape(-1, len(PLANES), N_TILE)
        if self.dense is None:
            return planes, out[:, n:]
        d = self.dense(h1)
        if self.hist is not None:
            pk = d[:, :N_KIND].float().softmax(1)
            pr = d[:, N_KIND:N_KIND + N_REGION].float().softmax(1)
            h = torch.einsum("brij,bkij->brk", pr, pk) / 8          # 手牌一张约 12 格，除 8 让数值在个位数
            planes = planes + self.hist(h.flatten(1)).reshape(-1, len(PLANES), N_TILE)
        res = (planes, out[:, n:], d)
        if self.roi is not None:
            g = self.grid(xn)
            win = torch.stack([g[:, :, r:r + ROI_H, c:c + ROI_W].flatten(1) for r, c in SCORE_ROI], 1)
            seat = torch.eye(4, device=win.device, dtype=win.dtype).expand(b, 4, 4)
            return res + (self.roi(torch.cat([win, seat], -1)),)              # [B, 4, N_TH + 10]
        return res + (self.score(z).reshape(-1, 4, N_TH + 10),) if self.score is not None else res

    def grid(self, t):
        """[B, 3072, C] 图块序列（(hb, wb, hi, wi) 顺序）→ [B, C, 48, 64] 按画面排列的网格。"""
        b = t.shape[0]
        return t.view(b, self.gh // 2, self.gw // 2, 2, 2, -1).permute(0, 5, 1, 3, 2, 4).reshape(b, -1, self.gh, self.gw)


class Perception(nn.Module):
    def __init__(self, llm, n_tok=768, lora_r=16, d=512, layers=4, head="conv", feat="patch", in_norm="bn",
                 dense=False, hist=False, center=False, score_cls=False, score_roi=False):
        super().__init__()
        vlm = AutoModelForImageTextToText.from_pretrained(llm, dtype=torch.bfloat16)
        visual = vlm.model.visual
        visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            visual = get_peft_model(visual, LoraConfig(
                r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                target_modules=r".*blocks.*\.(qkv|proj|linear_fc1|linear_fc2)"))
        else:
            visual.requires_grad_(False)
        base = visual.get_base_model() if hasattr(visual, "get_base_model") else visual
        self.feat = feat
        if feat == "merged":      # merger 之后的 768 个 token（LLM 空间）；merger 一起训
            base.merger.float().requires_grad_(True)
            vis_dim = vlm.config.text_config.hidden_size
        else:                     # merger 之前的图块特征（1024×768 → 3072 个 × 768 维），细节更多；merger 不参与
            n_tok *= base.spatial_merge_size ** 2
            base.merger.requires_grad_(False)
            vis_dim = vlm.config.vision_config.hidden_size
        self.visual = visual
        self.n_tok = n_tok
        self.head = {"dec": lambda: Head(vis_dim, n_tok, d, layers), "pool": lambda: PoolHead(vis_dim, n_tok),
                     "conv": lambda: ConvHead(vis_dim, n_tok, in_norm=in_norm, dense=dense, hist=hist,
                                              center=center, score_cls=score_cls and not score_roi,
                                              score_roi=score_roi)}[head]()
        del vlm

    def forward(self, pix, thw):
        o = self.visual(pix, grid_thw=thw, return_dict=True)
        out = o.pooler_output if self.feat == "merged" else o.last_hidden_state     # [B*n_tok, dim]
        assert out.shape[0] == len(thw) * self.n_tok, f"视觉 token 数 {out.shape[0]} ≠ {len(thw)}×{self.n_tok}"
        return self.head(out.reshape(len(thw), self.n_tok, -1).float())

    def trainable_state(self):
        # 可训练参数 + 读出头的缓冲区（BatchNorm 的 running_mean/var 不是参数，不存的话载入后 eval 模式全错）
        sd = {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}
        sd.update({n: b.detach().cpu() for n, b in self.named_buffers() if n.startswith("head.")})
        return sd


def load_encoder(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    enc = EncoderPolicy(**ck["enc_cfg"], value=any(k.startswith("vhead.") for k in sd))
    enc.load_state_dict(sd)
    return enc.to(device).eval().requires_grad_(False)


# ---------------------------------------------------------------- 损失 / 评测
def losses(pl_logit, sc_pred, planes, scalars, mask, enc, kl_w):
    tgt = planes.float().clamp(max=1)
    bce = F.binary_cross_entropy_with_logits(pl_logit[:, VISIBLE], tgt[:, VISIBLE])
    keep = [i for i in range(len(SCALARS)) if i not in UNSEEN_SCALARS]
    sl1 = F.smooth_l1_loss(sc_pred[:, keep], scalars[:, keep], beta=0.1)
    kl = torch.zeros((), device=planes.device)
    if kl_w > 0:
        tp, ts = zero_unseen(planes.float(), scalars)
        with torch.no_grad():
            t_logit = enc(tp, ts, mask).float()
        sp, ss = zero_unseen(torch.sigmoid(pl_logit), sc_pred)
        s_logit = enc(sp, ss, mask).float()
        kl = F.kl_div(F.log_softmax(s_logit, -1), F.softmax(t_logit, -1), reduction="batchmean")
    return bce, sl1, kl


_W = {}


def dense_loss(dense_logit, kind, region, center=None):
    """逐图块交叉熵：牌种类（37 + 背景）+ 区域（11 + 背景），背景格占约 9 成、权重 0.1；
    牌中心（37 + 背景）：每张牌只有一个中心格（约 75 / 3072），背景权重 0.05。"""
    dev = dense_logit.device
    if dev not in _W:
        one = lambda n, i, w: torch.ones(n, device=dev).index_fill_(0, torch.tensor([i], device=dev), w)  # noqa: E731
        _W[dev] = (one(N_KIND, BG_KIND, 0.1), one(N_REGION, 0, 0.1), one(N_KIND, BG_KIND, 0.05))
    wk, wr, wc = _W[dev]
    loss = F.cross_entropy(dense_logit[:, :N_KIND].float(), kind, weight=wk) + \
        F.cross_entropy(dense_logit[:, N_KIND:N_KIND + N_REGION].float(), region, weight=wr)
    if center is not None and center.numel():
        loss = loss + F.cross_entropy(dense_logit[:, N_KIND + N_REGION:].float(), center, weight=wc)
    return loss


@torch.no_grad()
def evaluate(model, enc, loader, device, save=None):
    model.eval()
    cells = cells_ok = n = agree_true_mask = agree_pred_mask = 0
    grp_ok = {g: 0 for g in GROUPS}
    all_ok = 0
    sc_err = 0.0
    preds_pl, preds_sc, rows = [], [], []
    d_tile = d_kind_ok = d_region_ok = 0
    tile_ok = tile_n = score_ok = score_n = 0
    keep = [i for i in range(len(SCALARS)) if i not in UNSEEN_SCALARS]
    for pix, thw, planes, scalars, mask, kind, region, center, idx in loader:
        pix, thw = pix.to(device, torch.bfloat16), thw.to(device)
        planes, scalars, mask = planes.to(device), scalars.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pix, thw)
        pl_logit, sc_pred = out[0], out[1]
        if len(out) > 2 and kind.numel():            # 逐图块：只在"是牌"的格子上统计
            kind, region = kind.to(device), region.to(device)
            is_tile = region > 0
            d_tile += int(is_tile.sum())
            d_kind_ok += int(((out[2][:, :N_KIND].argmax(1) == kind) & is_tile).sum())
            d_region_ok += int(((out[2][:, N_KIND:N_KIND + N_REGION].argmax(1) == region) & is_tile).sum())
        pred = (pl_logit.float() > 0).to(torch.uint8)
        sc = sc_pred.float()
        if len(out) > 3:                              # 点数分类头：替换回归出的点数
            sc[:, SCORE_IDX] = score_decode(out[3].float())
            score_ok += int((sc[:, SCORE_IDX] * 50000).round().eq((scalars[:, SCORE_IDX] * 50000).round()).all(1).sum())
            score_n += len(sc)
        assembled = len(out) > 2 and out[2].shape[1] == 2 * N_KIND + N_REGION
        if assembled:
            # 有牌中心头：与牌有关的平面改由组装规则从"中心格 + 区域"算出（assemble.py），其余平面仍用读出头；
            # 画面上单独摆出的摸牌直接给 last_tsumo
            d = out[2].float().cpu().numpy()
            for b in range(len(d)):
                tiles = assemble.tiles_from_dense(d[b, N_KIND + N_REGION:], d[b, N_KIND:N_KIND + N_REGION])
                pred[b, assemble.TILE_IDX] = torch.from_numpy(assemble.tile_planes(tiles)).to(device)
                ts_k = [k for _, _, k, r in tiles if r == "tsumo"]
                if ts_k:
                    pred[b, PLANES.index("last_tsumo")] = 0
                    pred[b, PLANES.index("last_tsumo"), assemble.KINDS.index(assemble.name(ts_k[0])[:2])] = 1
            ok_t = pred[:, assemble.TILE_IDX].clamp(max=1) == planes[:, assemble.TILE_IDX].clamp(max=1)
            tile_ok += int(ok_t.flatten(1).all(1).sum())
            tile_n += len(d)
        pred[:, UNSEEN_PLANES] = 0
        if assembled:                                 # 规则后处理（postprocess.py）：last_kawa、吃的种类、向听数、位次与点差
            rp, rs = postprocess.refine_all(pred.cpu().numpy(), sc.cpu().numpy())
            pred, sc = torch.from_numpy(rp).to(device), torch.from_numpy(rs).to(device)
        tgt = planes.clamp(max=1)
        ok = pred[:, VISIBLE].clamp(max=1) == tgt[:, VISIBLE]
        cells += ok.numel()
        cells_ok += int(ok.sum())
        for g, ids in GROUPS.items():
            grp_ok[g] += int((pred[:, ids].clamp(max=1) == tgt[:, ids]).flatten(1).all(1).sum())
        all_ok += int(ok.flatten(1).all(1).sum())
        sc_err += float((sc[:, keep] - scalars[:, keep]).abs().mean(1).sum())
        # 决策：编码器吃真实局面 vs 吃预测局面（两边都清掉看不见的特征）
        tp, ts = zero_unseen(planes.float(), scalars)
        sp, ss = zero_unseen(pred.float(), sc)
        t_act = enc(tp, ts, mask).argmax(-1)
        agree_true_mask += int((enc(sp, ss, mask).argmax(-1) == t_act).sum())
        pmask = torch.from_numpy(legal_mask(pred.cpu().numpy(), sc.cpu().numpy().round())).to(device)   # 合法动作标志按 0/1 取整
        agree_pred_mask += int((enc(sp, ss, pmask).argmax(-1) == t_act).sum())
        n += len(idx)
        if save:
            preds_pl.append(pred.cpu().numpy())
            preds_sc.append(sc.cpu().numpy())
            rows += idx
    model.train()
    res = {"n": n, "逐格准确率": cells_ok / max(1, cells), "整张全对": all_ok / max(1, n),
           **{f"{g}全对": v / max(1, n) for g, v in grp_ok.items()},
           "标量MAE": sc_err / max(1, n),
           "决策一致(真合法手)": agree_true_mask / max(1, n),
           "决策一致(预测合法手)": agree_pred_mask / max(1, n)}
    if d_tile:
        res["逐格认牌"] = d_kind_ok / d_tile
        res["逐格区域"] = d_region_ok / d_tile
    if tile_n:
        res[f"牌平面全对({len(assemble.TILE_IDX)}个)"] = tile_ok / tile_n
    if score_n:
        res["四家点数全对"] = score_ok / score_n
    if save:
        np.savez_compressed(save, planes=np.concatenate(preds_pl), scalars=np.concatenate(preds_sc),
                            rows=np.array(rows))
    return res


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--enc", default=f"{W}/mmj/encoder_tf256b6_v_x.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ga", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--head", choices=["conv", "pool", "dec"], default="conv",
                    help="conv = 排回网格 + 卷积 + MLP；pool = 按位置汇聚（区域图学不动）；dec = 查询 token 解码器（收敛慢）")
    ap.add_argument("--feat", choices=["patch", "merged"], default="patch", help="视觉塔 merger 之前 / 之后的特征")
    ap.add_argument("--in_norm", choices=["bn", "ln"], default="bn", help="conv 读出头输入的归一化：按通道（bn）/ 按 token（ln）")
    ap.add_argument("--kl_w", type=float, default=0.0,
                    help="经编码器的策略 KL 权重。默认关：实测 64 张图过拟合时，开 KL（权重 1）平面 BCE 停在 0.22，"
                         "比按频率猜的 0.171 还差；关掉后 0.135。编码器只见过 0/1 输入，概率输入时 KL 把平面拉离目标")
    ap.add_argument("--max_pixels", type=int, default=1024 * 768)
    ap.add_argument("--eval_n", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--train_n", type=int, default=None, help="只用训练集前 n 张（过拟合检查用）")
    ap.add_argument("--dense", action="store_true", help="加逐图块监督（需先跑 vlm/render_labels.py 生成标签）")
    ap.add_argument("--dense_w", type=float, default=1.0)
    ap.add_argument("--hist", action="store_true", help="逐格概率求直方图再读出平面（需 --dense）")
    ap.add_argument("--center", action="store_true", help="加牌中心头；评测时与牌有关的平面由 assemble.py 组装（需 --dense）")
    ap.add_argument("--dense_dir", default="dense2", help="逐格标签目录 work/vlm/<此目录>（center 只在 dense2 里有）")
    ap.add_argument("--score_cls", action="store_true", help="点数改成分类（千位 + 百位）")
    ap.add_argument("--score_roi", action="store_true", help="点数分类改从点数所在图块的局部特征读（替代 --score_cls 的整图读出）")
    ap.add_argument("--score_w", type=float, default=0.1, help="点数分类损失的权重")
    ap.add_argument("--init", default=None, help="从之前的 final.pt 载入权重（新加的头随机初始化）后接着训")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    proc = AutoProcessor.from_pretrained(args.llm)
    model = Perception(args.llm, lora_r=args.lora_r, head=args.head, feat=args.feat, in_norm=args.in_norm,
                       dense=args.dense, hist=args.hist, center=args.center, score_cls=args.score_cls,
                       score_roi=args.score_roi).to(device)
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=False)["model"]
        r = model.load_state_dict(sd, strict=False)
        print(f"从 {args.init} 载入 {len(sd)} 个张量；新加的：{[k for k in r.missing_keys if k.startswith('head.')]}",
              flush=True)
    enc = load_encoder(args.enc, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数 {n_train / 1e6:.2f}M；编码器 {args.enc}", flush=True)

    dk = dict(dense=args.dense, dense_dir=args.dense_dir, center=args.center)
    train = ImgState("train", proc, args.max_pixels, n=args.train_n, **dk)
    big = np.asarray(train.planes[:20000]).max(axis=(0, 2))
    print("取值超过 1 的平面（BCE 目标按 1 算）：", [PLANES[i] for i in np.where(big > 1)[0]], flush=True)
    test = ImgState("test", proc, args.max_pixels, n=args.eval_n, **dk)
    tl = DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    el = DataLoader(test, batch_size=16, shuffle=False, num_workers=args.workers, collate_fn=collate)
    quick = DataLoader(ImgState("test", proc, args.max_pixels, n=min(300, args.eval_n), **dk),
                       batch_size=16, num_workers=2, collate_fn=collate)

    steps = args.max_steps or math.ceil(args.epochs * len(train) / (args.bs * args.ga))
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    vis_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": vis_p, "lr": args.lr}, {"params": head_p, "lr": args.lr * 2}],
                            weight_decay=0.01)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    step = 0
    ck_path = f"{args.out}/ckpt.pt"
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"从第 {step} 步接着训", flush=True)

    log = open(f"{args.out}/train_log.jsonl", "a")
    print(f"训练 {len(train)} 张，共 {steps} 步（每步 {args.bs}×{args.ga} 张），KL 权重 {args.kl_w}", flush=True)
    model.train()
    t0, acc = time.time(), {"bce": 0.0, "sl1": 0.0, "kl": 0.0, "dense": 0.0, "score": 0.0}
    start_step, micro = step, 0
    it = iter(tl)
    while step < steps:
        try:
            pix, thw, planes, scalars, mask, kind, region, center, _ = next(it)
        except StopIteration:
            it = iter(tl)
            continue
        pix, thw = pix.to(device, torch.bfloat16, non_blocking=True), thw.to(device)
        planes, scalars, mask = planes.to(device), scalars.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pix, thw)
            bce, sl1, kl = losses(out[0].float(), out[1].float(), planes, scalars, mask, enc, args.kl_w)
            dl = dense_loss(out[2], kind.to(device), region.to(device), center.to(device)) if args.dense \
                else torch.zeros((), device=device)
            sl = score_loss(out[3], scalars) if len(out) > 3 else torch.zeros((), device=device)
            loss = bce + sl1 + args.kl_w * kl + args.dense_w * dl + args.score_w * sl
        (loss / args.ga).backward()
        acc["bce"] += float(bce) / args.ga
        acc["sl1"] += float(sl1) / args.ga
        acc["kl"] += float(kl) / args.ga
        acc["dense"] += float(dl) / args.ga
        acc["score"] += float(sl) / args.ga
        micro += 1
        if micro % args.ga:
            continue
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0 or step == steps:
            el_t = time.time() - t0
            sps = el_t / (step - start_step)
            rec = {"step": step, **{k: round(v / 50 if step % 50 == 0 else v, 5) for k, v in acc.items()},
                   "lr": sched.get_last_lr()[0], "s_per_step": round(sps, 3),
                   "eta_min": round(sps * (steps - step) / 60, 1)}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            acc = {k: 0.0 for k in acc}
        if step % args.eval_every == 0 and step < steps:
            print("快速评测", json.dumps(evaluate(model, enc, quick, device), ensure_ascii=False), flush=True)
        if step % args.save_every == 0 or step == steps:
            torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "args": vars(args)}, ck_path + ".tmp")
            os.replace(ck_path + ".tmp", ck_path)

    torch.save({"model": model.trainable_state(), "args": vars(args), "step": step}, f"{args.out}/final.pt")
    print(f"训练完成 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    res = evaluate(model, enc, el, device, save=f"{args.out}/test_pred.npz")
    print("最终评测", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{args.out}/eval.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
