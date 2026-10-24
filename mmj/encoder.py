"""局面编码器（多模态麻将模型的"视觉塔"）+ 第 0 步用的策略头。

StateEncoder：沿 34 种牌做 1D 卷积的 ResNet（和 Mortal 同一思路：牌种是"空间"维度，相邻牌能连成顺子、筋）。
  输入 planes [B, N_PLANES, 34] + scalars [B, N_SCALARS]
  输出 tiles [B, 34, C]（每种牌一个向量）+ glob [B, C]（全局向量）
  → 第 2 步接 LLM 时，这 35 个向量经投影层变成 35 个"局面 token"。
PolicyHead：只在第 0 步用，直接从编码器输出预测 46 个动作（纯 CNN 基线）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(groups, ch)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(groups, ch)
        self.se = nn.Sequential(nn.Linear(ch, ch // 8), nn.ReLU(), nn.Linear(ch // 8, ch), nn.Sigmoid())

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        h = h * self.se(h.mean(-1))[..., None]          # 通道注意力（SE）
        return F.relu(x + h)


class StateEncoder(nn.Module):
    def __init__(self, n_planes, n_scalars, ch=192, blocks=10):
        super().__init__()
        self.cfg = dict(n_planes=n_planes, n_scalars=n_scalars, ch=ch, blocks=blocks)
        self.inp = nn.Conv1d(n_planes, ch, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, ch, 34))     # 区分花色边界（9m 和 1p 在序列里相邻但无关）
        self.film = nn.Sequential(nn.Linear(n_scalars, ch), nn.ReLU(), nn.Linear(ch, 2 * ch))   # 场况调制每个通道
        self.blocks = nn.ModuleList(ResBlock(ch) for _ in range(blocks))
        self.glob = nn.Sequential(nn.Linear(2 * ch + n_scalars, ch), nn.ReLU(), nn.Linear(ch, ch))

    def forward(self, planes, scalars):
        x = self.inp(planes.float()) + self.pos
        g, b = self.film(scalars).chunk(2, -1)
        x = x * (1 + g[..., None]) + b[..., None]
        for blk in self.blocks:
            x = blk(x)
        glob = self.glob(torch.cat([x.mean(-1), x.amax(-1), scalars], -1))
        return x.transpose(1, 2), glob                       # [B, 34, C], [B, C]


class TileTransformer(nn.Module):
    """另一种编码器：34 种牌各一个 token + 1 个场况 token，过 Transformer（牌与牌的关系靠注意力学，不限于相邻）。
    输入输出与 StateEncoder 相同（tiles [B, 34, C]、glob [B, C]），后面的投影层 / LLM 不用改。"""

    def __init__(self, n_planes, n_scalars, ch=256, blocks=6, heads=8):
        super().__init__()
        self.cfg = dict(arch="tf", n_planes=n_planes, n_scalars=n_scalars, ch=ch, blocks=blocks, heads=heads)
        self.inp = nn.Linear(n_planes, ch)
        self.tile = nn.Parameter(torch.zeros(1, 34, ch))     # 牌种身份（哪张牌、哪个花色）
        nn.init.normal_(self.tile, std=0.02)
        self.sc = nn.Linear(n_scalars, ch)
        layer = nn.TransformerEncoderLayer(ch, heads, 4 * ch, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(ch)

    def forward(self, planes, scalars):
        x = self.inp(planes.float().transpose(1, 2)) + self.tile          # [B, 34, C]
        h = self.norm(self.tf(torch.cat([self.sc(scalars)[:, None], x], 1)))
        return h[:, 1:], h[:, 0]


def make_encoder(cfg):
    """按 checkpoint 里的 enc_cfg 建编码器（旧的 cfg 没有 arch 字段 = 卷积版）。"""
    cfg = dict(cfg)
    return TileTransformer(**cfg) if cfg.pop("arch", "cnn") == "tf" else StateEncoder(**cfg)


class PolicyHead(nn.Module):
    """46 个动作：0-33 打牌、34-36 打赤五、37-45 其它（reach … pass）。"""

    def __init__(self, ch, aka_pos=(4, 13, 22)):
        super().__init__()
        self.aka_pos = list(aka_pos)
        self.dahai = nn.Linear(ch, 1)
        self.aka = nn.Linear(ch, 1)
        self.calls = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, 9))

    def forward(self, tiles, glob):
        d = self.dahai(tiles).squeeze(-1)                    # [B, 34]
        a = self.aka(tiles[:, self.aka_pos]).squeeze(-1)     # [B, 3]
        return torch.cat([d, a, self.calls(glob)], -1)       # [B, 46]


class EncoderPolicy(nn.Module):
    """编码器 + 策略头；value=True 时再加一个同结构的价值头，回归各动作相对最优动作的 Q 值（优势）。"""

    def __init__(self, n_planes, n_scalars, ch=192, blocks=10, dropout=0.0, arch="cnn", heads=8, value=False):
        super().__init__()
        self.enc = TileTransformer(n_planes, n_scalars, ch, blocks, heads) if arch == "tf" \
            else StateEncoder(n_planes, n_scalars, ch, blocks)
        self.head = PolicyHead(ch)
        self.vhead = PolicyHead(ch) if value else None
        self.drop = nn.Dropout(dropout)

    def forward(self, planes, scalars, mask=None, with_value=False):
        tiles, glob = self.enc(planes, scalars)
        tiles, glob = self.drop(tiles), self.drop(glob)
        logits = self.head(tiles, glob)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e4)
        return (logits, self.vhead(tiles, glob)) if with_value else logits
