"""把一个决策点的局面渲染成"人看到的牌桌画面"（VLM 路线的输入）。

只画画面上本来就有的东西：自己的手牌（摸到的牌隔开放在右端，赤五画红）、四家牌河（6 张一行，对手的河按座位旋转，
立直宣言牌横放）、副露、宝牌指示牌、局况（场风・局・本场・供托・剩余张数）、四家风位 / 点数 / 立直标记、可执行动作按钮。
不画向听数、见过的牌数这类辅助信息——人得自己看、自己数（与"像人"的目标一致）。
牌面用 Windows 自带的 Segoe UI Symbol（Unicode 麻将牌 U+1F000–1F02B），文字用游ゴシック；只在本机渲染训练数据，不分发图片。
"""
import json
import re

from PIL import Image, ImageDraw, ImageFont

FONT_TILE = "/mnt/c/Windows/Fonts/seguisym.ttf"
FONT_TEXT = "/mnt/c/Windows/Fonts/YuGothM.ttc"
GLYPH = {**{"E": 0x1F000, "S": 0x1F001, "W": 0x1F002, "N": 0x1F003, "C": 0x1F004, "F": 0x1F005, "P": 0x1F006},
         **{f"{n}m": 0x1F007 + n - 1 for n in range(1, 10)},
         **{f"{n}s": 0x1F010 + n - 1 for n in range(1, 10)},
         **{f"{n}p": 0x1F019 + n - 1 for n in range(1, 10)}}
WIND = {"E": "東", "S": "南", "W": "西", "N": "北"}
SEAT_NAMES = ["自家", "下家", "対面", "上家"]
ACT_LABEL = {"reach": "リーチ", "chi_low": "チー", "chi_mid": "チー", "chi_high": "チー", "pon": "ポン", "kan": "カン",
             "hora": "和了", "ryukyoku": "流局", "pass": "スルー"}
BG, TILE_FILL, INK, RED = (34, 92, 66), (250, 248, 240), (20, 20, 20), (200, 30, 30)
FIELD = re.compile(r"\n\n\d+\. ([^\n]+):\n(.*?)(?=\n\n\d+\. |\n\nPlease tell me)", re.S)

_fonts = {}


def font(path, size):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def fields(prompt):
    return {k: json.loads(v) for k, v in FIELD.findall(prompt)}


def base(t):
    t = t.rstrip("|^!").rstrip("|^!")
    return t[:2] if t[:1].isdigit() else t[:1]


def tile_img(t, h, sideways=False):
    """一张牌：白底 + Unicode 牌面。h = 牌高（像素）。"""
    w = int(h * 0.74)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=max(2, h // 10), fill=TILE_FILL, outline=(90, 90, 90))
    g = chr(GLYPH[base(t)])
    f = font(FONT_TILE, int(h * 1.02))
    bb = d.textbbox((0, 0), g, font=f)
    color = RED if t.rstrip("|^!").endswith("r") else INK
    d.text(((w - (bb[2] - bb[0])) / 2 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), g, font=f, fill=color)
    return im.rotate(90, expand=True) if sideways else im


def row_img(tiles, h, per_row=None, gap=2, riichi_idx=None, hl_idx=None):
    """一行（或按 per_row 折行）牌。riichi_idx：立直宣言牌下标（横放）；hl_idx：高亮（刚打出、可以鸣的那张）。"""
    if not tiles:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    per_row = per_row or len(tiles)
    w1, wide = int(h * 0.74), h
    rows = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    W = per_row * (w1 + gap) + (wide - w1) + 4
    H = len(rows) * (h + gap)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    k = 0
    for r, row in enumerate(rows):
        x = 0
        for t in row:
            side = riichi_idx is not None and k == riichi_idx
            ti = tile_img(t, h, sideways=side)
            y = r * (h + gap) + (h - ti.height)
            im.alpha_composite(ti, (x, y))
            if hl_idx is not None and k == hl_idx:
                ImageDraw.Draw(im).rounded_rectangle([x, y, x + ti.width - 1, y + ti.height - 1], radius=3,
                                                     outline=(255, 200, 0), width=3)
            x += ti.width + gap
            k += 1
    return im


def render(f, legal=None, size=(1024, 768)):
    W, H = size
    img = Image.new("RGBA", size, BG + (255,))
    d = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2 - 40

    # ---- 中央面板：局况 + 宝牌指示牌 + 四家风位 / 点数
    pw, ph = 230, 230
    d.rounded_rectangle([cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2], radius=12, fill=(18, 50, 36))
    ft = font(FONT_TEXT, 22)
    title = f"{WIND[f['bakaze']]}{f['kyoku'] + 1}局  {f['honba']}本場"
    d.text((cx, cy - 60), title, font=ft, fill=(240, 240, 240), anchor="mm")
    d.text((cx, cy - 34), f"供託 {f['kyotaku']}  残り {f['title left']}", font=font(FONT_TEXT, 18), fill=(220, 220, 200), anchor="mm")
    dora = row_img(f["dora indicators"], 34)
    img.alpha_composite(dora, (cx - dora.width // 2, cy - 18))
    winds = "ESWN"
    me = winds.index(f["jikaze"])
    # 四家风位 / 点数：自家在下、下家在右、対面在上、上家在左
    pos = [(cx, cy + ph // 2 - 20), (cx + pw // 2 - 38, cy + 34), (cx, cy - ph // 2 + 18), (cx - pw // 2 + 38, cy + 34)]
    for p in range(4):
        wind = WIND[winds[(me + p) % 4]]
        s = f"{wind} {f['Scores'][p]}"
        color = (255, 210, 80) if f["riichi declared"][p] else (240, 240, 240)
        x, y = pos[p]
        if p in (0, 2):
            d.text((x, y), s, font=font(FONT_TEXT, 17), fill=color, anchor="mm")
        else:
            d.text((x, y - 6), wind, font=font(FONT_TEXT, 17), fill=color, anchor="mm")
            d.text((x, y + 14), str(f["Scores"][p]), font=font(FONT_TEXT, 14), fill=color, anchor="mm")
        if f["riichi declared"][p]:                  # 立直棒
            bx, by = {0: (cx, cy + ph // 2 + 10), 1: (cx + pw // 2 + 10, cy), 2: (cx, cy - ph // 2 - 10),
                      3: (cx - pw // 2 - 10, cy)}[p]
            if p in (0, 2):
                d.rounded_rectangle([bx - 40, by - 4, bx + 40, by + 4], radius=4, fill=(245, 245, 245))
                d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=RED)
            else:
                d.rounded_rectangle([bx - 4, by - 40, bx + 4, by + 40], radius=4, fill=(245, 245, 245))
                d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=RED)

    # ---- 四家牌河（6 张一行），立直宣言牌横放；对手的河按座位旋转
    rot = {0: 0, 1: 90, 2: 180, 3: 270}
    # 别人刚打出、自己可以鸣 / 荣和的那张牌：高亮（轮到自己打牌时不高亮）
    legal_all = legal or f["Legal actions"]
    hl_p = None
    lk = f.get("last kawa tile")
    if lk and not any(a.startswith("dahai") for a in legal_all):
        cands = [3] if any(a.startswith("chi") for a in legal_all) else [3, 2, 1]
        hl_p = next((p for p in cands if [t for t in f["kawa overview"][p] if t][-1:]
                     and base([t for t in f["kawa overview"][p] if t][-1]) == base(lk)), None)
    for p in range(4):
        kawa = [t for t in f["kawa overview"][p] if t]
        rs = f["riichi sutehais"][p]
        ridx = None
        if rs:   # 宣言牌：在河里找最后一张同种牌（简化）
            b = base(rs)
            ridx = max((i for i, t in enumerate(kawa) if base(t) == b), default=None)
        river = row_img(kawa, 34, per_row=6, riichi_idx=ridx, hl_idx=len(kawa) - 1 if p == hl_p else None)
        river = river.rotate(rot[p], expand=True)
        if p == 0:
            xy = (cx - pw // 2 + 10, cy + ph // 2 + 18)
        elif p == 1:
            xy = (cx + pw // 2 + 18, cy + ph // 2 - river.height + 10)
        elif p == 2:
            xy = (cx + pw // 2 - river.width - 10, cy - ph // 2 - 18 - river.height)
        else:
            xy = (cx - pw // 2 - 18 - river.width, cy - ph // 2 - 10)
        img.alpha_composite(river, (int(xy[0]), int(xy[1])))

    # ---- 副露（对手的放在各自一侧外沿，自己的放在手牌右侧上方）
    for p in range(4):
        melds = [m for m in f["fuuro overview"][p]] + [[t] * 4 for t in f["ankan overview"][p]]
        if not melds:
            continue
        flat = [t for m in melds for t in m + [None]][:-1]
        mi = row_img([t for t in flat if t], 28)
        mi = mi.rotate(rot[p], expand=True)
        xy = {0: (W - mi.width - 20, H - 150), 1: (W - mi.width - 12, 20), 2: (20, 12),
              3: (12, H - 230 - mi.height)}[p]
        img.alpha_composite(mi, xy)

    # ---- 自己的手牌（摸到的牌隔开放右端）
    hand = list(f["my hands"])
    tsumo = f.get("last self tsumo")
    if len(hand) % 3 == 2 and tsumo and tsumo in hand:
        hand.remove(tsumo)
        drawn = tsumo
    else:
        drawn = None
    hi = row_img(hand, 64)
    x0 = (W - hi.width - (60 if drawn else 0)) // 2
    img.alpha_composite(hi, (x0, H - 88))
    if drawn:
        img.alpha_composite(tile_img(drawn, 64), (x0 + hi.width + 14, H - 88))

    # ---- 可执行动作按钮（打牌以外的）
    acts = [a for a in (legal or f["Legal actions"]) if not a.startswith("dahai")]
    labels = []
    for a in acts:
        lab = ACT_LABEL.get(a, a)
        if lab not in labels:
            labels.append(lab)
    bx = W - 20
    for lab in reversed(labels):
        tw = d.textlength(lab, font=font(FONT_TEXT, 20)) + 28
        d.rounded_rectangle([bx - tw, H - 140, bx, H - 104], radius=8, fill=(250, 200, 60), outline=(120, 80, 0))
        d.text((bx - tw / 2, H - 122), lab, font=font(FONT_TEXT, 20), fill=(60, 30, 0), anchor="mm")
        bx -= tw + 10
    return img.convert("RGB")
