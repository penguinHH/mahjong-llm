"""最小 obs 构造演示：mjai 事件 -> PlayerState -> (obs, mask)"""
import sys, json
sys.path.insert(0, "/home/user/mahjong-llm/Mortal/target/release")
import numpy as np
from libriichi.state import PlayerState

# 一局最小的东1局0本场开局（mjai 标准格式）
events = [
    {"type": "start_game", "names": ["A", "B", "C", "D"]},
    {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
     "oya": 0, "scores": [25000,25000,25000,25000], "dora_marker": "3m",
     "tehais": [
        ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
        ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
        ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
        ["?","?","?","?","?","?","?","?","?","?","?","?","?"]]},
]

ps = PlayerState(0)
for e in events:
    ps.update(json.dumps(e))

obs, mask = ps.encode_obs(4, False)
print("obs shape:", obs.shape, "dtype:", obs.dtype, "range:", obs.min(), obs.max())
print("mask shape:", mask.shape, "legal actions:", int(mask.sum()))
print("mask true idx:", [i for i,v in enumerate(mask) if v])

# 按段打印非零通道
names = {
 0:"self tehai x4",4:"aka x3",7:"scores x8",15:"rank x4",19:"kyoku x4",
 23:"honba",24:"kyotaku",25:"bakaze",26:"jikaze",27:"kyoku combo",
 28:"dora indicators x7",35:"self kawa 6 x4",59:"self kawa 18 x4",
 131:"self kawa decay",132:"others kawa x3",717:"tiles_left",
 718:"doras owned x4",722:"doras unseen",723:"kawa overview x28",
 751:"fuuro overview x80",831:"ankan x4",835:"tiles seen",
 836:"last tedashi x3",845:"riichi sutehai x3",854:"riichi declared x3",
 857:"riichi accepted x3",860:"waits",861:"furiten",862:"shanten x7",
 869:"self riichi",870:"at kan select",871:"can pass tile x3",
 874:"discard cand x5",879:"can riichi",880:"can chi x3",883:"can pon",
 884:"can daiminkan",885:"can ankan",886:"can kakan",887:"can agari",
 888:"can ryukyoku",889:"v4 ev x2",891:"v4 sp tables",
}
nz = [(i, int((obs[i] > 0).sum())) for i in range(obs.shape[0]) if (obs[i] > 0).any()]
print("\nnon-zero channels (idx, count):")
print(nz)
print("\nseg boundaries check: total channels =", obs.shape[0])
