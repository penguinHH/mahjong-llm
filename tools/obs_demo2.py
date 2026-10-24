"""obs 构造 + 段解码 + 模型推理 演示"""
import sys, json
sys.path.insert(0, "/home/user/mahjong-llm/Mortal/target/release")
sys.path.insert(0, "/home/user/mahjong-llm/Mortal/mortal")
import numpy as np, torch
from libriichi.state import PlayerState
from model import Brain, DQN

TILES = ["1m","2m","3m","4m","5m","6m","7m","8m","9m",
         "1p","2p","3p","4p","5p","6p","7p","8p","9p",
         "1s","2s","3s","4s","5s","6s","7s","8s","9s",
         "E","S","W","N","P","F","C"]

ev = lambda **k: json.dumps(k)
events = [
 ev(type="start_game", names=["A","B","C","D"]),
 ev(type="start_kyoku", bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0,
    scores=[25000,25000,25000,25000], dora_marker="3m",
    tehais=[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
            ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
            ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
            ["?","?","?","?","?","?","?","?","?","?","?","?","?"]]),
 ev(type="tsumo", actor=0, pai="5pr"),
]
ps = PlayerState(0)
for e in events:
    ps.update(e)

obs, mask = ps.encode_obs(4, False)
print("obs:", obs.shape, obs.dtype, "min/max", round(float(obs.min()),4), round(float(obs.max()),4))
print("mask:", mask.shape, "legal:", int(mask.sum()))
print("legal idx:", [i for i,v in enumerate(mask) if v])

def decode(obs):
    def rows(a,b):
        out=[]
        for i in range(a,b):
            c=np.flatnonzero(obs[i])
            if len(c): out.append((i,[TILES[j] if j<34 else j for j in c], np.round(obs[i][c],3).tolist()))
        return out
    print("\n[0:4] 自家手牌 (每张牌占一行计数)")
    for i in range(4):
        c=np.flatnonzero(obs[i]); print("  ch",i, [(TILES[j],round(float(obs[i][j]),2)) for j in c])
    print("[4:7] 手中赤五:", np.flatnonzero(obs[4:7].any(1)).tolist())
    print("[7:15] 四家分数:", [round(float(obs[7+2*k][0]),4) for k in range(4)])
    print("[15:19] 名次 one-hot:", np.flatnonzero(obs[15:19].any(1)).tolist())
    print("[19:23] 局数 one-hot:", np.flatnonzero(obs[19:23].any(1)).tolist())
    print("[23] 本场:", float(obs[23][0]), "[24] 立直棒:", float(obs[24][0]))
    print("[25] 场风 col:", np.flatnonzero(obs[25]).tolist(), "[26] 自风 col:", np.flatnonzero(obs[26]).tolist())
    print("[28:35] 宝牌指示牌:")
    for i in range(28,35):
        c=np.flatnonzero(obs[i])
        if len(c): print("   ch",i,[TILES[j] for j in c])
    print("[717] 剩余牌数:", round(float(obs[717][0])*69,2))
    print("[718:722] 自家宝牌数:", [round(float(obs[718+k][0])*12,2) for k in range(4)])
    print("[722] 未见宝牌:", round(float(obs[722][0])*23,2))
    print("[835] 已见牌计数(非零):", [(TILES[j],round(float(obs[835][j]),2)) for j in np.flatnonzero(obs[835])])
    print("[860] 听牌:", [TILES[j] for j in np.flatnonzero(obs[860])])
    print("[862:869] 向听 one-hot:", np.flatnonzero(obs[862:869].any(1)).tolist())
    print("[874:879] 可切牌特征 5 行:")
    for i in range(874,879):
        c=np.flatnonzero(obs[i])
        print("   ch",i,[TILES[j] for j in c] if len(c) else "-")

decode(obs)

# 推理
st=torch.load("/home/user/mahjong-llm/work/mortal_weights/voidshine-298k/mortal_298k.pth", weights_only=True, map_location="cpu")
cfg=st["config"]
brain=Brain(version=cfg["control"]["version"], conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"]).eval()
dqn=DQN(version=cfg["control"]["version"]).eval()
brain.load_state_dict(st["mortal"]); dqn.load_state_dict(st["current_dqn"])
with torch.inference_mode():
    x=torch.as_tensor(obs[None])
    m=torch.as_tensor(mask[None])
    phi=brain(x); q=dqn(phi,m)
print("\nphi:", tuple(phi.shape), " q:", tuple(q.shape))
top=torch.topk(q[0][torch.isfinite(q[0])],5)
print("top5 action:", top.indices.tolist(), "q:", [round(v,3) for v in top.values.tolist()])
print("解码:", [("打"+TILES[i] if i<34 else {34:"打5mr",35:"打5pr",36:"打5sr",37:"立直",38:"吃低",39:"吃中",40:"吃高",41:"碰",42:"杠",43:"和",44:"九种九牌",45:"pass"}[i]) for i in top.indices.tolist()])
