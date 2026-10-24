"""46 个动作（与 Mortal 的动作空间同构）和从局面张量算合法动作掩码。

下标：0-33 打牌（按 KINDS 顺序）、34-36 打赤五（5mr/5pr/5sr）、37 reach、38 chi_low、39 chi_mid、40 chi_high、
41 pon、42 kan、43 hora、44 ryukyoku、45 pass。动作字符串与 prompt / 标签一致（如 "dahai:5mr"、"pon"）。
"""
import numpy as np

from state_features import CALLS, IDX, KINDS, P, S

ACTIONS = [f"dahai:{k}" for k in KINDS] + ["dahai:5mr", "dahai:5pr", "dahai:5sr"] + CALLS
A = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)
AKA_POS = [IDX["5m"], IDX["5p"], IDX["5s"]]
assert N_ACTIONS == 46


def norm(action):
    """'dahai: 5m' → 'dahai:5m'。"""
    return action.replace(" ", "")


def legal_mask(planes, scalars):
    """planes uint8 [N, N_PLANES, 34]、scalars float [N, N_SCALARS] → bool [N, 46]。"""
    n = planes.shape[0]
    m = np.zeros((n, N_ACTIONS), bool)
    m[:, :34] = planes[:, P["legal_dahai"], :] > 0
    m[:, 34:37] = planes[:, P["legal_dahai_aka"]][:, AKA_POS] > 0
    m[:, 37:] = scalars[:, [S[f"legal_{c}"] for c in CALLS]] > 0
    return m


def kind(action):
    a = action.split(":")[0]
    return "chi" if a.startswith("chi") else a
