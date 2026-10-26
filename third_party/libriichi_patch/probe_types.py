import glob
import json
import os
import sys

PAPER = os.environ.get("LLM4CARDGAME_DIR", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")
sys.path.insert(0, MJ_DIR)
sys.path.insert(0, PAPER)

from libriichi.dataset import GameplayLoader  # noqa: E402

f = sorted(glob.glob(os.path.expanduser("~/mahjong-llm/work/selftest/logs/*.json.gz")))[0]
loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)
games_ls = loader.load_gz_log_files([f])
g = games_ls[0][0]
raw = g.take_obs_raw()
ev = raw[0]

attrs = ["player_id", "kyoku", "honba", "kyotaku", "is_oya", "scores", "rank",
         "at_turn", "tiles_left", "shanten", "tehai", "akas_in_hand", "doras_owned",
         "doras_seen", "akas_seen", "riichi_declared", "riichi_accepted",
         "can_w_riichi", "is_w_riichi", "at_furiten", "is_menzen"]
for name in attrs:
    v = getattr(ev, name)
    print(f"{name}: {type(v).__name__} = {v!r}"[:160])

methods = ["bakaze", "jikaze", "waits", "dora_indicators", "tiles_seen",
           "keep_shanten_discards", "next_shanten_discards", "ankan_candidates",
           "kakan_candidates", "kawa", "kawa_overview", "fuuro_overview",
           "ankan_overview", "last_tedashis", "riichi_sutehais",
           "last_self_tsumo", "last_kawa_tile"]
for name in methods:
    v = getattr(ev, name)()
    print(f"{name}(): {type(v).__name__} = {v!r}"[:200])

try:
    json.dumps({"tehai": ev.tehai})
    print("tehai json OK")
except TypeError as e:
    print("tehai json FAIL:", e)
try:
    json.dumps({"doras_owned": ev.doras_owned})
    print("doras_owned json OK")
except TypeError as e:
    print("doras_owned json FAIL:", e)
