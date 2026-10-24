#!/usr/bin/env python3
"""Self-test for the take_obs_raw libriichi patch.

- takes a staged .json.gz log (or stages one from the 2026 pack)
- loads it with the paper-style GameplayLoader
- asserts take_obs/take_masks/take_actions/take_obs_raw length alignment
- asserts at_turn/shanten sequences from raw match take_at_turns/take_shantens
- dumps every field of the first and last raw event (JSON serializable check)
- runs convert_to_mjai_tile / obtain_legal_actions exactly like gen_action_data
- repeats the alignment checks over a small batch of files
"""
import glob
import gzip
import json
import os
import sys

PAPER = "/mnt/d/mahjong-llm/refs/LLM4CardGame"
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")
sys.path.insert(0, MJ_DIR)
sys.path.insert(0, PAPER)

import libriichi  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402
from consts import convert_to_mjai_tile, obtain_legal_actions  # noqa: E402

print("libriichi:", libriichi.__file__)

_GP = getattr(getattr(libriichi, "dataset", None), "Gameplay", None)
if _GP is not None and not hasattr(_GP, "take_obs_raw"):
    print("FATAL: take_obs_raw missing on Gameplay -> .so not replaced?")
    sys.exit(1)


def stage_miner():
    cands = sorted(glob.glob(os.path.expanduser("~/mahjong-llm/work/*/logs/*.json.gz")))
    if cands:
        return cands
    import zstandard

    srcs = sorted(glob.glob("/mnt/d/mahjong-llm/data/raw/tenhou-houou-2026/*.mjai.json.zst"))
    dst_dir = os.path.expanduser("~/mahjong-llm/work/selftest/logs")
    os.makedirs(dst_dir, exist_ok=True)
    n = 10
    for src in srcs[:n]:
        dst = os.path.join(dst_dir, os.path.basename(src).replace(".mjai.json.zst", ".json.gz"))
        if os.path.exists(dst):
            continue
        with open(src, "rb") as fin, gzip.open(dst, "wb") as fout:
            zstandard.ZstdDecompressor().copy_stream(fin, fout)
    return sorted(glob.glob(os.path.join(dst_dir, "*.json.gz")))


files = stage_miner()
loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)

def dump_fields(ev, tag):
    fields = {
        "player_id": ev.player_id, "bakaze": ev.bakaze(), "jikaze": ev.jikaze(),
        "kyoku": ev.kyoku, "honba": ev.honba, "kyotaku": ev.kyotaku, "is_oya": ev.is_oya,
        "scores": ev.scores, "rank": ev.rank, "at_turn": ev.at_turn,
        "tiles_left": ev.tiles_left, "shanten": ev.shanten,
        "tehai": ev.tehai, "waits": ev.waits(), "dora_indicators": ev.dora_indicators(),
        "doras_owned": ev.doras_owned, "akas_in_hand": ev.akas_in_hand,
        "doras_seen": ev.doras_seen, "akas_seen": ev.akas_seen,
        "tiles_seen": ev.tiles_seen(),
        "keep_shanten_discards": ev.keep_shanten_discards(),
        "next_shanten_discards": ev.next_shanten_discards(),
        "ankan_candidates": ev.ankan_candidates(), "kakan_candidates": ev.kakan_candidates(),
        "kawa": ev.kawa(), "kawa_overview": ev.kawa_overview(),
        "fuuro_overview": ev.fuuro_overview(), "ankan_overview": ev.ankan_overview(),
        "last_tedashis": ev.last_tedashis(), "riichi_sutehais": ev.riichi_sutehais(),
        "last_drew_tile_self": ev.last_self_tsumo(), "last_discarded_tile": ev.last_kawa_tile(),
        "riichi_declared": ev.riichi_declared, "riichi_accepted": ev.riichi_accepted,
        "can_w_riichi": ev.can_w_riichi, "is_w_riichi": ev.is_w_riichi,
        "at_furiten": ev.at_furiten, "is_menzen": ev.is_menzen,
    }
    json.dumps(fields)  # must be JSON serializable
    print(f"== {tag} ==")
    txt = json.dumps(fields, ensure_ascii=False)
    print(txt[:1800])
    return fields


n_files = 0
fail = 0
for f in files[:10]:
    games_ls = loader.load_gz_log_files([f])
    n_games = len(games_ls[0])
    for gi, g in enumerate(games_ls[0]):
        obs = g.take_obs()
        masks = g.take_masks()
        actions = g.take_actions()
        raw = g.take_obs_raw()
        at_turns = g.take_at_turns()
        shantens = g.take_shantens()
        dones = g.take_dones()
        ok = (
            len(obs) == len(raw) == len(masks) == len(actions)
            == len(at_turns) == len(shantens) == len(dones)
        )
        if not ok:
            fail += 1
            print(f"ALIGNMENT FAIL {os.path.basename(f)} game#{gi}: "
                  f"obs={len(obs)} raw={len(raw)} masks={len(masks)} "
                  f"actions={len(actions)} at_turns={len(at_turns)}")
            continue
        if [e.at_turn for e in raw] != list(at_turns):
            fail += 1
            print(f"AT_TURN MISMATCH {os.path.basename(f)} game#{gi}")
        if [e.shanten for e in raw] != list(shantens):
            fail += 1
            print(f"SHANTEN MISMATCH {os.path.basename(f)} game#{gi}")
        if n_files == 0 and gi == 0:
            print(f"file={os.path.basename(f)} games_in_file={n_games} n_entries={len(raw)}")
            ev = raw[0]
            dump_fields(ev, "FIRST entry")
            hands = convert_to_mjai_tile(ev.tehai, ev.akas_in_hand)
            print("my_hands:", hands)
            print("legal_actions(sample):", obtain_legal_actions(masks[0])[:12])
            dump_fields(raw[-1], "LAST entry")
    n_files += 1
    print(f"[{n_files}] {os.path.basename(f)}: {n_games} POV games OK")

print(f"checked {n_files} files, failures={fail}")
if fail:
    sys.exit(1)
print("SELFTEST PASSED")
