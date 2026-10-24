#!/usr/bin/env python3
"""Quick sanity check that libriichi is importable from the paper repo layout.

Run with the venv python after copying libriichi.so:
    ~/mahjong-llm/venv/bin/python tools/check_libriichi.py
"""
import os
import sys

PAPER = os.environ.get("LLM4CARD", "/mnt/d/mahjong-llm/refs/LLM4CardGame")
MJ_DIR = os.path.join(PAPER, "util", "mahjong_util")
sys.path.insert(0, MJ_DIR)
sys.path.insert(0, PAPER)

fail = 0

try:
    import libriichi
    print(f"[ok] import libriichi: {libriichi.__file__}")
except Exception as e:  # noqa: BLE001
    print(f"[FAIL] import libriichi: {e}")
    sys.exit(1)

try:
    from libriichi.dataset import GameplayLoader  # noqa: F401
    print("[ok] libriichi.dataset.GameplayLoader")
except Exception as e:  # noqa: BLE001
    print(f"[FAIL] GameplayLoader: {e}")
    fail += 1

try:
    from libriichi.arena import OneVsThree  # noqa: F401
    print("[ok] libriichi.arena.OneVsThree")
except Exception as e:  # noqa: BLE001
    print(f"[FAIL] OneVsThree: {e}")
    fail += 1

try:
    shape = libriichi.obs_shape(4)
    print(f"[ok] obs_shape(4) = {shape}")
except Exception as e:  # noqa: BLE001
    print(f"[FAIL] obs_shape(4): {e}")
    fail += 1

try:
    from libriichi import consts  # noqa: F401
    print(f"[ok] libriichi.consts (ACTION_SPACE={getattr(consts, 'ACTION_SPACE', '?')})")
except Exception as e:  # noqa: BLE001
    print(f"[FAIL] consts: {e}")
    fail += 1

print("ALL OK" if fail == 0 else f"{fail} check(s) failed")
sys.exit(1 if fail else 0)
