#!/usr/bin/env python3
"""End-to-end eval-chain dry run with a FAKE llm_function (no LLM server).

Wires the paper's LLMEngine (util/mortal_util/llm_engine.py) into the patched
libriichi arena:

  - arena: freshly built .so from ~/mahjong-llm/Mortal/target/release
    (contains the mortal2 eval patch)
  - paper modules: /mnt/d/mahjong-llm/refs/LLM4CardGame (util.*)
  - llm_function monkeypatched -> returns '' (unparseable) so LLMEngine falls
    back to a random legal action. This validates the full Python chain
    (parse_riichi_state -> prompt_function_riichi -> out_parse -> legality
    fallback -> convert_action_str_to_id) without any HTTP call.
  - champion: stub 'mortal' engine (4-tuple), so no torch / Mortal weights are
    needed either.

Note: no network/LLM traffic happens; the OpenAI client object is constructed
but never used because llm_function never reaches the API call.
"""
import os
import sys

BUILD_DIR = os.path.join(os.environ.get("MORTAL_DIR", os.path.expanduser("~/mahjong-llm/Mortal")), "target", "release")
PAPER_ROOT = os.environ.get("LLM4CARDGAME_DIR", "/mnt/d/mahjong-llm/refs/LLM4CardGame")

sys.path.insert(0, os.path.join(PAPER_ROOT, "util", "mahjong_util"))
sys.path.insert(0, PAPER_ROOT)
sys.path.insert(0, BUILD_DIR)  # last insert = highest priority; newest libriichi wins

import numpy as np  # noqa: E402

import libriichi  # noqa: E402

print("libriichi:", libriichi.__file__)
from libriichi.arena import OneVsThree  # noqa: E402

from util.mortal_util import llm_engine as le  # noqa: E402

prompts = []


def fake_llm_function(prompt, history=None, system=None, client=None, model_config=None):
    prompts.append(prompt)
    return ""  # unparseable -> LLMEngine falls back to a random legal action


le.llm_function = fake_llm_function

engine_llm = le.LLMEngine(4, model="llm")
print("engine_type:", engine_llm.engine_type, "| name:", engine_llm.name)


class StubMortalEngine:
    engine_type = "mortal"
    version = 4
    name = "stub-mortal"
    is_oracle = False
    enable_quick_eval = False
    enable_rule_based_agari_guard = False

    def react_batch(self, obs, masks, invisible_obs):
        assert invisible_obs is None
        actions, q_values, masks_recv, is_greedy = [], [], [], []
        for m in masks:
            legal = np.flatnonzero(m)
            a = int(legal[0])
            actions.append(a)
            q = [0.0] * len(m)
            q[a] = 1.0
            q_values.append(q)
            masks_recv.append([bool(x) for x in m])
            is_greedy.append(True)
        return actions, q_values, masks_recv, is_greedy


log_dir = os.path.expanduser("~/mahjong-llm/work/arena_dryrun")
os.makedirs(log_dir, exist_ok=True)

env = OneVsThree(disable_progress_bar=True, log_dir=log_dir)
rankings = env.py_vs_py(engine_llm, StubMortalEngine(), (30000, 42), 1)
print("rankings:", rankings)
print(
    "llm requests:",
    engine_llm.request_count,
    "| format-correct:",
    engine_llm.correct_count,
)

assert engine_llm.request_count > 0, "LLMEngine.step was never exercised"
assert len(prompts) == engine_llm.request_count, "prompt capture mismatch"
assert all(isinstance(p, str) and len(p) > 100 for p in prompts), "bad prompt"
assert sum(rankings) == 4, f"expected 4 games, got {sum(rankings)}"

print("first prompt length:", len(prompts[0]))
print("prompt head:")
print(prompts[0][:500])
print("EVAL CHAIN DRY RUN OK")
