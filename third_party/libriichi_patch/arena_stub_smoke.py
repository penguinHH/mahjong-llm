#!/usr/bin/env python3
"""Arena smoke test with stub engines - no LLM, no Mortal weights.

Runs `OneVsThree.py_vs_py` with two Python stub engines:
  - challenger: engine_type="mortal2" -> react_batch(obs, masks, obs_raw)
  - champion:   engine_type="mortal"  -> react_batch(obs, masks, invisible_obs)

Both stubs pick the lowest-id legal action from each mask. This exercises:
  - the new Moratl2 dispatch in py_agent.rs
  - raw obs passing (RawObsEvent list) in agent/mortal.rs
  - the flat action-list return path (vs the 4-tuple for stock engines)
  - tolerant attribute extraction (stub LLM has no is_oracle /
    enable_quick_eval / enable_rule_based_agari_guard attributes)

Usage (WSL):
  venv/bin/python arena_stub_smoke.py --lib-dir ~/mahjong-llm/Mortal/target/release \
      --seed-count 4 [--log-dir ~/mahjong-llm/work/arena_smoke]
"""
import argparse
import os
import sys

import numpy as np

# Prefer rare/risky decision paths when legal: hora > kan > reach.
_PREFERRED = (43, 42, 37)


def pick_action(mask):
    legal = np.flatnonzero(mask)
    assert len(legal) > 0, "engine received an empty legal-action mask"
    legal_set = {int(x) for x in legal}
    for p in _PREFERRED:
        if p in legal_set:
            return p
    return int(legal[0])


class StubLLMEngine:
    engine_type = "mortal2"
    version = 4
    name = "stub-llm"

    def __init__(self):
        self.calls = 0
        self.actions = 0

    def react_batch(self, obs, masks, obs_raw):
        assert obs_raw is not None, "mortal2 engine must receive obs_raw, not None"
        assert len(obs_raw) == len(obs) == len(masks), (
            f"batch mismatch: obs={len(obs)} masks={len(masks)} raw={len(obs_raw)}"
        )
        self.calls += 1
        self.actions += len(obs)
        return [pick_action(m) for m in masks]


class StubMortalEngine:
    engine_type = "mortal"
    version = 4
    name = "stub-mortal"
    is_oracle = False
    enable_quick_eval = False
    enable_rule_based_agari_guard = False

    def __init__(self):
        self.calls = 0

    def react_batch(self, obs, masks, invisible_obs):
        assert invisible_obs is None, "is_oracle=False must receive invisible_obs=None"
        self.calls += 1
        actions, q_values, masks_recv, is_greedy = [], [], [], []
        for m in masks:
            a = pick_action(m)
            actions.append(a)
            q = [0.0] * len(m)
            q[a] = 1.0
            q_values.append(q)
            masks_recv.append([bool(x) for x in m])
            is_greedy.append(True)
        return actions, q_values, masks_recv, is_greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lib-dir",
        default=os.path.expanduser("~/mahjong-llm/Mortal/target/release"),
        help="directory containing the freshly built libriichi.so",
    )
    ap.add_argument("--seed-count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--key", type=int, default=42)
    ap.add_argument("--log-dir", default=None, help="optional mjai log output dir")
    args = ap.parse_args()

    sys.path.insert(0, args.lib_dir)
    import libriichi

    print("libriichi:", libriichi.__file__)
    from libriichi.arena import OneVsThree

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    llm = StubLLMEngine()
    mortal = StubMortalEngine()
    env = OneVsThree(disable_progress_bar=True, log_dir=args.log_dir)
    rankings = env.py_vs_py(
        llm,
        mortal,
        (args.seed, args.key),
        args.seed_count,
    )
    total = sum(rankings)
    print(f"rankings: {rankings} (games={total})")
    print(f"stub-llm: react_batch calls={llm.calls}, decisions={llm.actions}")
    print(f"stub-mortal: react_batch calls={mortal.calls}")

    seeds = args.seed_count
    assert total in (seeds, 4 * seeds), f"unexpected game count {total} for {seeds} seeds"
    assert llm.actions > 0
    print("ARENA SMOKE OK")


if __name__ == "__main__":
    main()
