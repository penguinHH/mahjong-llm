"""Apply the take_obs_raw patch onto the Mortal clone (WSL side).

Usage: python apply_patch.py <mortar_repo_root>

Every replacement asserts an exact occurrence count so that a stale/dirty
tree fails loudly instead of silently mis-patching.
"""
import pathlib
import sys

M = pathlib.Path(sys.argv[1])


def patch(rel: str, old: str, new: str, count: int = 1) -> None:
    p = M / rel
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    assert n == count, f"{rel}: expected {count} occurrence(s), found {n}:\n---\n{old}\n---"
    s = s.replace(old, new, count)
    p.write_text(s, encoding="utf-8")
    print(f"patched {rel}")


# --- libriichi/src/state/mod.rs -------------------------------------------
patch(
    "libriichi/src/state/mod.rs",
    "mod player_state;\nmod sp_tables;",
    "mod player_state;\nmod raw;\nmod sp_tables;",
)
patch(
    "libriichi/src/state/mod.rs",
    "pub use player_state::PlayerState;",
    "pub use player_state::PlayerState;\npub use raw::RawObsEvent;",
)
patch(
    "libriichi/src/state/mod.rs",
    "    m.add_class::<PlayerState>()?;",
    "    m.add_class::<PlayerState>()?;\n    m.add_class::<RawObsEvent>()?;",
)

# --- libriichi/src/dataset/gameplay.rs ------------------------------------
patch(
    "libriichi/src/dataset/gameplay.rs",
    "use crate::state::PlayerState;",
    "use crate::state::{PlayerState, RawObsEvent};",
)
patch(
    "libriichi/src/dataset/gameplay.rs",
    "    pub at_turns: Vec<u8>,\n    pub shantens: Vec<i8>,",
    "    pub at_turns: Vec<u8>,\n    pub shantens: Vec<i8>,\n    /// Per-decision raw state snapshot (aligned with obs/masks/actions).\n    pub obs_raw: Vec<RawObsEvent>,",
)
patch(
    "libriichi/src/dataset/gameplay.rs",
    "    fn take_shantens(&mut self) -> Vec<i8> {\n        mem::take(&mut self.shantens)\n    }",
    "    fn take_shantens(&mut self) -> Vec<i8> {\n        mem::take(&mut self.shantens)\n    }\n\n    fn take_obs_raw(&mut self) -> Vec<RawObsEvent> {\n        mem::take(&mut self.obs_raw)\n    }",
)
patch(
    "libriichi/src/dataset/gameplay.rs",
    "        self.at_turns.push(ctx.state.at_turn());\n        self.shantens.push(ctx.state.shanten());",
    "        self.at_turns.push(ctx.state.at_turn());\n        self.shantens.push(ctx.state.shanten());\n        self.obs_raw.push(ctx.state.snapshot_raw());",
)

print("ALL PATCHES OK")
