#!/usr/bin/env python3
"""Follow-up fixes for an already-applied eval patch (WSL Mortal tree).

- add `use pyo3::types::PyList;` import
- wrap the mortal2 return tuple in `Ok(...)` so both if/else branches of the
  with_gil closure produce `Result<tuple, anyhow::Error>`

Usage: fix_eval_build.py <repo_root>
"""
import pathlib
import sys

M = pathlib.Path(sys.argv[1])
p = M / "libriichi/src/agent/mortal.rs"
s = p.read_text(encoding="utf-8")

old = "use pyo3::intern;\nuse pyo3::prelude::*;"
new = "use pyo3::intern;\nuse pyo3::prelude::*;\nuse pyo3::types::PyList;"
assert s.count(old) == 1, "import anchor not found exactly once"
s = s.replace(old, new)

old = (
    "                (\n"
    "                    actions,\n"
    "                    vec![[0.0; ACTION_SPACE]; batch_size],\n"
    "                    masks_meta.unwrap_or_default(),\n"
    "                    vec![true; batch_size],\n"
    "                )\n"
)
new = (
    "                Ok((\n"
    "                    actions,\n"
    "                    vec![[0.0; ACTION_SPACE]; batch_size],\n"
    "                    masks_meta.unwrap_or_default(),\n"
    "                    vec![true; batch_size],\n"
    "                ))\n"
)
assert s.count(old) == 1, "return tuple anchor not found exactly once"
s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("FIXED libriichi/src/agent/mortal.rs")
