"""Apply the eval-side `mortal2` patch onto the Mortal clone (WSL side).

Usage: python apply_eval_patch.py <mortar_repo_root>

Adds an `engine_type == "mortal2"` Python-engine variant so that the arena
can drive LLM engines whose `react_batch(obs, masks, obs_raw)` expects raw
state snapshots (`RawObsEvent`, from the dataset patch) and returns a plain
list of action ids instead of the `(actions, q_values, masks, is_greedy)`
tuple used by regular Mortal engines.

Every replacement asserts an exact occurrence count so a stale/dirty tree
fails loudly instead of silently mis-patching.
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


# --- libriichi/src/agent/py_agent.rs --------------------------------------
patch(
    "libriichi/src/agent/py_agent.rs",
    "enum EngineType {\n    Mortal,\n    MjaiLog,\n}",
    "enum EngineType {\n    Mortal,\n    Mortal2,\n    MjaiLog,\n}",
)
patch(
    "libriichi/src/agent/py_agent.rs",
    '            "mortal" => Ok(Self::Mortal),\n            "mjai-log" => Ok(Self::MjaiLog),',
    '            "mortal" => Ok(Self::Mortal),\n            "mortal2" => Ok(Self::Mortal2),\n'
    '            "mjai-log" => Ok(Self::MjaiLog),',
)
patch(
    "libriichi/src/agent/py_agent.rs",
    "        EngineType::Mortal => Box::new(MortalBatchAgent::new(engine, player_ids)?) as _,\n",
    "        EngineType::Mortal => Box::new(MortalBatchAgent::new(engine, player_ids)?) as _,\n"
    "        EngineType::Mortal2 => {\n"
    "            Box::new(MortalBatchAgent::new_with_raw(engine, player_ids)?) as _\n"
    "        }\n",
)

# --- libriichi/src/agent/mortal.rs -----------------------------------------
patch(
    "libriichi/src/agent/mortal.rs",
    "use crate::state::PlayerState;",
    "use crate::state::{PlayerState, RawObsEvent};",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "use pyo3::intern;\nuse pyo3::prelude::*;",
    "use pyo3::intern;\nuse pyo3::prelude::*;\nuse pyo3::types::PyList;",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "pub struct MortalBatchAgent {",
    "/// Like `getattr` + `extract`, but returns `None` when the attribute is\n"
    "/// missing instead of failing. LLM engines (`engine_type == \"mortal2\"`)\n"
    "/// expose only a subset of the attributes regular Mortal engines define.\n"
    "fn optional_attr<'py, T: FromPyObject<'py>>(obj: &Bound<'py, PyAny>, name: &str) -> Option<T> {\n"
    "    obj.getattr(name).ok()?.extract().ok()\n"
    "}\n"
    "\n"
    "pub struct MortalBatchAgent {",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "    name: String,\n    player_ids: Vec<u8>,\n\n    actions: Vec<usize>,",
    "    name: String,\n    player_ids: Vec<u8>,\n"
    "    /// Call the engine with raw obs snapshots and accept a plain action\n"
    "    /// list (see [`Self::new_with_raw`]).\n"
    "    send_obs_raw: bool,\n"
    "\n    actions: Vec<usize>,",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "struct SyncFields {\n"
    "    states: Vec<Array2<f32>>,\n"
    "    invisible_states: Vec<Array2<f32>>,\n"
    "    masks: Vec<Array1<bool>>,\n"
    "    action_idxs: Vec<usize>,\n"
    "    kan_action_idxs: Vec<Option<usize>>,\n"
    "}",
    "struct SyncFields {\n"
    "    states: Vec<Array2<f32>>,\n"
    "    invisible_states: Vec<Array2<f32>>,\n"
    "    raw_states: Vec<RawObsEvent>,\n"
    "    masks: Vec<Array1<bool>>,\n"
    "    action_idxs: Vec<usize>,\n"
    "    kan_action_idxs: Vec<Option<usize>>,\n"
    "}",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "                let name = obj.getattr(\"name\")?.extract()?;\n"
    "                let is_oracle = obj.getattr(\"is_oracle\")?.extract()?;\n"
    "                let version = obj.getattr(\"version\")?.extract()?;\n"
    "                let enable_quick_eval = obj.getattr(\"enable_quick_eval\")?.extract()?;\n"
    "                let enable_rule_based_agari_guard =\n"
    "                    obj.getattr(\"enable_rule_based_agari_guard\")?.extract()?;",
    "                let name = obj.getattr(\"name\")?.extract()?;\n"
    "                let is_oracle = optional_attr(&obj, \"is_oracle\").unwrap_or(false);\n"
    "                let version = obj.getattr(\"version\")?.extract()?;\n"
    "                let enable_quick_eval = optional_attr(&obj, \"enable_quick_eval\").unwrap_or(false);\n"
    "                let enable_rule_based_agari_guard =\n"
    "                    optional_attr(&obj, \"enable_rule_based_agari_guard\").unwrap_or(false);",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "        let sync_fields = Arc::new(Mutex::new(SyncFields {\n"
    "            states: vec![],\n"
    "            invisible_states: vec![],\n"
    "            masks: vec![],",
    "        let sync_fields = Arc::new(Mutex::new(SyncFields {\n"
    "            states: vec![],\n"
    "            invisible_states: vec![],\n"
    "            raw_states: vec![],\n"
    "            masks: vec![],",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "            name,\n            player_ids: player_ids.to_vec(),\n",
    "            name,\n            player_ids: player_ids.to_vec(),\n            send_obs_raw: false,\n",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "    }\n\n    fn evaluate(&mut self) -> Result<()> {",
    "    }\n"
    "\n"
    "    /// Same as [`Self::new`], but the engine is called as\n"
    "    /// `react_batch(obs, masks, obs_raw)` with raw state snapshots and may\n"
    "    /// return a plain list of action ids instead of the\n"
    "    /// `(actions, q_values, masks, is_greedy)` tuple (LLM engines,\n"
    "    /// `engine_type == \"mortal2\"`).\n"
    "    pub fn new_with_raw(engine: PyObject, player_ids: &[u8]) -> Result<Self> {\n"
    "        let mut agent = Self::new(engine, player_ids)?;\n"
    "        agent.send_obs_raw = true;\n"
    "        Ok(agent)\n"
    "    }\n"
    "\n    fn evaluate(&mut self) -> Result<()> {",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "        let version = self.version;\n        let state = state.clone();",
    "        let version = self.version;\n        let send_obs_raw = self.send_obs_raw;\n        let state = state.clone();",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "            let kan = need_kan_select.then(|| state.encode_obs(version, true));\n"
    "            let (feature, mask) = state.encode_obs(version, false);\n",
    "            let kan = need_kan_select.then(|| state.encode_obs(version, true));\n"
    "            let (feature, mask) = state.encode_obs(version, false);\n"
    "\n"
    "            let raw = send_obs_raw.then(|| state.snapshot_raw());\n",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "            let SyncFields {\n"
    "                states,\n"
    "                invisible_states,\n"
    "                masks,\n"
    "                action_idxs,\n"
    "                kan_action_idxs,\n"
    "            } = &mut *sync_fields.lock();",
    "            let SyncFields {\n"
    "                states,\n"
    "                invisible_states,\n"
    "                raw_states,\n"
    "                masks,\n"
    "                action_idxs,\n"
    "                kan_action_idxs,\n"
    "            } = &mut *sync_fields.lock();",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "                if let Some(invisible_state) = invisible_state.clone() {\n"
    "                    invisible_states.push(invisible_state);\n"
    "                }\n"
    "            }",
    "                if let Some(invisible_state) = invisible_state.clone() {\n"
    "                    invisible_states.push(invisible_state);\n"
    "                }\n"
    "                if let Some(raw) = &raw {\n"
    "                    raw_states.push(raw.clone());\n"
    "                }\n"
    "            }",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "            action_idxs[index] = states.len();\n"
    "            states.push(feature);\n"
    "            masks.push(mask);\n"
    "            if let Some(invisible_state) = invisible_state {\n"
    "                invisible_states.push(invisible_state);\n"
    "            }\n"
    "        });",
    "            action_idxs[index] = states.len();\n"
    "            states.push(feature);\n"
    "            masks.push(mask);\n"
    "            if let Some(invisible_state) = invisible_state {\n"
    "                invisible_states.push(invisible_state);\n"
    "            }\n"
    "            if let Some(raw) = raw {\n"
    "                raw_states.push(raw);\n"
    "            }\n"
    "        });",
)
patch(
    "libriichi/src/agent/mortal.rs",
    "            let masks: Vec<_> = sync_fields\n"
    "                .masks\n"
    "                .drain(..)\n"
    "                .map(|v| PyArray1::from_owned_array(py, v))\n"
    "                .collect();\n"
    "            let invisible_states: Option<Vec<_>> = self.is_oracle.then(|| {\n"
    "                sync_fields\n"
    "                    .invisible_states\n"
    "                    .drain(..)\n"
    "                    .map(|v| PyArray2::from_owned_array(py, v))\n"
    "                    .collect()\n"
    "            });\n"
    "\n"
    "            let args = (states, masks, invisible_states);\n"
    "            self.engine\n"
    "                .bind_borrowed(py)\n"
    "                .call_method1(intern!(py, \"react_batch\"), args)\n"
    "                .context(\"failed to execute `react_batch` on Python engine\")?\n"
    "                .extract()\n"
    "                .context(\"failed to extract to Rust type\")",
    "            let masks_meta: Option<Vec<[bool; ACTION_SPACE]>> =\n"
    "                self.send_obs_raw.then(|| {\n"
    "                    sync_fields\n"
    "                        .masks\n"
    "                        .iter()\n"
    "                        .map(|row| {\n"
    "                            let mut arr = [false; ACTION_SPACE];\n"
    "                            for (dst, &src) in arr.iter_mut().zip(row.iter()) {\n"
    "                                *dst = src;\n"
    "                            }\n"
    "                            arr\n"
    "                        })\n"
    "                        .collect()\n"
    "                });\n"
    "            let masks: Vec<_> = sync_fields\n"
    "                .masks\n"
    "                .drain(..)\n"
    "                .map(|v| PyArray1::from_owned_array(py, v))\n"
    "                .collect();\n"
    "            let third: PyObject = if self.send_obs_raw {\n"
    "                let raws = sync_fields\n"
    "                    .raw_states\n"
    "                    .drain(..)\n"
    "                    .map(|r| Py::new(py, r))\n"
    "                    .collect::<PyResult<Vec<_>>>()?;\n"
    "                PyList::new(py, raws)?.into_any().unbind()\n"
    "            } else {\n"
    "                let invisible_states: Option<Vec<_>> = self.is_oracle.then(|| {\n"
    "                    sync_fields\n"
    "                        .invisible_states\n"
    "                        .drain(..)\n"
    "                        .map(|v| PyArray2::from_owned_array(py, v))\n"
    "                        .collect()\n"
    "                });\n"
    "                match invisible_states {\n"
    "                    Some(v) => PyList::new(py, v)?.into_any().unbind(),\n"
    "                    None => py.None(),\n"
    "                }\n"
    "            };\n"
    "\n"
    "            let args = (states, masks, third);\n"
    "            let result = self\n"
    "                .engine\n"
    "                .bind_borrowed(py)\n"
    "                .call_method1(intern!(py, \"react_batch\"), args)\n"
    "                .context(\"failed to execute `react_batch` on Python engine\")?;\n"
    "            if self.send_obs_raw {\n"
    "                let actions: Vec<usize> = result\n"
    "                    .extract()\n"
    "                    .context(\"failed to extract action list from engine\")?;\n"
    "                let batch_size = actions.len();\n"
    "                Ok((\n"
    "                    actions,\n"
    "                    vec![[0.0; ACTION_SPACE]; batch_size],\n"
    "                    masks_meta.unwrap_or_default(),\n"
    "                    vec![true; batch_size],\n"
    "                ))\n"
    "            } else {\n"
    "                result\n"
    "                    .extract()\n"
    "                    .context(\"failed to extract to Rust type\")\n"
    "            }",
)

print("ALL EVAL PATCHES OK")
