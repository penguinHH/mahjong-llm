# libriichi patch (AGPL-3.0)

論文の評価・データ生成コードが呼び出す `take_obs_raw()` と `engine_type = "mortal2"` は、公開版の Mortal には含まれていません。このディレクトリは、それを Mortal 上で作り直した変更です。

The paper's pipeline calls `take_obs_raw()` and uses `engine_type = "mortal2"`, neither of which exists in upstream Mortal. This directory re-implements both on top of Mortal.

- `mortal-0cff2b5.diff`：Mortal のコミット `0cff2b5` に対する差分（`libriichi/src/state/raw.rs` 新規、`state/mod.rs`・`dataset/gameplay.rs`・`agent/mortal.rs`・`agent/py_agent.rs` 変更）。
- `raw.rs`、`apply_patch.py`、`apply_eval_patch.py`、`build*.sh`：差分を当ててビルド・配置するスクリプト。
- `selftest_raw.py`、`arena_stub_smoke.py`、`arena_llmchain_dryrun.py`：LLM なしで動く自己テスト。
- `eval_patch_notes.md`：評価側パッチの設計メモ（中国語）。

Mortal（libriichi）の派生物なので、このディレクトリは AGPL-3.0 です（`LICENSE`）。リポジトリの他の部分は Apache-2.0 です。
Derived from Mortal (libriichi), so this directory is licensed under AGPL-3.0; the rest of the repository is Apache-2.0.
