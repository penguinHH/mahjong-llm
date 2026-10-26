# libriichi patch (AGPL-3.0)

論文の評価・データ生成コードが呼び出す `take_obs_raw()` と `engine_type = "mortal2"` は、公開版の Mortal には含まれていません。このディレクトリは、それを Mortal 上で作り直した変更です。

The paper's pipeline calls `take_obs_raw()` and uses `engine_type = "mortal2"`, neither of which exists in upstream Mortal. This directory re-implements both on top of Mortal.

## 差分 / Diffs（Mortal `0cff2b5` に対する `git apply` 形式）

| ファイル | 内容 |
|---|---|
| `0001-take-obs-raw.diff` | データ側：`libriichi/src/state/raw.rs`（新規、`RawObsEvent` と `PlayerState::snapshot_raw()`）、`state/mod.rs`、`dataset/gameplay.rs`（`Gameplay::take_obs_raw()`） |
| `0002-mortal2-engine.diff` | 評価側：`agent/py_agent.rs`（`EngineType::Mortal2`）、`agent/mortal.rs`（`send_obs_raw` モード）。0001 が前提 |

```bash
git clone https://github.com/Equim-chan/Mortal && git -C Mortal checkout 0cff2b5
bash apply.sh Mortal            # 0001 → 0002 の順に、確認してから適用（適用済みならスキップ）
# 手動なら: git -C Mortal apply 0001-take-obs-raw.diff 0002-mortal2-engine.diff
# 元に戻す: git -C Mortal apply -R 0002-mortal2-engine.diff 0001-take-obs-raw.diff
```

## スクリプト / Scripts

- `build.sh [all|0001|0002] [--no-deploy]`：適用 → `cargo build`（release、pymod、mimalloc なし）→ 論文ツリーへ配置 → `selftest_raw.py`。
- `deploy_eval.sh`：ビルド済みの `.so` を配置して自己テストだけ行う。
- `selftest_raw.py`、`probe_types.py`：`take_obs_raw()` の整合・JSON 化のテスト。
- `arena_stub_smoke.py`（`run_arena_smoke.sh`）、`arena_llmchain_dryrun.py`（`check_and_dryrun.sh`）、`inspect_arena_logs.sh`：LLM なしで動く mortal2 経路のテスト。
- `eval_patch_notes.md`：評価側パッチの設計メモ（中国語）。

パスは環境変数で指定します（既定値はこのプロジェクトの WSL 上の配置）：`MORTAL_DIR`、`LLM4CARDGAME_DIR`、`VENV_PY`、`TENHOU_MJAI_DIR`。

Mortal（libriichi）の派生物なので、このディレクトリは AGPL-3.0 です（`LICENSE`）。リポジトリの他の部分は Apache-2.0 です。
Derived from Mortal (libriichi), so this directory is licensed under AGPL-3.0; the rest of the repository is Apache-2.0.
