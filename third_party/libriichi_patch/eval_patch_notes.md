# 评估侧补丁（mortal2）设计笔记

目标：让论文评估链路 `util/mortal_util/one_vs_three.py`（`LLMEngine.react_batch(obs, masks, obs_raw)`）
与本地 libriichi arena 对接。**不加载任何 LLM 即已完成设计还原与验证。**

## 架构还原（证据链）

- `util/mortal_util/one_vs_three.py` → `OneVsThree.py_vs_py(challenger, champion, …)`
  → `libriichi/src/arena/one_vs_three.rs::py_vs_py` → `new_py_agent(...)`。
- `libriichi/src/agent/py_agent.rs::new_py_agent` 读取引擎对象的 `engine_type` 字符串分派
  agent 类型；官方 master 仅 `"mortal"`（MortalBatchAgent）与 `"mjai-log"`。
- `LLMEngine`（`util/mortal_util/llm_engine.py`）把 `engine_type` 设为 **`"mortal2"`**
  ——天然是作者私有补丁的挂点（否则 master 会 `unknown engine type` 直接 bail）。
- `MortalBatchAgent::evaluate()` 批量调 `react_batch`：
  - 官方第三参 = invisible_states（`is_oracle` 才非 None，否则 None）；
  - LLM 引擎要第三参 = **raw obs（RawObsEvent 列表）**，且返回**扁平动作 id 列表**，
    而非 Mortal 的 `(actions, q_values, masks, is_greedy)` 四元组。
- `LLMEngine` 缺 `is_oracle / enable_quick_eval / enable_rule_based_agari_guard` 属性
  → 官方 `MortalBatchAgent::new` 会 AttributeError；补丁必须容错。
- 动作 id 语义两边一致：0..=36 切牌、37 立直、38-40 吃、41 碰、42 杠、43 和、45 pass
  （`LLMEngine.step` 返回 `convert_action_str_to_id` 的全局 id ✓）。

## 实现（WSL `~/mahjong-llm/Mortal`，与数据侧补丁同一棵树）

`libriichi/src/agent/py_agent.rs`：
- `EngineType` 增加 `Mortal2`；`"mortal2" => MortalBatchAgent::new_with_raw(...)`。

`libriichi/src/agent/mortal.rs`：
- `MortalBatchAgent` 新增字段 `send_obs_raw: bool`；`new()` 签名不变（默认 false），
  新增 `new_with_raw()`。`mjai/bot.rs` 既有调用方零改动。
- `SyncFields` 新增 `raw_states: Vec<RawObsEvent>`；`set_scene` 的 rayon 编码任务里
  `let raw = send_obs_raw.then(|| state.snapshot_raw());`，与 states 同步 push
  （kan 条目 + 主条目同序）→ `actions[i]` / `kan_action_idxs[i]` 索引对齐保持不变。
- `evaluate()`（raw 模式）：第三参 = `PyList<RawObsEvent>`；非 raw 模式完全原样。
  返回值接受 `Vec<usize>`；合成 `q_values=全 0`、`masks=真实掩码`、`is_greedy=true`
  （gen_meta 仅写日志 meta，不参与决策；`enable_rule_based_agari_guard=false` 不读 q）。
- 属性容错：上述三属性缺失时默认 false（`optional_attr`）；`name / version` 仍必需。
- 编译坑（已修入 apply_eval_patch.py）：`PyList` 需 `use pyo3::types::PyList;`；
  raw 分支返回值需包 `Ok(...)`（闭包整体返回 `Result<tuple, anyhow::Error>`）。

## 验证（无 LLM 桩引擎冒烟）

`work/patch/arena_stub_smoke.py`：challenger=mortal2（无三个可选属性、返回扁平 id）；
champion=mortal（标准四元组）；策略：合法动作优先 43[hora] > 42[kan] > 37[reach] > 最低 id。

- 构建 `build_eval.sh`：2m39s → `target/release/libriichi.so`。
- 冒烟①（seed_count=2, seed=10000）：8 局，rankings [2,2,2,2]；llm 1050 批 / 1798 决策。
- 冒烟②（seed_count=4, seed=20000）：16 局，rankings [4,4,4,4]；决策 3354。
  事件普查：**ankan 16 / daiminkan 44 / kakan 28 / hora 20 / reach 4 / dora 88 /
  ryukyoku 136 / chi 880 / pon 472** —— kan 选择路径（`actions[kan_idx]` 索引对齐）实测通过。
- mjai 日志（`~/mahjong-llm/work/arena_smoke*`）：names/meta 完整，mask_bits 为真实掩码。
- 局数语义：`seed_count` 个种子 × 4 座次轮转 = 4×seed_count 局；challenger 单 agent 覆盖 4 座。

## 端到端干跑（无 LLM，真 LLMEngine + 假 llm_function）

`work/patch/arena_llmchain_dryrun.py`：把论文原版 `util/mortal_util/llm_engine.py`
（真 `LLMEngine`）接进补丁版 arena，monkeypatch `llm_function` 返回空串（不可解析 →
LLMEngine 自动回退 `random.choice(legal_actions)`），**全程零 HTTP / 零 LLM**：

- 结果：4 局完成（rankings [0,0,2,2]）；LLMEngine 被问 1166 次；prompt 实渲染
  3175 字符（论文原版文本）；format-correct=0（预期，假输出永不合格）。
- 覆盖链：`parse_riichi_state(RawObsEvent, mask)` → `prompt_function_riichi` →
  `llm_function` → `out_parse_function_riichi` → 合法动作回退 →
  `convert_action_str_to_id` → mortal2 Rust 路径。即评估 Python 侧全链路（除真 LLM
  响应外）已通。
- 依赖：venv 补装 `openai requests tenacity rlcard`（`install_eval_pydeps.sh`；rlcard
  是论文评估路径的公开依赖，`one_vs_three.py` 用 `rlcard.utils.set_seed`）。
- 注意：干跑经 `sys.path` 优先用 `target/release` 新 .so；真实评估前必须完成部署。

## 待办（评估阶段开始时）

1. **部署**（等全量数据任务结束）：`cp target/release/libriichi.so /mnt/d/.../mahjong_util/`
   （先备份）→ 重跑 `selftest_raw.py` + 小样本管线冒烟确认数据行为不变。
2. **Mortal 权重**：`util/mortal_util/config.py` champion `state_file` → 社区权重
   `VoidShine/mortal-298k`；核对 `state['config']['control']['version']` 与 Brain/DQN 版本。
3. **LLM 服务**：`util/llm_config.py`（model_config）+ OpenAI 兼容端点（训练后 vLLM + LoRA）。
   注意 base_url 硬编码 `0.0.0.0:8555`（`API_PORT` 可改端口）；若 vLLM 跑在 Windows 侧需改 host。
4. `scripts/eval_riichi.sh` 参数已读：`$1 seed $2 num_games $3 cuda $4 model_chal $5 model_champ $6 out_dir`
   → `python -u -m util.mortal_util.one_vs_three --out_dir $6 --models $4 $5 --cuda $3 --num_games $2 --seed $1`。
5. 记录：`one_vs_three.py` L65-68 `else: engine_chal = LLMEngine(...)` 系笔误（应为
   engine_cham），默认 `--models llm mortal` 不受影响，暂不改动。
6. ~~prompt 兼容性干跑~~ ✅ 已完成（见上节「端到端干跑」）。

## 文件清单（work/patch/）

- `apply_eval_patch.py`：补丁源（每处替换带出现次数断言；含 PyList/Ok 修正）
- `fix_eval_build.py` / `fix_then_build.sh`：对已打补丁树的一次性修正（本次用）
- `build_eval.sh`：幂等 补丁+编译（产出 target/release/libriichi.so；不部署）
- `arena_stub_smoke.py` / `run_arena_smoke.sh`：桩引擎冒烟（参数：seed_count / seed / log_dir）
- `inspect_arena_logs.sh`：日志普查（文件数 + 事件计数）
