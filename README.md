# 画像認識・麻雀・理由説明 / See, Play, Explain

小さなモデルに役割を分担させ、麻雀卓の画像から打牌を決め、日本語で理由を説明する研究用プロジェクトです。NeurIPS 2025 の論文 *Can Large Language Models Master Complex Card Games?*（arXiv 2509.01328）の再現から出発し、RTX 5070 Ti（16 GB）1 枚で「何がボトルネックか」を調べました。

A research project in which small models with divided roles read a mahjong table image, choose an action and explain it in Japanese. It started as a reproduction of *Can Large Language Models Master Complex Card Games?* (NeurIPS 2025) on a single 16 GB GPU.

> 学習済みの重みと牌譜データは公開していません（下記「含まれないもの」）。このリポジトリはコード・設定・評価スクリプト・結果のみです。
> Weights and game-log data are not included; this repository contains code, configs, evaluation scripts and results only.

## 各段階の最終結果 / Final results by stage

テスト 5,000 局面（学習より後の日付の対局）。Mortal 一致 = 強いオープンソース麻雀 AI「Mortal」の最善手と同じ手を選んだ割合。Mortal 自身の牌譜一致は 81.9%。

| 段階 / Stage | 構成 / Setup | 最終結果 / Result |
|---|---|---|
| ① データ / Data | 天鳳 鳳凰卓 12,188 局 → 174 万の判断局面、全合法手に Mortal の Q 値 | 174 万局面 |
| ② テキスト LLM | Qwen3.5-0.8B + LoRA、局面を約 1,300 トークンの文章で入力 | Mortal 一致 **76.8%**（2B は 77.8%）。後学習前の 0.8B は回答形式 0%、牌効率の貪欲ルールは 57.3% |
| ③ 局面エンコーダ / State encoder | Transformer 4.9M、局面テンソル 58×34 + 71、Mortal の Q 値で蒸留 | Mortal 一致 **94.7%**、牌譜一致 81.9%、Q 値差 0.014 |
| ④ 理由 LLM / Rationale | 凍結エンコーダの 35 トークン + プログラムが計算した事実 → Qwen3.5-0.8B | 理由の文単位精度 **92.8%**、理由全体の一致 80.2% |
| ⑤ 画像から / From pixels | 1024×768 の卓画像 → 知覚モジュール → ③ → ④ | Mortal 一致 **92.7%**、理由 90.7%（画像を直接読む VLM は 68.9%） |

**正直な注記 / Honest note**：④ の打牌はエンコーダの評価順に従うので、打牌の一致は ③ と同じです。評価順を渡さずに再学習すると、LLM 自身の判断で 94.5%（③ との差は有意でない、McNemar p = 0.134）。テキスト入力の 76.8% から、同じ 0.8B が局面トークンを読むだけで大きく伸びる、というのが主な発見です。
In ④ the move follows the encoder's ranking; retrained without it, the LLM alone reaches 94.5% (not significantly different from ③).

## 構成 / Pipeline

```
天鳳牌譜 (mjai) ──► libriichi（拡張：生の観測を出力）──► 局面 36 項目
   ├─ テキスト prompt ──────────────────────────► Qwen3.5 + LoRA ──► 行動          (train_small/)
   └─ 局面テンソル 58×34 + 71 ──► エンコーダ（Mortal Q で蒸留）               (mmj/)
          └─ 35 トークン ＋ 事実テキスト（facts.py）──► Qwen3.5-0.8B + LoRA ──► 理由 + 行動
卓画像 1024×768（自作レンダラー）──► 知覚（視覚塔 + 読み出し + ルール組み立て）──► 局面テンソル ──► 上と同じ   (vlm/, mm3/)
```

## リポジトリ構成 / Layout

| パス | 内容 |
|---|---|
| `tools/` | データ構築：牌譜 → 判断局面 → SFT データ、Mortal ラベル、事実付きの理由 |
| `train_small/` | テキスト LLM の LLaMA-Factory 設定、評価（`eval_vs_mortal.py`・`eval_reason.py`・`mcnemar.py`・`zero_shot_report.py` ほか）、GPU キューの runner |
| `mmj/` | 局面テンソル、エンコーダ、価値ヘッド、エンコーダ + LLM、事実テキスト |
| `vlm/` | 卓のレンダラー、パッチ単位の密なラベル、VLM の設定 |
| `mm3/` | 知覚モジュール、ルール後処理、画像からのエンドツーエンド評価、3 モーダル |
| `queue/` | 各実験を実際に実行したジョブスクリプト（再現用の正確なコマンド） |
| `modal/`, `colab/` | クラウド学習（Modal A100、Colab / Kaggle ノートブック生成） |
| `third_party/libriichi_patch/` | libriichi の拡張（AGPL-3.0、Mortal 由来） |
| `docs/` | 構成図（SVG） |

## 含まれないもの / Not included

| 内容 | 理由 |
|---|---|
| 牌譜データと派生データ | 天鳳の牌譜は研究目的のみの利用に限られるため |
| 学習済みの重み（LLM の LoRA、エンコーダ、知覚モジュール） | すべて牌譜由来のため |
| Mortal の重み | 非公開（AGPL、牌譜由来） |
| 画面キャプチャによるリアルタイム推奨・自動操作のツール | オンライン対戦中の支援に使えるため。本プロジェクトの用途は振り返りと研究のみ |
| 雀魂画面の YOLO 認識 | 別プロジェクト。画面の利用条件を確認していないため |

## 再現の流れ / Reproduction outline

前提：WSL2（Ubuntu）、Python 3.12、Rust、CUDA GPU（16 GB）、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。スクリプトは元の配置（リポジトリ = `/mnt/d/mahjong-llm`、作業ディレクトリ = `~/mahjong-llm/work`）を前提にしたパスを含むので、環境に合わせて書き換えてください。

1. 論文コード [THUDM/LLM4CardGame](https://github.com/THUDM/LLM4CardGame) と [Mortal](https://github.com/Equim-chan/Mortal) を取得し、`third_party/libriichi_patch/` の差分（`git apply` 形式）を当てて libriichi をビルド（`bash third_party/libriichi_patch/build.sh`、詳細は同ディレクトリの README）。
2. 天鳳 鳳凰卓の mjai 形式の牌譜を各自で入手し、`tools/run_pipeline_fast.py` で判断局面と SFT データを作る。
3. `tools/build_mortal_labels.py` で Mortal の Q 値ラベルを付与（Mortal の重みが別途必要）。`tools/build_explanations.py` で理由の教師を生成。
4. テキスト LLM：`train_small/*.yaml` を LLaMA-Factory で学習・推論。
5. エンコーダ：`mmj/build_full.py` → `mmj/train_encoder.py`（→ `mmj/train_value.py`）。
6. 理由 LLM：`mmj/policy_facts.py` → `mmj/train_mm.py`。
7. 画像：`vlm/render_dataset.py`・`vlm/render_labels.py` → `mm3/perception.py` → `mm3/e2e_eval.py`。レンダラーは麻雀牌の字形を含むフォント（Windows の Segoe UI Symbol）を使います。
8. 評価：`train_small/eval_vs_mortal.py`、`eval_reason.py`、`mcnemar.py`、`zero_shot_report.py`。

各実行の正確なコマンドは `queue/` にあります。

## 方針 / Policy

- 用途は牌譜の振り返りと研究に限ります。オンライン対戦中のリアルタイム支援には使いません。
- 学習済みの重みと牌譜データは公開しません。

## ライセンス / License

- このリポジトリのコード：Apache License 2.0（`LICENSE`）。
- `third_party/libriichi_patch/`：Mortal（libriichi）への変更なので AGPL-3.0（同ディレクトリの `LICENSE`）。
- 依存：Qwen3.5（Apache-2.0）、LLaMA-Factory（Apache-2.0）、Mortal / libriichi（AGPL-3.0）。LLM4CardGame はライセンス表記がないため、このリポジトリには含めず上流から取得してください。詳しくは `NOTICE`。
