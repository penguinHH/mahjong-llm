# VLM 基座零样本对照（R83）：补"未后训练 VLM 能不能看图打麻将"的一手证据
# 依据：R37/R38（后训练 VLM 看图决策 65.7/68.6）之外没有基座数据；文字侧已有 R76/R77（格式 0%）
# 与 R37 同批 5000 张测试图、同模板、贪心、128 token、不挂 LoRA；汇总用 zero_shot_report.py + eval_vs_mortal.py
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
T=/mnt/d/mahjong-llm/train_small
P=~/mahjong-llm/work/pred
ts() { date '+%F %T'; }
echo "=== VLM 基座零样本预测 $(ts)"
$PY -m llamafactory.cli train /mnt/d/mahjong-llm/vlm/qwen35_0_8b_vlm_base_predict.yaml > ~/mahjong-llm/work/predict_qwen35_vlm_base.log 2>&1 \
  || { echo "失败"; tail -30 ~/mahjong-llm/work/predict_qwen35_vlm_base.log; exit 1; }
echo "=== 汇总 $(ts)"
$PY -I $T/zero_shot_report.py "0.8B VLM基座零样本=$P/qwen35-0.8b-vlm-base/generated_predictions.jsonl" \
  "0.8B VLM后训练(人标签)=$P/qwen35-0.8b-vlm-human/generated_predictions.jsonl"
$PY -I $T/eval_vs_mortal.py $P/qwen35-0.8b-vlm-base/generated_predictions.jsonl
echo "=== 完成 $(ts)"
