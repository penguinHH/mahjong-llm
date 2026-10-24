# VLM 基座零样本对照（关闭思考）：49 号（R83）用 qwen3_5_nothink 模板，基座先写 <think>，格式 0%，结果无效
# 依据：48 号对文字基座做同样的重测；两者合起来回答"未后训练的 0.8B 能不能直接打"
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
T=/mnt/d/mahjong-llm/train_small
P=~/mahjong-llm/work/pred
ts() { date '+%F %T'; }
echo "=== VLM 基座零样本预测（关思考）$(ts)"
$PY -m llamafactory.cli train /mnt/d/mahjong-llm/vlm/qwen35_0_8b_vlm_base_nothink_predict.yaml > ~/mahjong-llm/work/predict_qwen35_vlm_base_nothink.log 2>&1 \
  || { echo "失败"; tail -30 ~/mahjong-llm/work/predict_qwen35_vlm_base_nothink.log; exit 1; }
echo "=== 汇总 $(ts)"
$PY -I $T/zero_shot_report.py "0.8B VLM基座零样本(关思考)=$P/qwen35-0.8b-vlm-base-nothink/generated_predictions.jsonl" \
  "0.8B VLM后训练(人标签)=$P/qwen35-0.8b-vlm-human/generated_predictions.jsonl"
echo "=== 完成 $(ts)"
