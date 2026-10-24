# 0.8B 基座零样本（关思考、生成 512）：48 号 64 token 下 0.8B 基座全部先写英文分析、格式 0%；放宽长度，排除"只是被截断"
# 依据：2B 基座在 64 token 内就能给 JSON（格式 100%），0.8B 是否只是啰嗦需要单独确认
cd ~/mahjong-llm
source ~/mahjong-llm/venv-train/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=/mnt/d/mahjong-llm/train_small
P=~/mahjong-llm/work/pred
ts() { date '+%F %T'; }
echo "=== 0.8B 基座（关思考，512）$(ts)"
llamafactory-cli train $T/qwen35_0_8b_base_nothink512_predict.yaml > ~/mahjong-llm/work/predict_qwen35_0_8b_base_nothink512.log 2>&1 \
  || { echo "失败"; tail -30 ~/mahjong-llm/work/predict_qwen35_0_8b_base_nothink512.log; exit 1; }
echo "=== 汇总 $(ts)"
python -I $T/zero_shot_report.py "0.8B基座(关思考,512)=$P/qwen35-0.8b-base-nothink-512/generated_predictions.jsonl" \
  "0.8B基座(关思考,64)=$P/qwen35-0.8b-base-nothink/generated_predictions.jsonl" \
  "0.8B后训练=$P/qwen35-0.8b-mortal/generated_predictions.jsonl"
echo "=== 完成 $(ts)"
