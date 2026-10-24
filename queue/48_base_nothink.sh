# 基座零样本对照（关闭思考）：template qwen3_5 + enable_thinking false；47 号用 qwen3_5_nothink 时基座全部先写 <think>，128 token 截断，格式 0%
# 依据：项目里一直没有未训练基座的数字，"小模型经后训练能完成特定任务"这句话缺对照
cd ~/mahjong-llm
source ~/mahjong-llm/venv-train/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=/mnt/d/mahjong-llm/train_small
P=~/mahjong-llm/work/pred
ts() { date '+%F %T'; }
for m in 0_8b 2b; do
  echo "=== 基座 $m $(ts)"
  llamafactory-cli train $T/qwen35_${m}_base_nothink_predict.yaml > ~/mahjong-llm/work/predict_qwen35_${m}_base_nothink.log 2>&1 \
    || { echo "失败"; tail -30 ~/mahjong-llm/work/predict_qwen35_${m}_base_nothink.log; exit 1; }
done
echo "=== 汇总 $(ts)"
python -I $T/zero_shot_report.py "0.8B基座零样本(关思考)=$P/qwen35-0.8b-base-nothink/generated_predictions.jsonl" \
  "2B基座零样本(关思考)=$P/qwen35-2b-base-nothink/generated_predictions.jsonl" \
  "0.8B后训练=$P/qwen35-0.8b-mortal/generated_predictions.jsonl" \
  "2B后训练=$P/qwen35-2b-mortal/generated_predictions.jsonl"
echo "=== 完成 $(ts)"
