# 基座零样本对照：Qwen3.5-0.8B / 2B 不挂 LoRA，在同一 5000 条上直接答，衡量"后训练带来了多少"
# 依据：项目里一直没有未训练基座的数字，"小模型经后训练能完成特定任务"这句话缺对照
cd ~/mahjong-llm
source ~/mahjong-llm/venv-train/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=/mnt/d/mahjong-llm/train_small
P=~/mahjong-llm/work/pred
ts() { date '+%F %T'; }
for m in 0_8b 2b; do
  echo "=== 基座 $m $(ts)"
  llamafactory-cli train $T/qwen35_${m}_base_predict.yaml > ~/mahjong-llm/work/predict_qwen35_${m}_base.log 2>&1 \
    || { echo "失败"; tail -30 ~/mahjong-llm/work/predict_qwen35_${m}_base.log; exit 1; }
done
echo "=== 汇总 $(ts)"
python -I $T/zero_shot_report.py "0.8B基座零样本=$P/qwen35-0.8b-base/generated_predictions.jsonl" \
  "2B基座零样本=$P/qwen35-2b-base/generated_predictions.jsonl" \
  "0.8B后训练=$P/qwen35-0.8b-mortal/generated_predictions.jsonl" \
  "2B后训练=$P/qwen35-2b-mortal/generated_predictions.jsonl"
echo "=== 完成 $(ts)"
