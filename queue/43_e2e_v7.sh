# 端到端评测：图片 → 感知 v7（点数 ROI 头，已含组装 + 规则后处理）→ 编码器 + 理由 LLM
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
E=/mnt/d/mahjong-llm/mm3/e2e_eval.py
T=/mnt/d/mahjong-llm/train_small
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== 感知 v7 $(ts)"
$PY -I $E --state $D/perception_v7/test_pred.npz --out $D/e2e_v7 > $D/e2e_v7.log 2>&1 \
  || { echo "失败"; tail -20 $D/e2e_v7.log; exit 1; }
grep -E '一致|端到端' $D/e2e_v7.log | cut -c1-400
$PY -I $T/eval_vs_mortal.py $D/e2e_true/generated_predictions.jsonl $D/e2e_v7/generated_predictions.jsonl 2>&1 | tail -6
$PY -I $T/eval_reason.py $D/e2e_v7/generated_predictions.jsonl 2>&1 | grep -E '样本|有理由|按句子类型'
echo "=== 完成 $(ts)"
