# 端到端评测（补跑感知那一遍）：图片 → 感知 v5 → 编码器 + 理由 LLM。上次因 last kawa tile 为空在事实生成处报错，已修
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
E=/mnt/d/mahjong-llm/mm3/e2e_eval.py
T=/mnt/d/mahjong-llm/train_small
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== 感知模块预测的局面 $(ts)"
$PY -I $E --state $D/perception_v5/test_pred.npz --out $D/e2e_perception > $D/e2e_perception.log 2>&1 \
  || { echo "失败"; tail -20 $D/e2e_perception.log; exit 1; }
grep -E '一致|端到端' $D/e2e_perception.log | cut -c1-400
echo "=== 与 Mortal 对比 $(ts)"
$PY -I $T/eval_vs_mortal.py $D/e2e_true/generated_predictions.jsonl $D/e2e_perception/generated_predictions.jsonl 2>&1 | tail -8
for x in e2e_true e2e_perception; do
  echo "--- 理由准确率 $x"
  $PY -I $T/eval_reason.py $D/$x/generated_predictions.jsonl 2>&1 | grep -E '样本|有理由|按句子类型'
done
echo "=== 完成 $(ts)"
