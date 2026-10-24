# 端到端评测：感知 v5 的预测经规则后处理（postprocess.py：last_kawa、吃的种类、向听数、位次点差）后再走一遍
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
E=/mnt/d/mahjong-llm/mm3/e2e_eval.py
T=/mnt/d/mahjong-llm/train_small
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== v5 + 规则后处理 $(ts)"
$PY -I $E --state $D/perception_v5/test_pred_refined.npz --out $D/e2e_v5refined > $D/e2e_v5refined.log 2>&1 \
  || { echo "失败"; tail -20 $D/e2e_v5refined.log; exit 1; }
grep -E '一致|端到端' $D/e2e_v5refined.log | cut -c1-400
$PY -I $T/eval_vs_mortal.py $D/e2e_perception/generated_predictions.jsonl $D/e2e_v5refined/generated_predictions.jsonl 2>&1 | tail -6
$PY -I $T/eval_reason.py $D/e2e_v5refined/generated_predictions.jsonl 2>&1 | grep -E '样本|有理由|按句子类型'
echo "=== 完成 $(ts)"
