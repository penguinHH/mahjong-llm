# 端到端评测：評価順改由次点 / 档位头生成（e2e_eval.py --gap），真实局面与感知 v7 各一遍，对照不加 --gap 的 e2e_true / e2e_v7
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
E=/mnt/d/mahjong-llm/mm3/e2e_eval.py
T=/mnt/d/mahjong-llm/train_small
D=~/mahjong-llm/work/mm3
G=~/mahjong-llm/work/mmj/gap_head_tf256b6_v_x.pt
ts() { date '+%F %T'; }
[ -f $G ] || { echo "没有 $G（44 号没跑成）"; exit 1; }
echo "=== 对照：真实局面、原評価順 的理由 $(ts)"
$PY -I $T/eval_reason.py $D/e2e_true/generated_predictions.jsonl 2>&1 | grep -E '样本|有理由|按句子类型'
for s in true v7; do
  st=$([ $s = true ] && echo true || echo $D/perception_v7/test_pred.npz)
  echo "=== $s + 次点头 $(ts)"
  $PY -I $E --state $st --gap $G --out $D/e2e_${s}_gap > $D/e2e_${s}_gap.log 2>&1 \
    || { echo "失败"; tail -20 $D/e2e_${s}_gap.log; exit 1; }
  grep -E '一致|端到端' $D/e2e_${s}_gap.log | cut -c1-300
  $PY -I $T/eval_vs_mortal.py $D/e2e_${s}_gap/generated_predictions.jsonl 2>&1 | tail -3
  $PY -I $T/eval_reason.py $D/e2e_${s}_gap/generated_predictions.jsonl 2>&1 | grep -E '样本|有理由|按句子类型'
done
echo "=== 完成 $(ts)"
