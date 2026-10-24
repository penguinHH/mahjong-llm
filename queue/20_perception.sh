# 感知模块：图片 → 局面张量（先冒烟 20 步，通过后正式训练 1 轮并评测 5000 张测试图）
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
mkdir -p $D
ts() { date '+%F %T'; }
echo "=== 冒烟 $(ts)"
rm -rf /tmp/perc_smoke
$PY -I $P --max_steps 20 --eval_n 64 --workers 4 --out /tmp/perc_smoke > $D/perception_smoke.log 2>&1 \
  || { echo "冒烟失败"; tail -30 $D/perception_smoke.log; exit 1; }
grep -E '可训练|超过 1|"step"|训练完成|最终评测' $D/perception_smoke.log | cut -c1-400
echo "=== 正式训练 $(ts)"
$PY -I $P --out $D/perception_tfx > $D/perception_tfx.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_tfx.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_tfx.log | cut -c1-600
echo "=== 完成 $(ts)"
