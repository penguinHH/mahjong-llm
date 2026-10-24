# 感知模块 v2：按位置汇聚 + MLP（v1 的查询 token 解码器 1000 步没学会读牌，已停）。先冒烟 20 步，再正式训练
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== 冒烟 $(ts)"
rm -rf /tmp/perc_smoke2
$PY -I $P --max_steps 20 --eval_n 64 --workers 4 --out /tmp/perc_smoke2 > $D/perception_v2_smoke.log 2>&1 \
  || { echo "冒烟失败"; tail -30 $D/perception_v2_smoke.log; exit 1; }
grep -E '可训练|"step"|训练完成|最终评测' $D/perception_v2_smoke.log | cut -c1-400
echo "=== 正式训练 $(ts)"
$PY -I $P --out $D/perception_v2 --eval_every 500 > $D/perception_v2.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_v2.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_v2.log | cut -c1-600
echo "=== 完成 $(ts)"
