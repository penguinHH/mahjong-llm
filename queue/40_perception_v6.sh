# 感知 v6：从 v5 接着微调 2000 步，加点数分类头（千位 + 百位）；评测时组装规则给出摸牌、再做规则后处理
# 依据：v5 点数回归平均差 400～480 点（误差 <50 点仅约 11%）；向听数需要知道刚摸的牌
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
A="--dense --hist --center --score_cls --init $D/perception_v5/final.pt --lr 1e-4"
ts() { date '+%F %T'; }
echo "=== 冒烟 $(ts)"
rm -rf /tmp/perc_smoke6
$PY -I $P $A --max_steps 10 --eval_n 64 --workers 4 --out /tmp/perc_smoke6 > $D/perception_v6_smoke.log 2>&1 \
  || { echo "冒烟失败"; tail -30 $D/perception_v6_smoke.log; exit 1; }
grep -E '载入|可训练|"step"|最终评测' $D/perception_v6_smoke.log | cut -c1-700
echo "=== 微调 $(ts)"
$PY -I $P $A --max_steps 2000 --eval_every 500 --out $D/perception_v6 > $D/perception_v6.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_v6.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_v6.log | cut -c1-900
echo "=== 完成 $(ts)"
