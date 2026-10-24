# 感知模块 v5：v4（逐格监督 + 直方图）+ 牌中心头；评测时与牌有关的 41 个平面由 assemble.py 按规则组装
# 依据：v4 测试集逐格认牌 99.96%，但网络读平面仍慢；规则用真实牌位置组装，41 个平面全对 97.3%
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== 冒烟 $(ts)"
rm -rf /tmp/perc_smoke5
$PY -I $P --dense --hist --center --max_steps 20 --eval_n 64 --workers 4 --out /tmp/perc_smoke5 > $D/perception_v5_smoke.log 2>&1 \
  || { echo "冒烟失败"; tail -30 $D/perception_v5_smoke.log; exit 1; }
grep -E '可训练|"step"|训练完成|最终评测' $D/perception_v5_smoke.log | cut -c1-600
echo "=== 正式训练 $(ts)"
$PY -I $P --dense --hist --center --out $D/perception_v5 --eval_every 500 > $D/perception_v5.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_v5.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_v5.log | cut -c1-800
echo "=== 完成 $(ts)"
