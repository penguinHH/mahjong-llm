# 感知 v7：从 v6 接着训 3000 步，点数改用 ROI 头（--score_roi）：直接取点数所在图块的视觉特征读千位 + 百位
# 依据：v6 的点数头从整图降采样后的特征读，四家全对只有 22.8%，21.6% 的读数错 1000 点以上（数字太小，降到 24×32 就丢了）
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
A="--dense --hist --center --score_roi --score_w 0.3 --init $D/perception_v6/final.pt --lr 1e-4"
ts() { date '+%F %T'; }
echo "=== 冒烟 $(ts)"
rm -rf /tmp/perc_smoke7
$PY -I $P $A --max_steps 10 --eval_n 64 --workers 4 --out /tmp/perc_smoke7 > $D/perception_v7_smoke.log 2>&1 \
  || { echo "冒烟失败"; tail -30 $D/perception_v7_smoke.log; exit 1; }
grep -E '载入|可训练|"step"|最终评测' $D/perception_v7_smoke.log | cut -c1-700
echo "=== 训练 $(ts)"
$PY -I $P $A --max_steps 3000 --eval_every 500 --out $D/perception_v7 > $D/perception_v7.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_v7.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_v7.log | cut -c1-900
echo "=== 完成 $(ts)"
