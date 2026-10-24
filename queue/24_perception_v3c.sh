# 感知模块 v3c：v3 结构 + 关掉 KL + 读出头输入按通道标准化（BN；视觉塔有两个通道均值约 650，按 token 的 LN 会压掉其余通道）
# 过拟合对照（64 张、300 步）：LN 0.135 → BN 0.107
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== 正式训练 $(ts)"
$PY -I $P --out $D/perception_v3c --kl_w 0 --in_norm bn --eval_every 500 > $D/perception_v3c.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_v3c.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_v3c.log | cut -c1-600
echo "=== 完成 $(ts)"
