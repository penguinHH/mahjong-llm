# 感知模块 v3b：v3 结构（网格 + 卷积 + MLP），关掉经编码器的 KL（过拟合测试证明 KL 让平面学不动）
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== 正式训练 $(ts)"
$PY -I $P --out $D/perception_v3b --kl_w 0 --eval_every 500 > $D/perception_v3b.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_v3b.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_v3b.log | cut -c1-600
echo "=== 完成 $(ts)"
