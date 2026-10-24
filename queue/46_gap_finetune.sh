# 第 2 步：解冻编码器一起微调次点 / 档位头（gap_head.py --finetune），策略头用 KL 贴住原编码器，尽量不改打牌决策
# 依据：冻结编码器时新头 = 价值头（测试整句 79.98% vs 79.96%），瓶颈在特征；误差拆解显示约 4 成错误不在阈值附近
# 判断标准：验证整句对 ≥ 82%，首选不变 ≥ 99%
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
G=/mnt/d/mahjong-llm/mmj/gap_head.py
D=~/mahjong-llm/work/mmj
ts() { date '+%F %T'; }
echo "=== 冒烟 $(ts)"
$PY -I $G --finetune --max_train 20000 --epochs 1 --out /tmp/gap_ft_smoke.pt > $D/gap_finetune_smoke.log 2>&1 \
  || { echo "冒烟失败"; tail -30 $D/gap_finetune_smoke.log; exit 1; }
grep -E '^ep|测试集' $D/gap_finetune_smoke.log | cut -c1-400
echo "=== 微调 $(ts)"
$PY -I $G --finetune --epochs 4 --out $D/gap_head_ft_tf256b6_v_x.pt > $D/gap_finetune.log 2>&1 \
  || { echo "失败"; tail -30 $D/gap_finetune.log; exit 1; }
grep -E '训练|价值头对照|^ep|测试集|已保存' $D/gap_finetune.log | cut -c1-500
echo "=== 完成 $(ts)"
