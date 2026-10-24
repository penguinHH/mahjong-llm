# 次点 / 档位分类头（mmj/gap_head.py）：冻结 tf256b6_v_x 编码器，直接预测理由里"次点の○○と僅差/より良い/より明確に良い"
# 依据：这句是理由里唯一不到 100% 的句型（真实局面 R35 79.5%，感知 v7 端到端 74.8%）；价值头回归再切档整句只对 80.1%
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
D=~/mahjong-llm/work/mmj
ts() { date '+%F %T'; }
echo "=== 次点 / 档位头 $(ts)"
$PY -I /mnt/d/mahjong-llm/mmj/gap_head.py --out $D/gap_head_tf256b6_v_x.pt > $D/gap_head.log 2>&1 \
  || { echo "失败"; tail -30 $D/gap_head.log; exit 1; }
grep -E '训练|价值头对照|^ep|测试集|已保存' $D/gap_head.log | cut -c1-400
echo "=== 完成 $(ts)"
