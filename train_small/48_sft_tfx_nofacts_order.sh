# 单变量对照（report §6.2-1）：R35 去掉評価順重训 —— 编码器 tf256b6_v_x + 投影层不变，facts 用无評価順版
# 依据：R80 已证 R35 的动作 = 照抄評価順第 1 名（=R26），"LLM 读 R26 token 自己决策"一直没有干净数字
#   （三合一 state 94.42% 混了图片任务；R31 是 R20 编码器）。本任务 = R35 只改 facts_val → facts 一个变量
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
M=~/mahjong-llm/work/mmj
E=/mnt/d/mahjong-llm/mmj
T=/mnt/d/mahjong-llm/train_small
ts() { date '+%F %T'; }
echo "=== LLM 微调（事实，无評価順，R26 编码器）$(ts)"
$PY -I $E/train_mm.py --stage sft --target reason --facts --facts_name facts --micro_bs 8 --enc $M/encoder_tf256b6_v_x.pt \
  --init $M/mm_q35_08b_align_tf256b6_v_x/final.pt --out $M/mm_q35_08b_reason_tfx_nofacts --eval_n 5000 \
  > $M/mm_sft_tfx_nofacts.log 2>&1 || { echo "微调失败"; tail -5 $M/mm_sft_tfx_nofacts.log; exit 1; }
grep -E '训练完成|最终评测' $M/mm_sft_tfx_nofacts.log | cut -c1-300
P=$M/mm_q35_08b_reason_tfx_nofacts/generated_predictions.jsonl
$PY -I $T/eval_vs_mortal.py $P
$PY -I $T/eval_reason.py $P
$PY -I $T/analyze_discards.py $P | grep -E '牌效最优|现物|一致率：'
echo "=== 全部完成 $(ts)"
