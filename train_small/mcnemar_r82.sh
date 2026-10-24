#!/usr/bin/env bash
# R82 补充 McNemar：编码器 R26 vs LLM 自决策 R82；R82 vs 三合一 state/both（结果 tee 到 W/mmj/mcnemar_r82.log）
cd ~/mahjong-llm
source ~/mahjong-llm/venv-train/bin/activate
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
D=~/mahjong-llm/work/mmj
T=~/mahjong-llm/work/mm3
python -I /mnt/d/mahjong-llm/train_small/mcnemar.py \
  R26=enc:$D/encoder_tf256b6_v_x.pt \
  R82=$D/mm_q35_08b_reason_tfx_nofacts/generated_predictions.jsonl \
  triS=$T/tri_q35_08b_tfx/generated_predictions_state.jsonl \
  triB=$T/tri_q35_08b_tfx/generated_predictions_both.jsonl \
  --pairs R26-R82,R82-triS,R82-triB 2>&1 | tee ~/mahjong-llm/work/mmj/mcnemar_r82.log
