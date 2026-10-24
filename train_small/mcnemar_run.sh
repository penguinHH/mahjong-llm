#!/usr/bin/env bash
# McNemar 一键跑法（10-23）：编码器 9 版重推理 + e2e/三合一预测文件，指定的对比对
cd ~/mahjong-llm
source ~/mahjong-llm/venv-train/bin/activate
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
D=~/mahjong-llm/work/mmj
T=~/mahjong-llm/work/mm3
python -I /mnt/d/mahjong-llm/train_small/mcnemar.py \
  R17=enc:$D/encoder_100k_c192b10.pt \
  R18=enc:$D/encoder_full_c192b10.pt \
  R19=enc:$D/encoder_full_c192b10_softT1a0.5.pt \
  R20=enc:$D/encoder_full_soft_e16.pt \
  R21=enc:$D/encoder_full_c192b10_human.pt \
  R23=enc:$D/encoder_cnn192b10_v.pt \
  R24=enc:$D/encoder_cnn256b20_v.pt \
  R25=enc:$D/encoder_tf256b6_v.pt \
  R26=enc:$D/encoder_tf256b6_v_x.pt \
  e2e48=$T/e2e_true/generated_predictions.jsonl \
  e2e50=$T/e2e_perception/generated_predictions.jsonl \
  e2e54=$T/e2e_v5refined/generated_predictions.jsonl \
  e2e56=$T/e2e_v6/generated_predictions.jsonl \
  e2e71=$T/e2e_v7/generated_predictions.jsonl \
  triS=$T/tri_q35_08b_tfx/generated_predictions_state.jsonl \
  triB=$T/tri_q35_08b_tfx/generated_predictions_both.jsonl \
  --pairs R17-R20,R18-R20,R20-R21,R20-R23,R23-R24,R24-R25,R25-R26,R26-e2e48,e2e48-e2e50,e2e50-e2e54,e2e54-e2e56,e2e56-e2e71,e2e48-e2e71,e2e71-triS,R26-triS,triS-triB
