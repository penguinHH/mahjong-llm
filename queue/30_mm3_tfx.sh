# 三合一（mm3）正式训练：换成最强的编码器 tf256b6_v_x + 对应的对齐投影层，三种输入模式混合，视觉塔挂 LoRA、merger 全训
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
M=~/mahjong-llm/work/mmj
D=~/mahjong-llm/work/mm3
T=/mnt/d/mahjong-llm/train_small
mkdir -p $D
ts() { date '+%F %T'; }
echo "=== 冒烟（省显存的 forward + 梯度检查点）$(ts)"
$PY -I /mnt/d/mahjong-llm/mm3/train.py --enc $M/encoder_tf256b6_v_x.pt \
  --init_proj $M/mm_q35_08b_align_tf256b6_v_x/final.pt --lora_visual --train_merger \
  --bs 2 --ga 8 --max_steps 3 --eval_n 4 --out /tmp/mm3_smoke2 > $D/tri_smoke2.log 2>&1 \
  || { echo "冒烟失败"; tail -20 $D/tri_smoke2.log; exit 1; }
grep -E '"step"|训练完成|最终评测' $D/tri_smoke2.log | cut -c1-300
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== 三合一训练 $(ts)"
$PY -I /mnt/d/mahjong-llm/mm3/train.py --enc $M/encoder_tf256b6_v_x.pt \
  --init_proj $M/mm_q35_08b_align_tf256b6_v_x/final.pt --lora_visual --train_merger \
  --bs 2 --ga 8 --eval_n 5000 --out $D/tri_q35_08b_tfx > $D/tri_q35_08b_tfx.log 2>&1 \
  || { echo "三合一训练失败"; tail -30 $D/tri_q35_08b_tfx.log; exit 1; }
grep -E '训练完成|最终评测' $D/tri_q35_08b_tfx.log | cut -c1-600
for m in state image both; do
  echo "--- 模式 $m"
  $PY -I $T/eval_vs_mortal.py $D/tri_q35_08b_tfx/generated_predictions_$m.jsonl 2>&1 | tail -6
done
echo "=== 完成 $(ts)"
