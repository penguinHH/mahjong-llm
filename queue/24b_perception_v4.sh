# 感知模块 v4：v3c + 逐图块监督（vlm/render_labels.py 的标签）+ 直方图读出平面。先冒烟 20 步，再正式训练
# 依据：64 张过拟合时逐格 loss 3.3→0.03（认牌学得会），但整图 MLP 读平面仍慢；v3c 1250 步平面 BCE 仍 0.156
cd ~
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/mahjong-llm/venv-train/bin/python
P=/mnt/d/mahjong-llm/mm3/perception.py
D=~/mahjong-llm/work/mm3
ts() { date '+%F %T'; }
echo "=== 冒烟 $(ts)"
rm -rf /tmp/perc_smoke4
$PY -I $P --dense --hist --max_steps 20 --eval_n 64 --workers 4 --out /tmp/perc_smoke4 > $D/perception_v4_smoke.log 2>&1 \
  || { echo "冒烟失败"; tail -30 $D/perception_v4_smoke.log; exit 1; }
grep -E '可训练|"step"|训练完成|最终评测' $D/perception_v4_smoke.log | cut -c1-500
echo "=== 正式训练 $(ts)"
$PY -I $P --dense --hist --out $D/perception_v4 --eval_every 500 > $D/perception_v4.log 2>&1 \
  || { echo "训练失败"; tail -30 $D/perception_v4.log; exit 1; }
grep -E '训练完成|最终评测' $D/perception_v4.log | cut -c1-700
echo "=== 完成 $(ts)"
