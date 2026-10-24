#!/usr/bin/env bash
# 通用训练启动 / 暂停 / 查看（tmux 会话里跑；配置需 save_only_model: false + overwrite_output_dir: false 才能续训）
#
#   wsl -d Ubuntu -- bash /mnt/d/mahjong-llm/train_small/run_train.sh qwen35_0_8b_distill.yaml start
#   wsl -d Ubuntu -- bash /mnt/d/mahjong-llm/train_small/run_train.sh qwen35_0_8b_distill.yaml stop
#   wsl -d Ubuntu -- bash /mnt/d/mahjong-llm/train_small/run_train.sh qwen35_0_8b_distill.yaml status
#   wsl -d Ubuntu -- tmux attach -t <会话名>      # 会话名 = 配置文件名去掉 .yaml、点换成下划线
set -u
NAME=$(basename "${1:?用法: run_train.sh <配置.yaml> start|stop|status}" .yaml)
CFG=/mnt/d/mahjong-llm/train_small/$NAME.yaml
[ -f "$1" ] && CFG=$(realpath "$1")             # 也接受其他目录下配置的完整路径（如 vlm/）
OUT=$(grep -E '^output_dir:' "$CFG" | awk '{print $2}')
LOG=$HOME/mahjong-llm/work/train_$NAME.log
S=${NAME//./_}
PROC="bin/llamafactory-cli train $CFG"          # 只匹配训练进程本身，不匹配 tmux 会话的命令行
# OMP_NUM_THREADS=1：数据预处理开 8 个进程，每个默认再开 8 个线程，8 核 CPU 上严重超额（VLM 预处理 10 万张图要 1.5 小时）

progress() { tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it' | tail -1; }
last_ckpt() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1; }

case "${2:-status}" in
start)
    if tmux has-session -t "$S" 2>/dev/null; then echo "已经在跑（tmux 会话 $S）"; exit 0; fi
    echo "最新 checkpoint: $(last_ckpt || true)"
    tmux new-session -d -s "$S" "bash -lc 'source ~/mahjong-llm/venv-train/bin/activate; cd ~/mahjong-llm; \
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1; \
echo \"=== start \$(date +%F\ %T)\" >> $LOG; \
llamafactory-cli train $CFG 2>&1 | tee -a $LOG; echo \"=== exit \$(date +%F\ %T)\" >> $LOG; exec bash'"
    echo "已启动：tmux 会话 $S，日志 $LOG"
    ;;
stop)
    if ! tmux has-session -t "$S" 2>/dev/null; then echo "没有在跑"; exit 0; fi
    echo "当前进度: $(progress)"
    echo "最新 checkpoint: $(last_ckpt) —— 下次 start 从这里继续"
    tmux send-keys -t "$S" C-c
    for _ in $(seq 30); do pgrep -f "$PROC" >/dev/null || break; sleep 2; done
    pkill -f "$PROC" 2>/dev/null
    tmux kill-session -t "$S"
    echo "已暂停，显存已释放"
    ;;
status)
    if pgrep -f "$PROC" >/dev/null; then echo "状态: 运行中"; else echo "状态: 未运行"; fi
    echo "进度: $(progress)"
    echo "最新 checkpoint: $(last_ckpt)"
    [ -f "$OUT/all_results.json" ] && echo "已完成: $(tr -d '\n ' < "$OUT/all_results.json")"
    nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
    ;;
*)
    echo "用法: $0 <配置.yaml> start|stop|status"; exit 1 ;;
esac
