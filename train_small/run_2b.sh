#!/usr/bin/env bash
# Qwen3.5-2B 训练的启动 / 暂停 / 查看（在 tmux 会话 train 里跑，可以随时停、随时续）
#
#   wsl -d Ubuntu -- bash /mnt/d/mahjong-llm/train_small/run_2b.sh start    # 开始；输出目录里有 checkpoint 就自动续训
#   wsl -d Ubuntu -- bash /mnt/d/mahjong-llm/train_small/run_2b.sh stop     # 暂停（丢掉最近一个 checkpoint 之后的进度）
#   wsl -d Ubuntu -- bash /mnt/d/mahjong-llm/train_small/run_2b.sh status   # 进度、最新 checkpoint、显存
#   wsl -d Ubuntu -- tmux attach -t train                                   # 看实时画面（Ctrl-b d 离开）
set -u
CFG=/mnt/d/mahjong-llm/train_small/qwen35_2b_lora.yaml
OUT=$HOME/mahjong-llm/work/ckpt/qwen35-2b-riichi_100k
LOG=$HOME/mahjong-llm/work/train_qwen35_2b.log
S=train

progress() { tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it' | tail -1; }
last_ckpt() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1; }

case "${1:-status}" in
start)
    if tmux has-session -t $S 2>/dev/null; then echo "已经在跑（tmux 会话 $S）"; exit 0; fi
    echo "最新 checkpoint: $(last_ckpt || true)"
    # PYTHONUNBUFFERED：loss 记录实时写进日志；expandable_segments：验证后显存不碎片化
    tmux new-session -d -s $S "bash -lc 'source ~/mahjong-llm/venv-train/bin/activate; cd ~/mahjong-llm; \
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
echo \"=== start \$(date +%F\ %T)\" >> $LOG; \
llamafactory-cli train $CFG 2>&1 | tee -a $LOG; echo \"=== exit \$(date +%F\ %T)\" >> $LOG; exec bash'"
    echo "已启动：tmux 会话 $S，日志 $LOG"
    ;;
stop)
    if ! tmux has-session -t $S 2>/dev/null; then echo "没有在跑"; exit 0; fi
    echo "当前进度: $(progress)"
    echo "最新 checkpoint: $(last_ckpt) —— 下次 start 从这里继续"
    tmux send-keys -t $S C-c
    for _ in $(seq 30); do pgrep -f "llamafactory-cli train $CFG" >/dev/null || break; sleep 2; done
    pkill -f "llamafactory-cli train $CFG" 2>/dev/null
    tmux kill-session -t $S
    echo "已暂停，显存已释放"
    ;;
status)
    tmux has-session -t $S 2>/dev/null && echo "状态: 运行中" || echo "状态: 未运行"
    echo "进度: $(progress)"
    echo "最新 checkpoint: $(last_ckpt)"
    nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
    ;;
*)
    echo "用法: $0 start|stop|status"; exit 1 ;;
esac
