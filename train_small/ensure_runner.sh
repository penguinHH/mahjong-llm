#!/usr/bin/env bash
# GPU 队列看门狗（Windows 计划任务每 5 分钟调）：确保 runner 在跑。统一走 systemd；实在不行才 tmux/nohup 兜底。
# 注意：非登录 wsl.exe 上下文里 XDG_RUNTIME_DIR 常为空，systemctl --user 会连不上总线 → 先补上再查。
Q=$HOME/mahjong-llm/work/queue
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
systemctl --user is-active --quiet gpuq-runner.service 2>/dev/null && exit 0
if systemctl --user start gpuq-runner.service 2>/dev/null; then
  echo "=== $(date '+%F %T') watchdog 通过 systemd 拉起 runner" >> $Q/runner.log
  exit 0
fi
pgrep -f "bash $Q/runner.sh" >/dev/null 2>&1 && exit 0
tmux new-session -d -s gpuq "bash $Q/runner.sh" 2>/dev/null || nohup bash $Q/runner.sh >> $Q/runner-watchdog.log 2>&1 &
echo "=== $(date '+%F %T') watchdog 兜底拉起 runner（systemd 不可用，用 tmux/nohup）" >> $Q/runner.log
