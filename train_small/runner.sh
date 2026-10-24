#!/usr/bin/env bash
# GPU 任务队列 v2：按文件名顺序执行 pending/ 里的 .sh，跑完移到 done/（失败移到 failed/）。
# 由 systemd 用户服务 gpuq-runner.service 常驻；Windows 计划任务每 5 分钟调 ensure_runner.sh 兜底。
# 本文件是唯一来源，安装：sed 's/\r$//' 本文件 > W/queue/runner.sh.new && mv 覆盖 W/queue/runner.sh。
# v2（10-24 重建）：10-23 晚的修复被 start_queue.sh 用原版覆盖，按 03_runs.md 的记录补回，另加单实例锁：
#   - flock 单实例：第二个 runner 阻塞等待，前一个退出后才接手（两个 runner 并存时同一任务会被各跑一次）
#   - 启动时把 running/ 里的残留任务放回 pending/（上次运行被 WSL 重启等打断）
#   - nvidia-smi 读数为空或不是数字时按"占用"处理，不开任务；systemd 上下文补 PATH（/usr/lib/wsl/lib）
#   - .out 追加写，同名任务的多次尝试都留在同一个文件里
#   - mv 失败（任务已被拿走）就跳过，不再照样执行
export PATH=/usr/lib/wsl/lib:$PATH
Q=$HOME/mahjong-llm/work/queue
mkdir -p $Q/pending $Q/running $Q/done $Q/failed
log() { echo "=== $(date '+%F %T') $*" >> $Q/runner.log; }
exec 9> $Q/.runner.lock
if ! flock -n 9; then
  log "已有 runner 持锁，本实例（pid $$）等待"
  flock 9
fi
log "runner 启动（v2，pid $$）"
for f in $Q/running/*.sh; do
  [ -e "$f" ] || continue
  mv "$f" $Q/pending/ && log "恢复：$(basename "$f" .sh) 放回 pending（上次运行被打断）"
done
while true; do
  job=$(ls $Q/pending/*.sh 2>/dev/null | LC_ALL=C sort | head -1)
  if [ -z "$job" ]; then sleep 60; continue; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if ! [[ "$used" =~ ^[0-9]+$ ]] || [ "$used" -gt 5000 ]; then
    log "GPU 显存读数异常或已用 ${used}MiB，等别的任务结束"; sleep 120; continue
  fi
  name=$(basename "$job" .sh)
  mv "$job" $Q/running/ || continue
  log "开始 $name"
  if bash $Q/running/$name.sh >> $Q/$name.out 2>&1; then
    mv $Q/running/$name.sh $Q/done/; log "完成 $name"
  else
    mv $Q/running/$name.sh $Q/failed/; log "失败 $name（看 $name.out）"
  fi
done
