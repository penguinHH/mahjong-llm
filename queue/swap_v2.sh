#!/usr/bin/env bash
# 先把 v2 和改过的端到端评测放进队列，再停掉 v1 的训练进程（runner 会把 20 记为失败并接着跑 21）
SRC=$(dirname "$0")
bash $SRC/enqueue.sh 21_perception_v2.sh 25_e2e_eval.sh
pid=$(pgrep -f "perception.py --out $HOME/mahjong-llm/work/mm3/perception_tfx")
echo "v1 进程: $pid"
[ -n "$pid" ] && kill $pid
sleep 40
tail -4 $HOME/mahjong-llm/work/queue/runner.log
ls $HOME/mahjong-llm/work/queue/running
