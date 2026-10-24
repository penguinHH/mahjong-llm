#!/usr/bin/env bash
# 放入 v3b 和改过的端到端评测，再停掉开着 KL 的 v3
SRC=$(dirname "$0")
bash $SRC/enqueue.sh 23_perception_v3b.sh 25_e2e_eval.sh
pid=$(pgrep -f "perception.py --out $HOME/mahjong-llm/work/mm3/perception_v3 ")
echo "v3 进程: $(echo $pid)"
[ -n "$pid" ] && kill $pid
sleep 30
tail -3 $HOME/mahjong-llm/work/queue/runner.log
ls $HOME/mahjong-llm/work/queue/running
