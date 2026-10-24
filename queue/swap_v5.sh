#!/usr/bin/env bash
# 放入 v5 和改过的端到端评测，再停掉 v4
SRC=$(dirname "$0")
bash $SRC/enqueue.sh 24c_perception_v5.sh 25_e2e_eval.sh
pid=$(pgrep -f "perception.py --dense --hist --out $HOME/mahjong-llm/work/mm3/perception_v4")
echo "v4 进程: $(echo $pid)"
[ -n "$pid" ] && kill $pid
sleep 30
tail -3 $HOME/mahjong-llm/work/queue/runner.log
ls $HOME/mahjong-llm/work/queue/running
