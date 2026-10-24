#!/usr/bin/env bash
# 放入 v3 和改过的端到端评测，再停掉 v2（runner 记 21 失败后接着跑 22）
SRC=$(dirname "$0")
bash $SRC/enqueue.sh 22_perception_v3.sh 25_e2e_eval.sh
pid=$(pgrep -f "perception.py --out $HOME/mahjong-llm/work/mm3/perception_v2")
echo "v2 进程: $(echo $pid)"
[ -n "$pid" ] && kill $pid
sleep 30
tail -3 $HOME/mahjong-llm/work/queue/runner.log
ls $HOME/mahjong-llm/work/queue/running
