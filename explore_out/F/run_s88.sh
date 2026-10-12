#!/bin/bash
# Session F S88a then S88b (EXPLORATORY, not a result), detached; kept alive by explore_out/F/keepalive.sh.
cd /home/user/clankers
PY=/usr/bin/python3.11
for s in a b; do
  LOG=explore_out/F/s88$s.log
  [ $s = a ] && SCR=explore_f_write.py || SCR=explore_f_horizon.py
  echo "=== S88$s start $(date -u)" >> $LOG
  $PY -u $SCR >> $LOG 2>&1; echo "exit $?" >> $LOG
  git add explore_out/F/ >/dev/null 2>&1
  git commit -q -m "explore_out/F: S88$s runs (EXPLORATORY, not a result)" >> $LOG 2>&1
  git push -q origin claude/explore-F >> $LOG 2>&1
  echo "=== S88$s done $(date -u)" >> $LOG
done
