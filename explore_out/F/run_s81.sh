#!/bin/bash
# Session F S81 queue (EXPLORATORY, not a result), detached; kept alive by explore_out/F/keepalive.sh. Waits for S80.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s81.log
while pgrep -f "^/bin/bash explore_out/F/run_s80" >/dev/null; do sleep 60; done
echo "=== S81 start $(date -u)" >> $LOG
$PY -u explore_f_trigger.py >> $LOG 2>&1; echo "exit $?" >> $LOG
git add explore_out/F/ >/dev/null 2>&1
git commit -q -m "explore_out/F: S81 runs (EXPLORATORY, not a result)" >> $LOG 2>&1
git push -q origin claude/explore-F >> $LOG 2>&1
echo "=== S81 done $(date -u)" >> $LOG
