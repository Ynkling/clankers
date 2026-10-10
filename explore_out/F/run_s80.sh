#!/bin/bash
# Session F S80 queue (EXPLORATORY, not a result), detached; kept alive by explore_out/F/keepalive.sh.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s80.log
echo "=== S80 start $(date -u)" >> $LOG
$PY -u explore_f_configs.py >> $LOG 2>&1; echo "exit $?" >> $LOG
git add explore_out/F/ >/dev/null 2>&1
git commit -q -m "explore_out/F: S80 runs (EXPLORATORY, not a result)" >> $LOG 2>&1
git push -q origin claude/explore-F >> $LOG 2>&1
echo "=== S80 done $(date -u)" >> $LOG
