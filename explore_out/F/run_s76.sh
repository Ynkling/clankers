#!/bin/bash
# Session F S76 queue (EXPLORATORY, not a result), detached; kept alive by explore_out/F/keepalive.sh. Waits for S77.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s76.log
while pgrep -f "^/bin/bash explore_out/F/run_s77" >/dev/null; do sleep 60; done
echo "=== S76 phase 1 $(date -u)" >> $LOG
$PY -u explore_f_eight.py >> $LOG 2>&1; echo "exit $?" >> $LOG
$PY -u explore_f_eight.py --choose-star >> $LOG 2>&1
if [ -f explore_out/F/eight_delta_star.json ]; then
  git add explore_out/F/ >/dev/null 2>&1
  git commit -q -m "Session F S76: DELTA* chosen by the fixed rule from Phase 1, committed before Phase 2 (EXPLORATORY, not a result)" >> $LOG 2>&1
  git push -q origin claude/explore-F >> $LOG 2>&1
  echo "=== S76 phase 2 $(date -u)" >> $LOG
  $PY -u explore_f_eight.py >> $LOG 2>&1; echo "exit $?" >> $LOG
fi
echo "=== S76 done $(date -u)" >> $LOG
