#!/bin/bash
# Session F round 6 (S89-F): phase 0, pilots, [48000 pilots], baselines (EXPLORATORY, not a result), detached.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s89.log
stage() { echo "=== S89 $1 $(date -u)" >> $LOG; $PY -u explore_f_s89_tasks.py >> $LOG 2>&1; echo "exit $?" >> $LOG;
          $PY -u explore_f_s89_tasks.py --decide >> $LOG 2>&1;
          git add explore_out/F/ >/dev/null 2>&1; git commit -q -m "explore_out/F: S89 $1 (EXPLORATORY, not a result)" >> $LOG 2>&1;
          git push -q origin claude/explore-F >> $LOG 2>&1; }
stage phase0
stage phase1
stage phase1b
stage phase2
echo "=== S89 done $(date -u)" >> $LOG
