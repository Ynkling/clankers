#!/bin/bash
# Session F S89-F post-hoc diagnostic (EXPLORATORY, not a result), detached.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s89diag.log
echo "=== S89 diag start $(date -u)" >> $LOG
$PY -u explore_f_s89_diag.py >> $LOG 2>&1; echo "exit $?" >> $LOG
git add explore_out/F/ >/dev/null 2>&1; git commit -q -m "explore_out/F: S89 diagnostic runs (EXPLORATORY, not a result)" >> $LOG 2>&1
git push -q origin claude/explore-F >> $LOG 2>&1
echo "=== S89 diag done $(date -u)" >> $LOG
