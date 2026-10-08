#!/bin/bash
# Session F: one harness-tracked segment (EXPLORATORY, not a result). Usage: run_segment.sh <driver.py> <log> [extra args]
# Runs the driver with a 110-min budget (only runs that fit start), pushing explore_out/F every 20 min and at the end.
cd /home/user/clankers
DRIVER=$1; LOG=$2; shift 2
push() { git add explore_out/F/ >/dev/null 2>&1; git commit -q -m "explore_out/F: progress (EXPLORATORY, not a result)" >/dev/null 2>&1; git push -q origin claude/explore-F >/dev/null 2>&1; }
( while true; do sleep 1200; push; done ) &
P=$!
echo "=== segment $(date -u)" >> $LOG
/usr/bin/python3.11 -u $DRIVER --budget-min 110 "$@" >> $LOG 2>&1; echo "exit $?" >> $LOG
kill $P; push
