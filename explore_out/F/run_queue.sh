#!/bin/bash
# Session F's run queue (EXPLORATORY, not a result). Restarted after a container restart at 07:49 UTC (S53 complete):
# S54, then S59's main arms (oracle first). A side loop commits and pushes explore_out/F every 20 minutes.
cd /home/user/clankers
PY=/usr/bin/python3.11
( while true; do sleep 1200; git add explore_out/F/ >/dev/null 2>&1 && git commit -q -m "explore_out/F: progress (EXPLORATORY, not a result)" >/dev/null 2>&1 && git push -q origin claude/explore-F >/dev/null 2>&1; done ) &
PUSHER=$!
echo "=== queue: S54 (re)start $(date)" >> explore_out/F/s54.log
$PY -u explore_f_window_delta.py >> explore_out/F/s54.log 2>&1; echo "exit $?" >> explore_out/F/s54.log
echo "=== queue: S59 oracle arms $(date)" >> explore_out/F/s59.log
$PY -u explore_f_decay.py --only ORACLE_FIXED,ORACLE_ROUTED,ORACLE_GDN >> explore_out/F/s59.log 2>&1; echo "exit $?" >> explore_out/F/s59.log
echo "=== queue: S59 single arms $(date)" >> explore_out/F/s59.log
$PY -u explore_f_decay.py --only SINGLE_FIXED,SINGLE_ROUTED,SINGLE_GDN >> explore_out/F/s59.log 2>&1; echo "exit $?" >> explore_out/F/s59.log
echo "=== queue: done $(date)" >> explore_out/F/s59.log
kill $PUSHER
