#!/bin/bash
# Session F's run queue (EXPLORATORY, not a result): S53 to completion, then S54, then S59's main arms (oracle first).
cd /home/user/clankers
PY=/usr/bin/python3.11
while pgrep -f "explore_f_delta_controls.py --budget-min" >/dev/null; do sleep 30; done
echo "=== queue: S53 resume $(date)" >> explore_out/F/s53.log
$PY -u explore_f_delta_controls.py >> explore_out/F/s53.log 2>&1; echo "exit $?" >> explore_out/F/s53.log
echo "=== queue: S54 start $(date)" >> explore_out/F/s54.log
$PY -u explore_f_window_delta.py >> explore_out/F/s54.log 2>&1; echo "exit $?" >> explore_out/F/s54.log
echo "=== queue: S59 oracle arms $(date)" >> explore_out/F/s59.log
$PY -u explore_f_decay.py --only ORACLE_FIXED,ORACLE_ROUTED,ORACLE_GDN >> explore_out/F/s59.log 2>&1; echo "exit $?" >> explore_out/F/s59.log
echo "=== queue: S59 single arms $(date)" >> explore_out/F/s59.log
$PY -u explore_f_decay.py --only SINGLE_FIXED,SINGLE_ROUTED,SINGLE_GDN >> explore_out/F/s59.log 2>&1; echo "exit $?" >> explore_out/F/s59.log
echo "=== queue: done $(date)" >> explore_out/F/s59.log
