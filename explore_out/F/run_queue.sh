#!/bin/bash
# Session F's run queue (EXPLORATORY, not a result), detached, no segment budget: S54, then S59's main arms (oracle
# first). Kept alive by a harness-tracked keepalive (explore_out/F/keepalive.sh), which also pushes explore_out/F.
cd /home/user/clankers
PY=/usr/bin/python3.11
echo "=== queue: S54 (re)start $(date -u)" >> explore_out/F/s54.log
$PY -u explore_f_window_delta.py >> explore_out/F/s54.log 2>&1; echo "exit $?" >> explore_out/F/s54.log
echo "=== queue: S59 oracle arms $(date -u)" >> explore_out/F/s59.log
$PY -u explore_f_decay.py --only ORACLE_FIXED,ORACLE_ROUTED,ORACLE_GDN >> explore_out/F/s59.log 2>&1; echo "exit $?" >> explore_out/F/s59.log
echo "=== queue: S59 single arms $(date -u)" >> explore_out/F/s59.log
$PY -u explore_f_decay.py --only SINGLE_FIXED,SINGLE_ROUTED,SINGLE_GDN >> explore_out/F/s59.log 2>&1; echo "exit $?" >> explore_out/F/s59.log
echo "=== queue: done $(date -u)" >> explore_out/F/s59.log
