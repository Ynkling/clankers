#!/bin/bash
# Session F: the type-level perfect gate at k = 16 (explore_f_s90_k16; EXPLORATORY, not a result), detached.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s90k16.log
echo "=== S90 k16 start $(date -u)" >> $LOG
$PY -u explore_f_s90_k16.py >> $LOG 2>&1; echo "exit $?" >> $LOG
git add explore_out/F/ >/dev/null 2>&1
git commit -q -m "explore_out/F: S90 k = 16 type-level pilot (EXPLORATORY, not a result)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016a79XyXzJKpgHo6vpuK6B2" >> $LOG 2>&1
git push -q origin claude/explore-F >> $LOG 2>&1
echo "=== S90 k16 done $(date -u)" >> $LOG
