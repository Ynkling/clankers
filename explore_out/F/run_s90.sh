#!/bin/bash
# Session F round 7 (S90-F): phase 0, pilots, [48000 pilots], the single-channel screen (EXPLORATORY, not a result), detached.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s90.log
stage() { echo "=== S90 $1 $(date -u)" >> $LOG; $PY -u explore_f_s90_ctx.py >> $LOG 2>&1; echo "exit $?" >> $LOG;
          $PY -u explore_f_s90_ctx.py --decide >> $LOG 2>&1;
          git add explore_out/F/ >/dev/null 2>&1
          git commit -q -m "explore_out/F: S90 $1 (EXPLORATORY, not a result)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016a79XyXzJKpgHo6vpuK6B2" >> $LOG 2>&1
          git push -q origin claude/explore-F >> $LOG 2>&1; }
stage phase0
stage phase1
stage phase1b
stage phase2
stage final
echo "=== S90 done $(date -u)" >> $LOG
