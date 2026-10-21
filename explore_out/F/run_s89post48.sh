#!/bin/bash
# Session F S89-F ALLPOOL fallback at 48000 (explore_f_s89_post; EXPLORATORY, not a result), detached.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/s89post.log
echo "=== S89 post48 start $(date -u)" >> $LOG
$PY -u explore_f_s89_post.py --only O_HEBB_CONV_ALLPOOL_48K >> $LOG 2>&1; echo "exit $?" >> $LOG
git add explore_out/F/ >/dev/null 2>&1
git commit -q -m "explore_out/F: S89-F ALLPOOL fallback at 48000 (EXPLORATORY, not a result)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016a79XyXzJKpgHo6vpuK6B2" >> $LOG 2>&1
git push -q origin claude/explore-F >> $LOG 2>&1
echo "=== S89 post48 done $(date -u)" >> $LOG
