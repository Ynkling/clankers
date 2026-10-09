#!/bin/bash
# Session F follow-up queue (EXPLORATORY, not a result), detached; kept alive by explore_out/F/keepalive.sh.
# S70 oracle phase first (24000; doubled once to 48000 if either oracle arm binds < 2/3), then S68, S69 and S70's
# learned arms in one pool.
cd /home/user/clankers
PY=/usr/bin/python3.11
LOG=explore_out/F/followup.log
echo "=== S70 phase 1 $(date -u)" >> $LOG
$PY -u -c "import explore_f_common as fc, explore_f_rh_lite as s; fc.drive('s70o', [s], [])" >> $LOG 2>&1
$PY -u explore_f_rh_lite.py >> $LOG 2>&1
if grep -q '"phase1": "double"' explore_out/F/rh_lite_budget.json && ! grep -q '"budget"' explore_out/F/rh_lite_budget.json; then
  echo "=== S70 phase 1, doubled budget $(date -u)" >> $LOG
  $PY -u -c "import explore_f_common as fc, explore_f_rh_lite as s; fc.drive('s70o', [s], [])" >> $LOG 2>&1
  $PY -u explore_f_rh_lite.py >> $LOG 2>&1
fi
echo "=== S68, S69, S70 phase 2 $(date -u)" >> $LOG
$PY -u explore_f_premise.py >> $LOG 2>&1; echo "exit $?" >> $LOG
echo "=== done $(date -u)" >> $LOG
