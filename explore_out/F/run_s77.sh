#!/bin/bash
# Session F S77 queue (EXPLORATORY, not a result), detached; kept alive by explore_out/F/keepalive.sh.
cd /home/user/clankers
echo "=== S77 $(date -u)" >> explore_out/F/s77.log
/usr/bin/python3.11 -u explore_f_probe.py >> explore_out/F/s77.log 2>&1; echo "exit $?" >> explore_out/F/s77.log
