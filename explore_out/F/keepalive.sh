#!/bin/bash
# Harness-tracked keepalive (~115 min): pushes explore_out/F every 23 min; exits early if the queue has finished.
cd /home/user/clankers
for i in 1 2 3 4 5; do
  sleep 1380
  git add explore_out/F/ >/dev/null 2>&1; git commit -q -m "explore_out/F: progress (EXPLORATORY, not a result)" >/dev/null 2>&1; git push -q origin claude/explore-F >/dev/null 2>&1
  pgrep -f "bin/bash explore_out/F/run_(queue|followup|s77|s76)" >/dev/null || { echo "queue finished"; exit 0; }
done
echo "keepalive period over"
