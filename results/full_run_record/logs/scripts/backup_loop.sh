#!/bin/bash
WORK=/root/f3; REPO=$WORK/agentic-robots-yiyuan
while true; do
  mkdir -p /workspace/f3-backup/results-full
  rsync -a --exclude '*.tmp' $REPO/results/full/ /workspace/f3-backup/results-full/ 2>/dev/null
  rsync -a $WORK/logs $WORK/env $WORK/status /workspace/f3-backup/ 2>/dev/null
  date -u +%FT%TZ > /workspace/f3-backup/last_backup
  sleep 60
done
