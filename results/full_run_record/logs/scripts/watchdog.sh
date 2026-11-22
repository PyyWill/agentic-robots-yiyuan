#!/bin/bash
# STOPS (does not terminate) this pod 45 min after the pipeline finished, or 6 h after launch, unless
# $WORK/status/watchdog.disable exists.  A stopped pod keeps /workspace (where the backups are) and bills storage only.
WORK=/root/f3; START=$(date +%s)
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs -d '\n')
while true; do
  [ -f $WORK/status/watchdog.disable ] && { echo "$(date -u +%FT%TZ) disabled" >> $WORK/logs/watchdog.log; exit 0; }
  now=$(date +%s); why=""
  [ -f $WORK/status/pipeline.done ] && [ $(( now - $(stat -c %Y $WORK/status/pipeline.done) )) -gt 2700 ] && why="45 min after pipeline.done"
  [ $(( now - START )) -gt 21600 ] && why="6 h hard limit"
  [ -n "$why" ] && break
  sleep 30
done
echo "$(date -u +%FT%TZ) stopping pod: $why" >> $WORK/logs/watchdog.log
rsync -a --exclude '*.tmp' $WORK/agentic-robots-yiyuan/results/full/ /workspace/f3-backup/results-full/; rsync -a $WORK/logs $WORK/env $WORK/status /workspace/f3-backup/
curl -sS -m 30 -X POST "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' -d '{"action":"stop"}' >> /workspace/f3-backup/watchdog_stop.out 2>&1
sleep 20
curl -sS -m 30 -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" >> /workspace/f3-backup/watchdog_stop.out 2>&1
