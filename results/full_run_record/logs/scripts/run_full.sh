#!/bin/bash
# usage: run_full.sh <workers> <attempt-number>.  Runs `python -m f3.run --preset full` and, ONLY if all 120 jobs
# are present with no FAIL line, `python -m f3.analyze results/full`.  Never touches f3/ or tasks.json.
set -uo pipefail
WORK=/root/f3; REPO=$WORK/agentic-robots-yiyuan; WORKERS=$1; N=$2
source $WORK/venv/bin/activate; cd $REPO
stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date -u +%FT%TZ)" "$line"; done; }
LOG=$WORK/logs/full.attempt$N.log
rm -f $WORK/status/pipeline.done
echo "attempt=$N workers=$WORKERS start=$(date -u +%FT%TZ) commit=$(git rev-parse HEAD)" | tee $WORK/status/full.started.$N
python -m f3.run --preset full --workers "$WORKERS" 2>&1 | stamp | tee "$LOG"
rc=${PIPESTATUS[0]}
have=$(find results/full -maxdepth 1 -name '*.jsonl' | wc -l)
fails=$(grep -c 'Z FAIL ' "$LOG" || true)
echo "attempt=$N end=$(date -u +%FT%TZ) run_exit=$rc jsonl=$have fail_lines=$fails" | tee $WORK/status/full.finished.$N
if [ "$rc" = 0 ] && [ "$have" = 120 ] && [ "$fails" = 0 ]; then
  python -m f3.analyze results/full 2>&1 | tee $WORK/logs/analyze.log
  echo "analyze_exit=${PIPESTATUS[0]} end=$(date -u +%FT%TZ)" | tee $WORK/status/analyze.finished
else
  echo "analysis NOT run: the run is incomplete or has failures" | tee $WORK/status/analyze.skipped
fi
date -u +%FT%TZ > $WORK/status/pipeline.done
