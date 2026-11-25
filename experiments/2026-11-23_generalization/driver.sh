#!/usr/bin/env bash
# Job-queue driver: runs queue.txt one line (shell command) at a time, so jobs can be inserted or reordered while it
# runs (edit queue.txt only through ./q.sh). Every job is resumable. Log: driver.log.  Stop: touch STOP.
cd "$(dirname "$0")"
touch queue.txt
while [ ! -f STOP ]; do
  JOB=$(flock queue.lock sh -c 'head -1 queue.txt; sed -i 1d queue.txt')
  if [ -z "$JOB" ]; then sleep 20; continue; fi
  echo "=== JOB $(date -u +%H:%M:%S) $JOB" >> driver.log
  bash -c "$JOB" >> driver.log 2>&1
  echo "=== END $(date -u +%H:%M:%S) rc=$? $JOB" >> driver.log
done
