#!/usr/bin/env bash
# ./q.sh front "CMD" | ./q.sh back "CMD" | ./q.sh show
cd "$(dirname "$0")"; touch queue.txt
case "$1" in
  front) flock queue.lock sh -c 'printf "%s\n" "$1" | cat - queue.txt > queue.tmp && mv queue.tmp queue.txt' _ "$2" ;;
  back)  flock queue.lock sh -c 'printf "%s\n" "$1" >> queue.txt' _ "$2" ;;
  show)  cat -n queue.txt ;;
esac
