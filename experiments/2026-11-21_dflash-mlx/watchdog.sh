#!/bin/zsh
# Log memory every 20 s; kill the experiment if system free memory falls below 25% or swap starts being used.
while pgrep -f "ar_drafter.py" >/dev/null; do
  free=$(memory_pressure | awk '/free percentage/ {gsub("%","",$5); print $5}')
  swap=$(sysctl -n vm.swapusage | awk '{print $6}')
  rss=$(ps -o rss= -p $(pgrep -f "ar_drafter.py" | head -1) 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
  echo "$(date +%H:%M:%S) free=${free}% swap_used=${swap} python_rss=${rss}G"
  if [ "${free:-100}" -lt 25 ]; then echo "LOW MEMORY: stopping run"; pkill -f ar_drafter.py; fi
  sleep 20
done
