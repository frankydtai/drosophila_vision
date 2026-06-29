#!/bin/bash
# Monitor download until all 74 dataset files are verified complete.
set -euo pipefail
DIR="/scratch/punim0477/yitai/Groschner2022_multiplication"
LOG="$DIR/monitor.log"
EXPECTED=74
TARGET_GB=5.2

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

log "=== monitor started ==="

while true; do
  running=$(pgrep -fc "download_dataset.py" 2>/dev/null || true)
  parts=$(find "$DIR" -name '*.part' 2>/dev/null | wc -l)
  files=$(find "$DIR/A_biophysical_account_of_multiplication_by_a_single_neuron" -type f 2>/dev/null | wc -l)
  size=$(du -sh "$DIR" | cut -f1)

  part_info=""
  if [[ "$parts" -gt 0 ]]; then
    part_info=$(find "$DIR" -name '*.part' -exec stat -c '%s %n' {} \; 2>/dev/null | while read -r sz path; do
      mb=$((sz / 1024 / 1024))
      echo "$(basename "$path"): ${mb}MB"
    done | tr '\n' ' ')
  fi

  log "running=$running files=$files/$EXPECTED size=$size parts=$parts $part_info"

  verify=$(cd "$DIR" && "$DIR/run.sh" -c "
from download_dataset import build_jobs, verify_all
jobs = build_jobs()
missing, bad = verify_all(jobs)
total = sum(s for _,_,s in jobs)
got = sum(d.stat().st_size for _,d,_ in jobs if d.exists())
print(f'verify missing={len(missing)} bad={len(bad)} got={got/1e9:.2f}GB')
" 2>/dev/null || echo "verify=error")

  log "$verify"

  if echo "$verify" | grep -q 'missing=0 bad=0'; then
    log "=== ALL FILES COMPLETE ==="
    exit 0
  fi

  if [[ "$running" -eq 0 ]]; then
    log "No download process running; starting fixed downloader..."
    cd "$DIR"
    nohup "$DIR/run.sh" -u download_dataset.py >> download.log 2>&1 &
    sleep 5
  fi

  sleep 60
done
