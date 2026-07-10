#!/usr/bin/env bash

set -euo pipefail

base_dir="${1:-/home/outputs/PDF}"
poll_interval="${POLL_INTERVAL:-2}"

find_latest_log() {
  find "$base_dir" -type f -name "log_train.log" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

echo "Watching latest train log under: $base_dir"

latest_log="$(find_latest_log || true)"
while [[ -z "${latest_log:-}" ]]; do
  echo "No training log found yet. Waiting..."
  sleep "$poll_interval"
  latest_log="$(find_latest_log || true)"
done

echo "Tailing: $latest_log"
tail -f "$latest_log"
