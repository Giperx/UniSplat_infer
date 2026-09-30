#!/usr/bin/env bash
# Run single-frame and multi-frame inference plus metrics for every dataset.
# This walks the full val lists and will take a long time.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for name in nuscenes ddad lyft1920 lyft1224 widedrive; do
  bash "$DIR/run_wide.sh" "$name" "$@"
done
