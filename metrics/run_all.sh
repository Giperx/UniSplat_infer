#!/usr/bin/env bash
# Run one mode for every dataset. Mode is required.
#   bash metrics/run_all.sh single
#   bash metrics/run_all.sh multiframes
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash metrics/run_all.sh single|multiframes [options]" >&2
  exit 1
fi
for name in nuscenes ddad lyft1920 lyft1224 widedrive; do
  bash "$DIR/run_wide.sh" "$name" "$@"
done
