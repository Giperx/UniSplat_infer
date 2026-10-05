#!/usr/bin/env bash
# Scores the 1224 subset, then writes the frame-count-weighted Lyft total
# under outputs/lyft1920_wide* when the 1920 report already exists.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_wide.sh" lyft1224 "$@"
