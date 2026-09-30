#!/usr/bin/env python3
"""Single-frame nuScenes wide-FOV timing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_wide import main  # noqa: E402


if __name__ == "__main__":
    main(default_dataset="nuscenes", multi=False)
