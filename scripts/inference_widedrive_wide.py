#!/usr/bin/env python3
"""Single-frame WideDrive wide-FOV inference. No ego-car mask."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _launch_wide import launch  # noqa: E402


def main(argv=None):
    return launch("inference_nuscenes_wide.py", "widedrive", argv)


if __name__ == "__main__":
    main()
