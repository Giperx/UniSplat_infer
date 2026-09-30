#!/usr/bin/env python3
"""3-frame DDAD wide-FOV inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _launch_wide import launch  # noqa: E402


def main(argv=None):
    return launch("inference_nuscenes_wide_multiframes.py", "ddad", argv)


if __name__ == "__main__":
    main()
