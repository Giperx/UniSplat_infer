#!/usr/bin/env python3
"""Single-frame Lyft 1920 wide-FOV inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _launch_wide import launch  # noqa: E402


def main(argv=None):
    return launch("inference_nuscenes_wide.py", "lyft1920", argv)


if __name__ == "__main__":
    main()
