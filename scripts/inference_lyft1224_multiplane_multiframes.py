#!/usr/bin/env python3
"""3-frame Lyft 1224 multiplane inference.

Only the newest frame is rendered. The three planes are the main camera yawed
by one horizontal field of view and stitched left-center-right.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _launch_wide import launch  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--multiplane" not in argv:
        argv = ["--multiplane", *argv]
    return launch("inference_nuscenes_wide_multiframes.py", "lyft1224", argv)


if __name__ == "__main__":
    main()
