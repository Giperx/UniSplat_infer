#!/usr/bin/env python3
"""Single-frame Lyft 1920 multiplane inference.

Three pinhole planes from the main camera, yawed by one horizontal field of
view and stitched left-center-right. Ground truth is sparseMultiplaneImages3.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _launch_wide import launch  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--multiplane" not in argv:
        argv = ["--multiplane", *argv]
    return launch("inference_nuscenes_wide.py", "lyft1920", argv)


if __name__ == "__main__":
    main()
