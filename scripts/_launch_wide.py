"""Load a wide-inference driver and pin its dataset preset."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def launch(core_name: str, dataset: str, argv=None):
    path = Path(__file__).resolve().parent / core_name
    spec = importlib.util.spec_from_file_location(core_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv, default_dataset=dataset)
