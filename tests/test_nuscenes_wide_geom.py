"""Geometry tests for the nuScenes wide-FOV helpers (numpy only, no torch/GPU).

Run with either::

    python -m pytest tests/test_nuscenes_wide_geom.py -q
    python tests/test_nuscenes_wide_geom.py

The module under test is loaded directly by path so that importing it never
triggers ``dataset/__init__.py`` (which imports torch).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "dataset" / "nuscenes_wide.py"

_spec = importlib.util.spec_from_file_location("nuscenes_wide", MODULE_PATH)
geom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geom)

try:  # pragma: no cover - depends on the environment
    import pytest

    _skip = pytest.skip
except ImportError:  # pragma: no cover
    class _Skipped(Exception):
        pass

    def _skip(message):
        raise _Skipped(message)


def test_plan_input_hw():
    out_h, out_w = geom.plan_input_hw(1600, 900)
    assert (out_h, out_w) == (294, 518), (out_h, out_w)
    assert out_h % 14 == 0 and out_w % 14 == 0


def test_scale_intrinsics():
    K = geom.intrinsics_from_params(1000.0, 2000.0, 800.0, 450.0)
    scaled = geom.scale_intrinsics(K, (1600, 900), (518, 294))
    assert scaled.shape == (3, 3)
    assert np.isclose(scaled[0, 0], 1000.0 * 518 / 1600)
    assert np.isclose(scaled[1, 1], 2000.0 * 294 / 900)
    assert np.isclose(scaled[0, 2], 800.0 * 518 / 1600)
    assert np.isclose(scaled[1, 2], 450.0 * 294 / 900)
    assert np.allclose(scaled[2], [0.0, 0.0, 1.0])


def test_wide_intrinsics():
    K = geom.intrinsics_from_params(809.2209905677063, 809.2209905677063, 829.2196003259838, 481.77842384512485)
    K_resized = geom.scale_intrinsics(K, (1600, 900), (518, 294))
    wide_K, (wide_h, wide_w) = geom.make_wide_intrinsics(K_resized, (518, 294), 3.0)
    assert (wide_h, wide_w) == (294, 1554)
    assert np.isclose(wide_K[0, 0], K_resized[0, 0])
    assert np.isclose(wide_K[1, 1], K_resized[1, 1])
    assert np.isclose(wide_K[1, 2], K_resized[1, 2])
    assert np.isclose(wide_K[0, 2], 1554 / 2.0)
    assert np.allclose(wide_K[2], [0.0, 0.0, 1.0])

    fovx, fovy = geom.intrinsics_to_fov(wide_K)
    assert np.isclose(fovx, 2 * np.arctan(wide_K[0, 2] / wide_K[0, 0]))
    assert np.isclose(fovy, 2 * np.arctan(wide_K[1, 2] / wide_K[1, 1]))


def test_nonsky_drop_mask_flat_order():
    S, H, W = 2, 3, 4
    sky = np.zeros((S, H, W), dtype=bool)
    keep = np.ones((S, H, W), dtype=bool)
    keep[1, 0, 0] = False

    drop = geom.nonsky_drop_mask(keep, sky)
    assert drop.shape == (S * H * W,)
    index = 1 * H * W + 0 * W + 0
    assert bool(drop[index]) is True
    assert int(drop.sum()) == 1


def test_nonsky_drop_mask_sky_removed_shifts_indices():
    S, H, W = 2, 3, 4
    keep = np.ones((S, H, W), dtype=bool)
    keep[1, 0, 0] = False

    sky = np.zeros((S, H, W), dtype=bool)
    sky[0, 0, 0] = True  # one sky pixel removed from the non-sky sequence
    keep[0, 0, 0] = False  # ...and the sky pixel itself is dropped

    drop = geom.nonsky_drop_mask(keep, sky)
    assert drop.shape == (S * H * W - 1,)
    # (1,0,0) was flattened index 12, now index 11 after the earlier sky pixel.
    assert bool(drop[1 * H * W - 1]) is True
    assert int(drop.sum()) == 1
    # Next non-sky pixel (1,0,1) is kept.
    assert bool(drop[1 * H * W]) is False

    # The mirror helper agrees pixel-for-pixel.
    sky_drop = geom.sky_drop_mask(keep, sky)
    assert sky_drop.shape == (1,)
    assert bool(sky_drop[0]) is True


def test_real_nuscenes_files_if_present():
    scene_dir = REPO_ROOT / "data" / "nuscenes" / "processed_10Hz" / "trainval2" / "037"
    intr_path = scene_dir / "intrinsics" / "5.txt"
    extr_path = scene_dir / "cam2ego_extrinsics" / "5.txt"
    if not intr_path.is_file() or not extr_path.is_file():
        _skip(f"nuScenes sample data not found under {scene_dir}")

    fx, fy, cx, cy = geom.parse_intrinsics_text(intr_path.read_text())
    assert all(np.isfinite(v) for v in (fx, fy, cx, cy))

    K = geom.read_intrinsics(intr_path)
    assert K.shape == (3, 3)
    assert np.allclose(K[2], [0.0, 0.0, 1.0])
    assert np.isclose(K[0, 0], fx) and np.isclose(K[1, 1], fy)

    extr = geom.read_cam2ego(extr_path)
    assert extr.shape == (4, 4)
    assert np.allclose(extr[3], [0.0, 0.0, 0.0, 1.0])


def _run_all():
    failures = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        try:
            func()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001 - simple standalone runner
            if error.__class__.__name__.endswith("Skipped"):
                print(f"SKIP {name}: {error}")
                continue
            failures += 1
            print(f"FAIL {name}: {error!r}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
