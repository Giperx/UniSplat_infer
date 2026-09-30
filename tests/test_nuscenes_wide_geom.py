"""Geometry tests for the nuScenes wide-FOV helpers (numpy only, no torch/GPU).

Run with either::

    python -m pytest tests/test_nuscenes_wide_geom.py -q
    python tests/test_nuscenes_wide_geom.py

The module under test is loaded directly by path so that importing it never
triggers ``dataset/__init__.py`` (which imports torch).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "dataset" / "nuscenes_wide.py"

_spec = importlib.util.spec_from_file_location("nuscenes_wide", MODULE_PATH)
geom = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = geom
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


def test_enumerate_windows_causal():
    frames = ["000", "001", "002", "003"]
    windows = geom.enumerate_windows(frames, 3)
    assert windows == [("000", "001", "002"), ("001", "002", "003")]
    assert geom.select_windows(frames, 3, frame="2") == [("000", "001", "002")]
    assert geom.enumerate_windows(["000", "001"], 3) == []
    assert geom.frames_are_consecutive(("000", "001", "002"))
    assert not geom.frames_are_consecutive(("000", "002", "003"))


def test_history_masks_render_camera():
    assert geom.mask_render_camera(is_newest=False) is True
    assert geom.mask_render_camera(is_newest=True) is False
    assert geom.mask_render_camera(is_newest=True, mask_render_view=True) is True
    cameras = (5, 4, 3)
    masks = {
        5: np.zeros((2, 2), dtype=bool),
        4: np.zeros((2, 2), dtype=bool),
        3: np.ones((2, 2), dtype=bool),
    }
    history = geom.build_car_keep_mask(
        masks, cameras, (2, 2), render_camera=5, mask_render_view=True
    )
    newest = geom.build_car_keep_mask(
        masks, cameras, (2, 2), render_camera=5, mask_render_view=False
    )
    assert history.shape == (3, 2, 2)
    assert not history[0].any()
    assert newest[0].all()
    assert not newest[1].any()
    assert newest[2].all()


def test_dataset_mask_paths():
    assert geom.camera_mask_path("/masks", 5).name == "CAM_BACK_mask.png"
    lyft = geom.camera_mask_path("/ego_car_masks", 5, mask_kind="lyft", mask_ext="jpg")
    assert lyft.as_posix().endswith("/ego_car_masks/5.jpg")
    ddad = geom.camera_mask_path("/valid", 3, mask_kind="ddad", scene="000", mask_ext="jpg")
    assert ddad.as_posix().endswith("/valid/000/ego_car_masks/3.jpg")
    try:
        geom.camera_mask_path("/x", 5, mask_kind="none")
    except ValueError:
        pass
    else:
        raise AssertionError("WideDrive must not resolve an ego-car mask path")


def test_fill_preset_widedrive_disables_mask():
    class Args:
        pass

    args = Args()
    args.dataset = "widedrive"
    args.data_root = None
    args.scene_list = None
    args.cameras = None
    args.render_camera = None
    args.car_mask_root = None
    args.output_dir = None
    args.disable_car_mask = False

    geom.fill_preset_args(args, multi=False)
    assert args.data_root.endswith("WideDriveVal")
    assert args.scene_list.endswith("val.txt")
    assert args.cameras == "5,4,3"
    assert args.render_camera == 5
    assert args.disable_car_mask is True
    assert args.mask_kind == "none"
    assert args.output_dir == "outputs/widedrive_wide"


def test_real_dataset_masks_if_present():
    ddad = REPO_ROOT / "data" / "ddad" / "valid" / "000" / "ego_car_masks" / "5.jpg"
    lyft = REPO_ROOT / "data" / "lyft" / "lyft_val1920_3cams" / "ego_car_masks" / "5.jpg"
    wide = REPO_ROOT / "data" / "WideDrive_processed" / "WideDriveVal" / "val.txt"
    if not ddad.is_file() or not lyft.is_file() or not wide.is_file():
        _skip("ddad/lyft/widedrive sample data not mounted")
    assert geom.camera_mask_path(ddad.parents[2], 5, mask_kind="ddad", scene="000", mask_ext="jpg") == ddad
    assert geom.camera_mask_path(lyft.parent, 5, mask_kind="lyft", mask_ext="jpg") == lyft
    scenes = geom.read_scene_list(wide)
    assert scenes and scenes[0].startswith("Town")


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
