"""Pair UniSplat wide renders with DWSplat-style ground truth.

Renders are ``outputs/<dataset>_wide/<scene>/rgb/{frame}_5_wide.jpg`` and the
multi-frame equivalents under ``outputs/<dataset>_wide_multiframes``. Multi-frame
saves only the newest frame of each window, so pairing by frame id is enough.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Pillow filter ids. Stable across Pillow 9 and 10, and clearer to type checkers
# than Image.BICUBIC / Image.Resampling.BICUBIC.
_BICUBIC = 3
_NEAREST = 0

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
RENDER_RE = re.compile(r"^(\d+)_\d+_wide\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
PHOTOMETRIC_NAMES = ("psnr", "mae", "rmse", "ssim", "lpips")
SPARSE_REGIONS = ("Left", "Center", "Center_masked", "Right", "Overall")
DENSE_REGIONS = ("Full", "Left", "Center", "Right")
DENSE_VARIANTS = ("unmasked", "masked")

# Paths match dataset/nuscenes_wide.py presets and the GT folders on disk.
# nuScenes sparse GT is the DWSplat location; it is not in this checkout.
PRESETS = {
    "nuscenes": {
        "style": "sparse",
        "gt_root": "data/nuscenes/sparseMultiplaneImages3_1554x294",
        "val_list": "data/nuscenes/processed_10Hz/trainval2/nuScenes_Val.txt",
        "single_render": "outputs/nuscenes_wide",
        "multi_render": "outputs/nuscenes_wide_multiframes",
        "car_mask": "data/nuscenes/processed_10Hz/nuscenes_mask/CAM_BACK_mask.png",
        "expected_wh": (1554, 294),
    },
    "ddad": {
        "style": "sparse",
        "gt_root": "data/ddad/sparseMultiplaneImages3_1554x322",
        "val_list": "data/ddad/valid/valid.txt",
        "single_render": "outputs/ddad_wide",
        "multi_render": "outputs/ddad_wide_multiframes",
        "car_mask": "data/ddad/valid/{scene}/ego_car_masks/5.jpg",
        "expected_wh": (1554, 322),
    },
    "lyft1920": {
        "style": "sparse",
        "gt_root": "data/lyft/1920_sparseMultiplaneWideFOVImages3",
        "val_list": "data/lyft/lyft_val1920_3cams/lyft_val1920.txt",
        "single_render": "outputs/lyft1920_wide",
        "multi_render": "outputs/lyft1920_wide_multiframes",
        "car_mask": "data/lyft/lyft_val1920_3cams/ego_car_masks/5.jpg",
        "expected_wh": (1554, 294),
    },
    "lyft1224": {
        "style": "sparse",
        "gt_root": "data/lyft/1224_sparseMultiplaneWideFOVImages3",
        "val_list": "data/lyft/lyft_val1224_3cams/lyft_val1224.txt",
        "single_render": "outputs/lyft1224_wide",
        "multi_render": "outputs/lyft1224_wide_multiframes",
        "car_mask": "data/lyft/lyft_val1224_3cams/ego_car_masks/5.jpg",
        "expected_wh": (1554, 434),
    },
    "widedrive": {
        "style": "dense",
        "gt_root": "data/WideDrive_processed/WideDriveVal",
        "val_list": "data/WideDrive_processed/WideDriveVal/val.txt",
        "single_render": "outputs/widedrive_wide",
        "multi_render": "outputs/widedrive_wide_multiframes",
        "car_mask": "",
        "expected_wh": (1554, 294),
    },
}

_MASK_CACHE = {}


def add_common_args(parser):
    parser.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--render-root", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--image-dir", default="rgb")


def _layout():
    """Load the inference path locator without importing dataset/__init__.py."""
    module = globals().get("_LAYOUT")
    if module is not None:
        return module
    path = Path(__file__).resolve().parents[1] / "dataset" / "nuscenes_wide.py"
    spec = importlib.util.spec_from_file_location("unisplat_nuscenes_wide_layout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    globals()["_LAYOUT"] = module
    return module


def resolve(args):
    preset = PRESETS[args.dataset]
    layout = _layout()
    render_root = Path(args.render_root or preset["single_render"])
    gt_root = Path(args.gt_root or preset["gt_root"])
    val_list = Path(args.val_list or preset["val_list"])
    gt_found = layout.locate_dir(gt_root)
    if gt_found is not None:
        gt_root = gt_found
    listed = layout.locate_file(val_list)
    if listed is not None:
        val_list = listed
    return preset, render_root, gt_root, val_list


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_scene_list(path):
    scenes = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            scenes.append(line.split()[0])
    return scenes


def load_scene_ids(render_root, val_list):
    """Val-list order, restricted to scenes that actually have a render directory."""
    root = Path(render_root)
    rendered = {path.name for path in root.iterdir() if path.is_dir()} if root.is_dir() else set()
    if val_list and Path(val_list).is_file():
        wanted = read_scene_list(val_list)
        scenes = [scene for scene in wanted if scene in rendered]
        missing = [scene for scene in wanted if scene not in rendered]
        return scenes, missing
    return sorted(rendered), []


def list_render_frames(scene_dir, image_dir="rgb"):
    folder = Path(scene_dir) / image_dir
    if not folder.is_dir():
        return []
    frames = []
    for path in folder.iterdir():
        match = RENDER_RE.match(path.name)
        if match:
            frames.append((match.group(1), path))
    frames.sort(key=lambda item: int(item[0]))
    return frames


def collect_renders(render_root, scenes, image_dir="rgb"):
    jobs = []
    for scene in scenes:
        for frame, path in list_render_frames(Path(render_root) / scene, image_dir):
            jobs.append((scene, frame, path))
    return jobs


def _first_existing(directory, stems):
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for stem in stems:
        for ext in IMAGE_EXTS:
            path = directory / f"{stem}{ext}"
            if path.is_file():
                return path
    return None


def find_gt(preset, gt_root, scene, frame):
    """Return (rgb_path, mask_path). Either may be None."""
    gt_root = Path(gt_root)
    if preset["style"] == "sparse":
        stems = (
            f"{frame}_5_multiplane_wide",
            f"{frame}_5_sparse_wide",
            f"{frame}_5_wide",
        )
        return (
            _first_existing(gt_root / scene / "rgb", stems),
            _first_existing(gt_root / scene / "mask", stems),
        )
    dense_stems = (f"{frame}_2", f"{frame}_2_sparse_wide", f"{frame}_5_multiplane_wide")
    rgb = _first_existing(gt_root / scene / "images", (f"{frame}_2",))
    if rgb is None:
        rgb = _first_existing(gt_root / scene / "rgb", dense_stems)
    mask = _first_existing(gt_root / scene / "mask", dense_stems)
    return rgb, mask


def car_mask_path(preset, scene):
    template = preset.get("car_mask") or ""
    if not template:
        return None
    return Path(template.format(scene=scene))


def load_rgb(path, size_wh=None):
    """RGB float image in [0, 1]. ``size_wh`` is PIL (width, height)."""
    image = Image.open(path).convert("RGB")
    resized = False
    if size_wh is not None and image.size != tuple(size_wh):
        image = image.resize(tuple(size_wh), _BICUBIC)
        resized = True
    array = np.asarray(image).astype(np.float32) / 255.0
    return array, resized


def load_uint8(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_binary_mask(path, size_wh=None):
    """Valid pixels are intensities above 127. ``size_wh`` is PIL (width, height)."""
    key = (str(path), None if size_wh is None else tuple(size_wh))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    image = Image.open(path).convert("L")
    if size_wh is not None and image.size != tuple(size_wh):
        image = image.resize(tuple(size_wh), _NEAREST)
    mask = np.asarray(image) > 127
    _MASK_CACHE[key] = mask
    return mask


def strip_bounds(width):
    """Three equal camera strips. 1554 -> 518 / 518 / 518."""
    third = width // 3
    return {
        "Left": (0, third),
        "Center": (third, 2 * third),
        "Right": (2 * third, width),
    }


def dense_bounds(width):
    third = width // 3
    return {
        "Full": (0, width),
        "Left": (0, third),
        "Center": (third, 2 * third),
        "Right": (2 * third, width),
    }


def fmt_photometric(name, value):
    if value is None or not np.isfinite(value):
        return "n/a"
    if name in ("mae", "rmse"):
        return f"{float(value) * 255.0:.2f}"
    if name == "psnr":
        return f"{float(value):.2f}"
    return f"{float(value):.4f}"


def format_bucket(rows, names, formatter):
    if not rows:
        return "no frames"
    parts = [f"n={len(rows)}"]
    for name in names:
        values = [row[name] for row in rows if name in row and row[name] is not None and np.isfinite(row[name])]
        if values:
            parts.append(f"{name.upper()}={formatter(name, float(np.mean(values)))}")
    return "  ".join(parts)


# Equal weight on the reported region means. Not the pixel-weighted Overall,
# and not Center_masked. WideDrive already has a full-image score, so callers
# skip these for dense reports.
EQUAL_REGION_MEANS = (
    ("Mean_LR", ("Left", "Right")),
    ("Mean_LRC", ("Left", "Right", "Center")),
)


def format_equal_region_means(buckets, names, formatter, indent=""):
    """(Left+Right)/2 and (Left+Right+Center)/3 of the region means.

    A metric is included only when every region in the group has a finite
    value, so sparse Left/Right LPIPS is left out instead of dividing Center
    by three.
    """
    lines = []
    for label, regions in EQUAL_REGION_MEANS:
        stats = []
        counts = []
        missing = False
        for region in regions:
            rows = buckets.get(region) or []
            if not rows:
                missing = True
                break
            region_stats = {}
            for name in names:
                values = [
                    row[name]
                    for row in rows
                    if name in row and row[name] is not None and np.isfinite(row[name])
                ]
                if values:
                    region_stats[name] = float(np.mean(values))
            stats.append(region_stats)
            counts.append(len(rows))
        if missing:
            continue
        shared = set(stats[0])
        for region_stats in stats[1:]:
            shared &= set(region_stats)
        parts = []
        if counts and len(set(counts)) == 1:
            parts.append(f"n={counts[0]}")
        for name in names:
            if name not in shared:
                continue
            value = sum(region_stats[name] for region_stats in stats) / len(regions)
            parts.append(f"{name.upper()}={formatter(name, value)}")
        if not any(part.startswith(("PSNR=", "MAE=", "SSIM=", "LPIPS=", "VALUE=")) for part in parts):
            continue
        lines.append(f"{indent}{label:20s}: {'  '.join(parts)}\n")
    return "".join(lines)


def write_bucket(handle, keys, buckets, names, formatter, indent=""):
    for key in keys:
        text = format_bucket(buckets.get(key, []), names, formatter)
        handle.write(f"{indent}{key:20s}: {text}\n")


def open_report(path, title, meta_lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    handle.write(f"=== {title} ===\n")
    handle.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    for line in meta_lines:
        handle.write(line.rstrip() + "\n")
    return handle
