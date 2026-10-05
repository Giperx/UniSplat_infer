"""Color-seam step on UniSplat wide renders. No ground-truth image is required.

Kept for comparison. ``metrics/run_wide.sh`` does not call this script.
Use ``metrics/eval_consistency.py`` for CBSR and PD.

CRCS is the mean absolute horizontal color step, in 0-255 units, at the left
and right third-boundaries and over the whole image. This matches
widedrive_CRCS.py. UniSplat has no render alpha mask, so the primary score is
unmasked. If a GT mask exists and matches the render size, a gt_mask row is
also written; that is not DWSplat's render-alpha score.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import (
    add_common_args,
    collect_renders,
    find_gt,
    load_binary_mask,
    load_scene_ids,
    load_uint8,
    open_report,
    resolve,
    timestamp,
    write_bucket,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


REGIONS = ("L", "R", "Overall")
VARIANTS = ("unmasked", "gt_mask")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--seam-ratio", type=float, default=0.10)
    parser.add_argument("--seam-vertical-ratio", type=float, default=0.5)
    return parser.parse_args()


def compute_crcs(image_uint8, mask, seam_ratio, vertical_ratio):
    height, width = image_uint8.shape[:2]
    diff = np.abs(image_uint8[:, 1:].astype(np.int16) - image_uint8[:, :-1].astype(np.int16)).mean(axis=2)
    v_half = int(height * vertical_ratio / 2)
    v_center = height // 2
    v0 = max(0, v_center - v_half)
    v1 = min(height, v_center + v_half)
    seam_half = int(width * seam_ratio)
    spans = {
        "L": (max(0, width // 3 - seam_half), min(width - 1, width // 3 + seam_half)),
        "R": (max(0, 2 * width // 3 - seam_half), min(width - 1, 2 * width // 3 + seam_half)),
        "Overall": (0, width - 1),
    }
    results = {}
    for name, (x0, x1) in spans.items():
        if name == "Overall":
            values = diff[:, x0:x1]
            pair = None if mask is None else (mask[:, x0:x1] & mask[:, x0 + 1:x1 + 1])
        else:
            values = diff[v0:v1, x0:x1]
            pair = None if mask is None else (mask[v0:v1, x0:x1] & mask[v0:v1, x0 + 1:x1 + 1])
        unmasked = float(values.mean()) if values.size else 0.0
        masked = float(values[pair].mean()) if pair is not None and pair.any() else None
        results[name] = {"unmasked": unmasked, "gt_mask": masked}
    return results


def _fmt(_name, value):
    return f"{float(value):.4f}"


def main():
    args = parse_args()
    preset, render_root, gt_root, val_list = resolve(args)
    if not render_root.is_dir():
        raise SystemExit(f"render root not found: {render_root}")
    scenes, missing_scenes = load_scene_ids(render_root, val_list)
    if not scenes:
        raise SystemExit(f"no rendered scenes under {render_root}")
    jobs = collect_renders(render_root, scenes, args.image_dir)
    if not jobs:
        raise SystemExit(f"no {{frame}}_*_wide images under {render_root}/*/{args.image_dir}")

    keys = tuple(f"{region}_{variant}" for region in REGIONS for variant in VARIANTS)
    names = ("value",)
    global_buckets = {key: [] for key in keys}
    scene_buckets = {}
    bad_mask = 0
    gt_available = gt_root.is_dir()

    for scene, frame, path in tqdm(jobs, desc="CRCS", unit="frame"):
        image = load_uint8(path)
        mask = None
        if gt_available:
            _, mask_path = find_gt(preset, gt_root, scene, frame)
            if mask_path is not None:
                loaded = load_binary_mask(mask_path)
                if loaded.shape == image.shape[:2]:
                    mask = loaded
                else:
                    bad_mask += 1
            else:
                bad_mask += 1
        else:
            bad_mask += 1
        scores = compute_crcs(image, mask, args.seam_ratio, args.seam_vertical_ratio)
        buckets = scene_buckets.setdefault(scene, {key: [] for key in keys})
        for region, values in scores.items():
            for variant in VARIANTS:
                if values[variant] is None:
                    continue
                row = {"value": values[variant]}
                buckets[f"{region}_{variant}"].append(row)
                global_buckets[f"{region}_{variant}"].append(row)

    out_path = render_root / f"unisplat_CRCS_{timestamp()}.txt"
    meta = [
        f"Dataset: {args.dataset}",
        f"Render root: {render_root}",
        f"GT root: {gt_root}",
        f"Val list: {val_list}",
        f"Seam ratio: {args.seam_ratio}",
        f"Seam vertical ratio: {args.seam_vertical_ratio}",
        "CRCS is mean |I(x+1)-I(x)| over RGB, already on the 0-255 scale.",
        "L/R use the middle vertical band around width/3 and 2*width/3. Overall uses the full image.",
        "Unmasked does not use a mask. gt_mask uses the sparse GT mask when it matches the render size.",
        "UniSplat writes no render alpha mask, so gt_mask is not DWSplat's render-mask score.",
        f"Frames without a usable GT mask: {bad_mask}.",
        f"Val-list scenes without a render directory: {len(missing_scenes)}.",
        f"Rendered frames: {len(jobs)}.",
    ]
    handle = open_report(out_path, "UniSplat wide CRCS", meta)
    with handle:
        handle.write("\n" + "=" * 80 + "\nSummary\n" + "=" * 80 + "\n")
        write_bucket(handle, keys, global_buckets, names, _fmt)
        handle.write("\n" + "=" * 80 + "\nPer-scene\n" + "=" * 80 + "\n")
        for scene in sorted(scene_buckets):
            handle.write(f"\nScene {scene}:\n")
            write_bucket(handle, keys, scene_buckets[scene], names, _fmt, indent="  ")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
