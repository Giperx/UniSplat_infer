"""Seam low-frequency gradient on UniSplat wide renders. No GT image is required.

Kept for comparison. ``metrics/run_wide.sh`` does not call this script.
Use ``metrics/eval_consistency.py`` for CBSR and PD.

IPS is the mean horizontal gradient of a Gaussian low-pass field, reported on
the 0-255 scale, at the left and right third-boundaries. This matches
widedrive_IPS.py. The primary score uses no mask. A gt_mask row, when a
matching GT mask exists, is the pair-valid mean of a GT-weighted low-pass and
is not DWSplat's render-alpha score.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.ndimage import gaussian_filter

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


REGIONS = ("L", "R", "MeanLR")
VARIANTS = ("unmasked", "gt_mask")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--seam-vertical-ratio", type=float, default=0.5)
    parser.add_argument("--band-ratio", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=5.0)
    return parser.parse_args()


def low_frequency(image_uint8, mask, sigma):
    image = image_uint8.astype(np.float64) / 255.0
    if mask is None:
        weight = np.ones(image.shape[:2] + (1,), dtype=np.float64)
    else:
        weight = mask.astype(np.float64)[..., None]
    blurred = gaussian_filter(image * weight, sigma=[sigma, sigma, 0], mode="nearest")
    blurred_weight = gaussian_filter(weight, sigma=[sigma, sigma, 0], mode="nearest")
    low = np.zeros_like(image)
    safe = blurred_weight[:, :, 0] > 1e-5
    for channel in range(3):
        low[:, :, channel][safe] = blurred[:, :, channel][safe] / blurred_weight[:, :, 0][safe]
    return low


def compute_ips(image_uint8, mask, vertical_ratio, band_ratio, sigma):
    height, width = image_uint8.shape[:2]
    low = low_frequency(image_uint8, mask, sigma)
    v_half = int(height * vertical_ratio / 2)
    v_center = height // 2
    v0 = max(0, v_center - v_half)
    v1 = min(height, v_center + v_half)
    band_half = max(1, int(width * band_ratio))
    centers = {"L": width // 3, "R": 2 * width // 3}
    results = {}
    for name, center in centers.items():
        x0 = max(0, center - band_half)
        x1 = min(width, center + band_half)
        band = low[v0:v1, x0:x1]
        grad = np.abs(band[:, 1:] - band[:, :-1]).mean(axis=2)
        unmasked = float(grad.mean() * 255.0) if grad.size else 0.0
        masked = None
        if mask is not None:
            band_mask = mask[v0:v1, x0:x1]
            pair = band_mask[:, 1:] & band_mask[:, :-1]
            if pair.any():
                masked = float(grad[pair].mean() * 255.0)
        results[name] = {"unmasked": unmasked, "masked": masked}
    return results


def _fmt(_name, value):
    return f"{float(value):.4f}"


def _row(value):
    return {"value": value}


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

    for scene, frame, path in tqdm(jobs, desc="IPS", unit="frame"):
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

        plain = compute_ips(image, None, args.seam_vertical_ratio, args.band_ratio, args.sigma)
        masked_scores = None
        if mask is not None:
            masked_scores = compute_ips(image, mask, args.seam_vertical_ratio, args.band_ratio, args.sigma)
        plain["MeanLR"] = {
            "unmasked": 0.5 * (plain["L"]["unmasked"] + plain["R"]["unmasked"]),
            "masked": None,
        }
        if masked_scores is not None:
            left = masked_scores["L"]["masked"]
            right = masked_scores["R"]["masked"]
            masked_scores["MeanLR"] = {
                "unmasked": None,
                "masked": None if left is None or right is None else 0.5 * (left + right),
            }

        buckets = scene_buckets.setdefault(scene, {key: [] for key in keys})
        for region in ("L", "R", "MeanLR"):
            row = _row(plain[region]["unmasked"])
            buckets[f"{region}_unmasked"].append(row)
            global_buckets[f"{region}_unmasked"].append(row)
            if masked_scores is not None and masked_scores[region]["masked"] is not None:
                row = _row(masked_scores[region]["masked"])
                buckets[f"{region}_gt_mask"].append(row)
                global_buckets[f"{region}_gt_mask"].append(row)

    out_path = render_root / f"unisplat_IPS_{timestamp()}.txt"
    meta = [
        f"Dataset: {args.dataset}",
        f"Render root: {render_root}",
        f"GT root: {gt_root}",
        f"Val list: {val_list}",
        f"Sigma: {args.sigma}",
        f"Band ratio: {args.band_ratio}",
        f"Seam vertical ratio: {args.seam_vertical_ratio}",
        "IPS is the mean horizontal gradient of a Gaussian low-pass, times 255.",
        "Unmasked uses a full-image low-pass. gt_mask is the pair-valid mean of a GT-weighted low-pass.",
        "MeanLR is the per-frame mean of L and R.",
        "UniSplat writes no render alpha mask, so gt_mask is not DWSplat's render-mask score.",
        f"Frames without a usable GT mask: {bad_mask}.",
        f"Val-list scenes without a render directory: {len(missing_scenes)}.",
        f"Rendered frames: {len(jobs)}.",
    ]
    handle = open_report(out_path, "UniSplat wide IPS", meta)
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
