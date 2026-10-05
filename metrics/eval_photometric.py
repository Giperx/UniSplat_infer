"""Score UniSplat wide renders against DWSplat-style ground truth.

Sparse datasets (nuScenes, DDAD, Lyft) follow NuScenes_multiplane_v2.py:
Left/Right use the GT mask and sparse luminance SSIM; Center uses dense
window SSIM; Center_masked also applies the ego-car mask. LPIPS is computed
only on Center and Center_masked.

WideDrive follows widedrive_metrics.py: camera-2 GT is resized bicubic to the
render size, and Full/Left/Center/Right are scored densely. UniSplat writes no
render alpha mask, so masked rows stay empty unless a GT mask exists.

``--histogram-match`` CDF-aligns rendered pixels to GT inside each scored mask
before the same metrics, matching the DWSplat HM scripts.
"""

from __future__ import annotations

import argparse

from common import (
    DENSE_REGIONS,
    DENSE_VARIANTS,
    PHOTOMETRIC_NAMES,
    SPARSE_REGIONS,
    add_common_args,
    collect_renders,
    find_gt,
    fmt_photometric,
    format_equal_region_means,
    load_binary_mask,
    load_rgb,
    load_scene_ids,
    open_report,
    resolve,
    timestamp,
    write_bucket,
)
from photometric import load_lpips, score_dense, score_sparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--histogram-match", action="store_true")
    return parser.parse_args()


def dense_keys():
    return tuple(f"{region}_{variant}" for region in DENSE_REGIONS for variant in DENSE_VARIANTS)


def main():
    args = parse_args()
    preset, render_root, gt_root, val_list = resolve(args)
    if not render_root.is_dir():
        raise SystemExit(f"render root not found: {render_root}")
    if not gt_root.is_dir():
        raise SystemExit(
            f"GT root not found: {gt_root}\n"
            "Photometric and histogram-matched metrics need this directory. "
            "CRCS and IPS do not."
        )

    scenes, missing_scenes = load_scene_ids(render_root, val_list)
    if not scenes:
        raise SystemExit(f"no rendered scenes under {render_root}")
    jobs = collect_renders(render_root, scenes, args.image_dir)
    if not jobs:
        raise SystemExit(f"no {{frame}}_*_wide images under {render_root}/*/{args.image_dir}")

    device_name = "cuda"
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = str(device)
    except ImportError as exc:
        raise SystemExit(f"torch is required for photometric metrics: {exc}") from exc
    print(f"Loading LPIPS alex on {device_name}. First run may download AlexNet weights.", flush=True)
    lpips_fn = load_lpips(device)

    style = preset["style"]
    keys = SPARSE_REGIONS if style == "sparse" else dense_keys()
    global_buckets = {key: [] for key in keys}
    scene_buckets = {}
    skipped = {
        "missing_gt": 0,
        "missing_mask": 0,
        "size_mismatch": 0,
        "empty": 0,
        "resized": 0,
        "car_mask_missing": 0,
    }
    render_size = None

    for scene, frame, path in tqdm(jobs, desc="photometric", unit="frame"):
        render, _ = load_rgb(path)
        render_size = render.shape[1], render.shape[0]
        gt_path, mask_path = find_gt(preset, gt_root, scene, frame)
        if gt_path is None:
            skipped["missing_gt"] += 1
            continue
        if style == "sparse":
            if mask_path is None:
                skipped["missing_mask"] += 1
                continue
            gt, _ = load_rgb(gt_path)
            gt_mask = load_binary_mask(mask_path)
            if gt.shape[:2] != render.shape[:2] or gt_mask.shape != render.shape[:2]:
                skipped["size_mismatch"] += 1
                continue
            frame_metrics, car_missing = score_sparse(
                render, gt, gt_mask, preset, scene, device, lpips_fn, args.histogram_match,
            )
            if car_missing:
                skipped["car_mask_missing"] += 1
        else:
            gt, resized = load_rgb(gt_path, (render.shape[1], render.shape[0]))
            if resized:
                skipped["resized"] += 1
            gt_mask = None
            if mask_path is not None:
                gt_mask = load_binary_mask(mask_path, (render.shape[1], render.shape[0]))
            frame_metrics = score_dense(render, gt, gt_mask, device, lpips_fn, args.histogram_match)
        if not frame_metrics:
            skipped["empty"] += 1
            continue
        buckets = scene_buckets.setdefault(scene, {key: [] for key in keys})
        for key, value in frame_metrics.items():
            if key not in buckets:
                continue
            buckets[key].append(value)
            global_buckets[key].append(value)

    scored = len(global_buckets[keys[0]])
    tag = "HM" if args.histogram_match else "photometric"
    out_path = render_root / f"unisplat_{tag}_{timestamp()}.txt"
    expected = preset["expected_wh"]
    meta = [
        f"Dataset: {args.dataset}",
        f"Style: {style}",
        f"Render root: {render_root}",
        f"GT root: {gt_root}",
        f"Val list: {val_list}",
        f"Histogram match: {str(bool(args.histogram_match)).lower()}",
        f"Expected render size WxH: {expected[0]}x{expected[1]}",
        f"Observed render size WxH: {render_size[0]}x{render_size[1]}" if render_size else "Observed render size: none",
        "UniSplat saves JPEG quality 95. GT is PNG, so photometric error includes JPEG compression.",
        "UniSplat does not write a render alpha mask.",
        "MAE and RMSE are reported on the 0-255 scale. PSNR uses [0, 1] images.",
    ]
    if style == "sparse":
        meta.extend([
            "Left/Right SSIM is sparse luminance SSIM on the GT mask. Center SSIM is dense window-11.",
            "LPIPS (alex, spatial) is computed on Center and Center_masked only.",
            "Center_masked is the GT mask AND the ego-car mask. Values above 127 are valid.",
            "Overall is the pixel-weighted mean of Left, Center, and Right, not Center_masked.",
            "Mean_LR is (Left+Right)/2 and Mean_LRC is (Left+Right+Center)/3 of those region means.",
            "Those two means use equal weight on the reported region scores. They are not pixel-weighted, and they exclude Center_masked.",
            "Histogram matching, when enabled, is done independently inside each region mask.",
        ])
    else:
        meta.extend([
            "Dense GT is camera 2, resized bicubic to the render size when the sizes differ.",
            "Regions are width thirds. Masked rows use a GT mask when one exists; WideDrive has none.",
            "LPIPS (alex, spatial) is computed on every region.",
        ])
    meta.append(
        "Skipped missing GT: {missing_gt}. Missing GT mask: {missing_mask}. "
        "Size mismatch: {size_mismatch}. Empty: {empty}. GT resized: {resized}. "
        "Frames without ego-car mask: {car_mask_missing}.".format(**skipped)
    )
    meta.append(f"Val-list scenes without a render directory: {len(missing_scenes)}.")
    meta.append(f"Scored frames: {scored} / {len(jobs)}.")

    handle = open_report(out_path, f"UniSplat wide {tag}", meta)
    with handle:
        handle.write("\n" + "=" * 80 + "\nSummary\n" + "=" * 80 + "\n")
        write_bucket(handle, keys, global_buckets, PHOTOMETRIC_NAMES, fmt_photometric)
        if style == "sparse":
            handle.write(format_equal_region_means(global_buckets, PHOTOMETRIC_NAMES, fmt_photometric))
        handle.write("\n" + "=" * 80 + "\nPer-scene\n" + "=" * 80 + "\n")
        for scene in sorted(scene_buckets):
            handle.write(f"\nScene {scene}:\n")
            write_bucket(handle, keys, scene_buckets[scene], PHOTOMETRIC_NAMES, fmt_photometric, indent="  ")
            if style == "sparse":
                handle.write(
                    format_equal_region_means(
                        scene_buckets[scene], PHOTOMETRIC_NAMES, fmt_photometric, indent="  ",
                    )
                )
        if missing_scenes:
            handle.write("\nScenes in the val list with no render directory:\n")
            for scene in missing_scenes:
                handle.write(f"  {scene}\n")
    print(f"Wrote {out_path}", flush=True)
    if scored == 0:
        raise SystemExit("no frames were scored; see the report for skip counts")


if __name__ == "__main__":
    main()
