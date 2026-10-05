"""Demo: wide-view visual consistency, with and without GT.

Scores one render root against the resized 1554x294 sparse wide GT.
No-GT metrics compare a seam to the image's own nearby texture, so global
blur does not look like a good seam. GT metrics separate a low-frequency
brightness error from the exposure correction needed to explain it.

This is a paired demo, not the full dataset runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab

SCENES = ("Town10HD_scene_0009", "Town10HD_scene_0019")
SEAMS = (518, 1036)
THIRDS = ((0, 518), (518, 1036), (1036, 1554))
INTERIORS = ((0, 454), (582, 972), (1100, 1554))
VERT_RATIO = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", default="data/WideDrive_expusre_demo_processed/sparseWideFOVImages3_1554x294")
    parser.add_argument("--unisplat-root", default="outputs/widedrive_expusre_demo_wide")
    parser.add_argument(
        "--dwsplat-root",
        default="/home/xiongzhp/FeedForward/2027cvpr/DWSplat_cvpr/renders_save/widedrive/maincam_cam2_wide/exposure_demo_epoch_24-step_20000_safe",
    )
    parser.add_argument("--out", default="outputs/consistency_demo/exposure_demo_compare.txt")
    return parser.parse_args()


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def luminance(rgb):
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def vertical_rows(height):
    half = int(height * VERT_RATIO / 2)
    center = height // 2
    return slice(max(0, center - half), min(height, center + half))


def grad_cols(width, center, near, far):
    xs = np.arange(width - 1)
    dist = np.abs(xs - center)
    return (dist >= near) & (dist <= far)


def row_median(grad, rows, cols):
    patch = grad[rows][:, cols]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch.mean(axis=1)))


def seam_pair(grad, rows, width, center):
    seam = row_median(grad, rows, grad_cols(width, center, 0, 8))
    ref = row_median(grad, rows, grad_cols(width, center, 24, 64))
    return seam, ref


def ratio_scores(seam, ref):
    raw = seam / (ref + 1e-3)
    signed = float(np.log((seam + 1.0) / (ref + 1.0)))
    return raw, signed, abs(signed)


def panel_detail(grad):
    scores = []
    for x0, x1 in INTERIORS:
        patch = grad[:, x0:x1 - 1]
        if patch.size == 0:
            continue
        scores.append(float(np.median(patch.mean(axis=1))))
    return float(np.median(scores)) if scores else 0.0


def panel_level(field):
    stats = []
    for x0, x1 in INTERIORS:
        values = field[:, x0:x1]
        mu = float(np.median(values))
        iqr = float(np.percentile(values, 90) - np.percentile(values, 10))
        stats.append((mu, iqr))
    gaps = []
    for (mu_a, iqr_a), (mu_b, iqr_b) in zip(stats, stats[1:]):
        gaps.append(abs(mu_a - mu_b) / (iqr_a + iqr_b + 1.0))
    return 0.5 * sum(gaps), stats


def fit_affine(src, dst):
    """Robust ``dst ~= a * src + b`` with ``a > 0``."""
    src = np.asarray(src, dtype=np.float64).reshape(-1)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1)
    if src.size > 80000:
        pick = np.linspace(0, src.size - 1, 80000).astype(np.int64)
        src = src[pick]
        dst = dst[pick]
    weights = np.ones_like(src)
    slope, bias = 1.0, 0.0
    for _ in range(6):
        sw = float(weights.sum())
        sx = float((weights * src).sum())
        sy = float((weights * dst).sum())
        sxx = float((weights * src * src).sum())
        sxy = float((weights * src * dst).sum())
        det = sxx * sw - sx * sx
        if abs(det) < 1e-8 or sw <= 0:
            break
        slope = (sxy * sw - sx * sy) / det
        bias = (sxx * sy - sx * sxy) / det
        if slope < 1e-3:
            slope = 1e-3
            bias = (sy - slope * sx) / sw
        resid = dst - (slope * src + bias)
        mad = float(np.median(np.abs(resid - np.median(resid)))) + 1e-6
        delta = 1.345 * 1.4826 * mad
        heavy = np.abs(resid) > delta
        weights = np.ones_like(resid)
        weights[heavy] = delta / np.abs(resid[heavy])
    return float(slope), float(bias)


def chroma_grad(rgb):
    lab = rgb2lab(rgb.astype(np.float64) / 255.0)
    ab = np.empty_like(lab[..., 1:3])
    for channel in range(2):
        ab[..., channel] = gaussian_filter(lab[..., 1 + channel], sigma=5)
    delta = ab[:, 1:] - ab[:, :-1]
    return np.sqrt(np.sum(delta * delta, axis=2))


def photometric(render, gt):
    diff = render.astype(np.float64) - gt.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean((diff / 255.0) ** 2))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    return mae, psnr


def score_pair(render, gt):
    if render.shape != gt.shape:
        raise ValueError(f"shape {render.shape} != {gt.shape}")
    height, width = render.shape[:2]
    rows = vertical_rows(height)
    y_render = luminance(render)
    y_gt = luminance(gt)
    low_render = gaussian_filter(y_render, sigma=5)
    low_gt = gaussian_filter(y_gt, sigma=5)
    detail = np.abs(np.diff(gaussian_filter(y_render, sigma=1), axis=1))
    low_grad = np.abs(np.diff(low_render, axis=1))
    chroma = chroma_grad(render)

    raw_y, signed_y, abs_y = [], [], []
    raw_c, signed_c, abs_c = [], [], []
    seam_energy = []
    for center in SEAMS:
        seam, ref = seam_pair(low_grad, rows, width, center)
        raw, signed, absolute = ratio_scores(seam, ref)
        raw_y.append(raw)
        signed_y.append(signed)
        abs_y.append(absolute)
        seam_energy.append(seam)
        seam, ref = seam_pair(chroma, rows, width, center)
        raw, signed, absolute = ratio_scores(seam, ref)
        raw_c.append(raw)
        signed_c.append(signed)
        abs_c.append(absolute)

    py, _ = panel_level(low_render)
    lf_full = float(np.mean(np.abs(low_render - low_gt)))
    lf_thirds = []
    slopes, biases, residuals = [], [], []
    for (x0, x1), (i0, i1) in zip(THIRDS, INTERIORS):
        lf_thirds.append(float(np.mean(np.abs(low_render[:, x0:x1] - low_gt[:, x0:x1]))))
        slope, bias = fit_affine(low_render[:, i0:i1], low_gt[:, i0:i1])
        corrected = slope * low_render[:, i0:i1] + bias
        slopes.append(slope)
        biases.append(bias)
        residuals.append(float(np.mean(np.abs(corrected - low_gt[:, i0:i1]))))
    log_slopes = np.log(np.asarray(slopes))
    bias_n = np.asarray(biases) / 255.0
    mae, psnr = photometric(render, gt)
    plain = np.abs(render[:, 1:].astype(np.int16) - render[:, :-1].astype(np.int16)).mean(axis=2)
    return {
        "psnr": psnr,
        "mae": mae,
        "crcs_like": float(plain.mean()),
        "seam_abs": float(np.mean(seam_energy)),
        "R_Y": float(np.mean(abs_y)),
        "R_Y_signed": float(np.mean(signed_y)),
        "R_Y_raw": float(np.mean(raw_y)),
        "R_C": float(np.mean(abs_c)),
        "R_C_signed": float(np.mean(signed_c)),
        "R_C_raw": float(np.mean(raw_c)),
        "T_detail": panel_detail(detail),
        "P_Y": py,
        "LF_MAE": lf_full,
        "LF_L": lf_thirds[0],
        "LF_C": lf_thirds[1],
        "LF_R": lf_thirds[2],
        "M_corr": float(np.mean(np.abs(log_slopes) + np.abs(bias_n))),
        "std_log_a": float(np.std(log_slopes)),
        "std_b": float(np.std(bias_n)),
        "M_aff": float(np.mean(residuals)),
    }


def collect(render_root, image_dir, pattern, gt_root):
    rows = []
    missing = 0
    for scene in SCENES:
        folder = Path(render_root) / scene / image_dir
        frames = {}
        for path in folder.glob(pattern):
            frames[path.name.split("_")[0]] = path
        gt_frames = {}
        for path in (Path(gt_root) / scene / "rgb").glob("*_2_sparse_wide.png"):
            gt_frames[path.name.split("_")[0]] = path
        for frame in sorted(set(frames) & set(gt_frames), key=int):
            rows.append((scene, frame, frames[frame], gt_frames[frame]))
        missing += len(set(frames) - set(gt_frames))
    return rows, missing


def summarize(records):
    keys = list(records[0][2].keys())
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for _, _, row in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }
    return summary


def fmt(summary, key):
    row = summary[key]
    return f"{row['mean']:.4f}  med {row['median']:.4f}  p90 {row['p90']:.4f}"


def main():
    args = parse_args()
    methods = {
        "UniSplat": collect(args.unisplat_root, "rgb", "*_5_wide.jpg", args.gt_root),
        "DWSplat": collect(args.dwsplat_root, "before_affine_rgb", "*_wide.jpg", args.gt_root),
    }
    by_method = {}
    for name, (jobs, missing) in methods.items():
        print(f"{name}: {len(jobs)} paired frames, render-only frames {missing}", flush=True)
        records = []
        for index, (scene, frame, render_path, gt_path) in enumerate(jobs, start=1):
            row = score_pair(load_rgb(render_path), load_rgb(gt_path))
            records.append((scene, frame, row))
            if index % 50 == 0 or index == len(jobs):
                print(f"  {name} {index}/{len(jobs)}", flush=True)
        by_method[name] = records

    common = set((scene, frame) for scene, frame, _ in by_method["UniSplat"])
    common &= set((scene, frame) for scene, frame, _ in by_method["DWSplat"])
    paired = {}
    for name, records in by_method.items():
        kept = [row for row in records if (row[0], row[1]) in common]
        paired[name] = summarize(kept)
        print(f"{name} common frames {len(kept)}", flush=True)

    gt_detail = []
    # Texture of the GT itself, one value per common frame, from the UniSplat pairing.
    gt_by_frame = {(scene, frame): gt_path for scene, frame, _, gt_path in methods["UniSplat"][0]}
    for scene, frame in sorted(common):
        y_gt = luminance(load_rgb(gt_by_frame[(scene, frame)]))
        gt_detail.append(panel_detail(np.abs(np.diff(gaussian_filter(y_gt, sigma=1), axis=1))))
    gt_detail = np.asarray(gt_detail)

    lines = [
        "Wide consistency demo, same sparse GT",
        f"GT: {args.gt_root}",
        f"Common frames: {len(common)}",
        "GT T_detail mean/median/p90: "
        f"{gt_detail.mean():.4f}  {np.median(gt_detail):.4f}  {np.percentile(gt_detail, 90):.4f}",
        "",
        "No GT. R_Y and R_C are |log((seam+1)/(nearby+1))| on a sigma-5 field.",
        "raw is seam/nearby without the +1 offset. signed > 0 means the seam is harder than nearby texture.",
        "T_detail is a floor, not a score to maximize. P_Y is diagnostic only.",
        "crcs_like and seam_abs are the absolute-gradient family and can reward blur.",
    ]
    order_nogt = [
        "crcs_like", "seam_abs", "R_Y", "R_Y_raw", "R_Y_signed",
        "R_C", "R_C_raw", "R_C_signed", "T_detail", "P_Y",
    ]
    order_gt = ["psnr", "mae", "LF_MAE", "LF_L", "LF_C", "LF_R", "M_corr", "std_log_a", "std_b", "M_aff"]
    for title, keys in (("No GT", order_nogt), ("With GT", order_gt)):
        lines.append("")
        lines.append(f"=== {title} ===")
        for key in keys:
            lines.append(f"{key:12s} Uni {fmt(paired['UniSplat'], key)}")
            lines.append(f"{'':12s} DW  {fmt(paired['DWSplat'], key)}")
    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
