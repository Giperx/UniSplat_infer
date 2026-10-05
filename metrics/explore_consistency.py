"""Probe no-GT consistency scores on a known-good / known-bad pair.

Controls, built from the DWSplat render:
  blur     Gaussian sigma=3, which should not be rewarded
  seam     left third x0.55 and right third x1.45, which must be punished

Candidates that still rank the blurred image as the best seam are rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

UNI = Path("outputs/widedrive_expusre_demo_wide")
DW = Path("/home/xiongzhp/FeedForward/2027cvpr/DWSplat_cvpr/renders_save/widedrive/maincam_cam2_wide/exposure_demo_epoch_24-step_20000_safe")
SCENES = ("Town10HD_scene_0009", "Town10HD_scene_0019")
SEAMS = (518, 1036)
WIDTH = 1554


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def frames():
    found = []
    for scene in SCENES:
        for frame in range(2, 200, 8):
            name = f"{frame:03d}"
            uni = UNI / scene / "rgb" / f"{name}_5_wide.jpg"
            dw = DW / scene / "before_affine_rgb" / f"{name}_wide.jpg"
            if uni.is_file() and dw.is_file():
                found.append((uni, dw))
    return found


def blur_image(rgb):
    out = np.empty_like(rgb)
    for channel in range(3):
        out[..., channel] = gaussian_filter(rgb[..., channel], sigma=3)
    return out


def seam_image(rgb):
    out = rgb.copy()
    out[:, :518] *= 0.55
    out[:, 1036:] *= 1.45
    return np.clip(out, 0, 255)


def crcs(rgb):
    diff = np.abs(rgb[:, 1:] - rgb[:, :-1]).mean(axis=2)
    return float(diff.mean())


def detail(y):
    grad = np.abs(np.diff(gaussian_filter(y, sigma=1), axis=1))
    spans = ((0, 454), (582, 972), (1100, 1554))
    scores = [float(np.median(grad[:, a:b - 1].mean(axis=1))) for a, b in spans]
    return float(np.median(scores))


def band_jumps(field, x, half=40, gap=8):
    """Signed right-minus-left level in each horizontal band."""
    height = field.shape[0]
    bands = 6
    band_h = height // bands
    jumps = []
    for band in range(bands):
        rows = slice(band * band_h, (band + 1) * band_h)
        left = field[rows, x - gap - half:x - gap].mean()
        right = field[rows, x + gap:x + gap + half].mean()
        jumps.append(right - left)
    return np.asarray(jumps)


def reference_xs():
    blocked = []
    for seam in SEAMS:
        blocked.append((seam - 80, seam + 80))
    xs = []
    for x in range(80, WIDTH - 80, 12):
        if any(lo <= x <= hi for lo, hi in blocked):
            continue
        xs.append(x)
    return xs


def coherence(field):
    """Median across bands, then how large the seam is versus other columns.

    A pole or car moves one band. An exposure change moves every band the
    same way, so the cross-band median stays large only at a real seam.
    """
    xs = reference_xs()

    def score_at(x):
        jumps = band_jumps(field, x)
        med = float(np.median(jumps))
        agree = float(np.mean(np.sign(jumps) == np.sign(med if med != 0 else 1)))
        return abs(med) * agree

    seam = float(np.mean([score_at(x) for x in SEAMS]))
    base = np.asarray([score_at(x) for x in xs])
    baseline = float(np.median(base))
    return seam, baseline, seam - baseline, seam / (baseline + 1e-3)


def profile_step(field):
    """One vertical-median profile. Compare the seam step with other columns."""
    profile = np.median(field, axis=0)

    def step(x, half=40, gap=8):
        left = profile[x - gap - half:x - gap].mean()
        right = profile[x + gap:x + gap + half].mean()
        return abs(float(right - left))

    seam = float(np.mean([step(x) for x in SEAMS]))
    base = np.asarray([step(x) for x in reference_xs()])
    return seam, float(np.median(base)), seam - float(np.median(base)), seam / (float(np.median(base)) + 1e-3)


def detrended_step(field):
    """Quadratic fit away from the seams. The leftover step is the seam."""
    profile = np.median(gaussian_filter(field, sigma=(0, 8)), axis=0)
    xs = np.arange(profile.size, dtype=np.float64)
    keep = np.ones(profile.size, dtype=bool)
    for seam in SEAMS:
        keep[seam - 50:seam + 50] = False
    coeff = np.polyfit(xs[keep], profile[keep], deg=2)
    resid = profile - np.polyval(coeff, xs)

    def step(x):
        return abs(float(resid[x + 8:x + 40].mean() - resid[x - 40:x - 8].mean()))

    seam = float(np.mean([step(x) for x in SEAMS]))
    base = np.asarray([step(x) for x in reference_xs()])
    return seam, float(np.median(base)), seam - float(np.median(base))


def raw_ratio(y):
    low = gaussian_filter(y, sigma=5)
    grad = np.abs(np.diff(low, axis=1))
    rows = slice(74, 220)

    def cols(center, near, far):
        xs = np.arange(grad.shape[1])
        dist = np.abs(xs - center)
        return (dist >= near) & (dist <= far)

    ratios = []
    for center in SEAMS:
        seam = float(np.median(grad[rows][:, cols(center, 0, 8)].mean(axis=1)))
        ref = float(np.median(grad[rows][:, cols(center, 24, 64)].mean(axis=1)))
        ratios.append(seam / (ref + 1e-3))
    return float(np.mean(ratios))


def pack(rgb):
    y = luminance(rgb)
    low = gaussian_filter(y, sigma=5)
    log_y = np.log(y + 1.0)
    coh_seam, coh_base, coh_excess, coh_ratio = coherence(low)
    log_seam, log_base, log_excess, log_ratio = coherence(log_y)
    prof_seam, prof_base, prof_excess, prof_ratio = profile_step(low)
    det_seam, det_base, det_excess = detrended_step(y)
    return {
        "crcs": crcs(rgb),
        "detail": detail(y),
        "grad_ratio": raw_ratio(y),
        "band_seam": coh_seam,
        "band_base": coh_base,
        "band_excess": coh_excess,
        "band_ratio": coh_ratio,
        "log_excess": log_excess,
        "log_ratio": log_ratio,
        "profile_excess": prof_excess,
        "profile_ratio": prof_ratio,
        "detrend_excess": det_excess,
        "detrend_seam": det_seam,
        "detrend_base": det_base,
    }


def mean_of(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main():
    pairs = frames()
    print(f"frames {len(pairs)}", flush=True)
    buckets = {"UniSplat": [], "DWSplat": [], "DW_blur": [], "DW_seam": []}
    for index, (uni_path, dw_path) in enumerate(pairs, start=1):
        uni = load_rgb(uni_path)
        dw = load_rgb(dw_path)
        buckets["UniSplat"].append(pack(uni))
        buckets["DWSplat"].append(pack(dw))
        buckets["DW_blur"].append(pack(blur_image(dw)))
        buckets["DW_seam"].append(pack(seam_image(dw)))
        if index % 10 == 0 or index == len(pairs):
            print(f"  {index}/{len(pairs)}", flush=True)
    summary = {name: mean_of(rows) for name, rows in buckets.items()}
    keys = list(summary["DWSplat"])
    print(f"{'metric':16s} {'Uni':>10s} {'DW':>10s} {'blur':>10s} {'seam':>10s}")
    for key in keys:
        print(
            f"{key:16s} {summary['UniSplat'][key]:10.4f} {summary['DWSplat'][key]:10.4f} "
            f"{summary['DW_blur'][key]:10.4f} {summary['DW_seam'][key]:10.4f}"
        )


if __name__ == "__main__":
    main()
