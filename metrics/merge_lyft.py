#!/usr/bin/env python3
"""Lyft 1920+1224 totals, and equal-weight Left/Right means for old reports.

1920 and 1224 are one dataset split in two. The total under
``outputs/lyft1920_wide*`` is the frame-count-weighted mean

    (value_1920 * n_1920 + value_1224 * n_1224) / (n_1920 + n_1224)

not the average of the two subset means.

Sparse photometric reports also get Mean_LR = (Left+Right)/2 and
Mean_LRC = (Left+Right+Center)/3 of the reported region means. WideDrive
is skipped because it already scores the full image.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from common import timestamp

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
DECIMALS = {
    "PSNR": 2,
    "MAE": 2,
    "RMSE": 2,
    "SSIM": 4,
    "LPIPS": 4,
    "VALUE": 4,
    "CBSR": 4,
    "PD": 4,
}
PHOTOMETRIC_ORDER = ("PSNR", "MAE", "RMSE", "SSIM", "LPIPS")
MEAN_NOTE = (
    "Mean_LR is (Left+Right)/2 and Mean_LRC is (Left+Right+Center)/3 of those region means. "
    "Equal weight on the reported unmasked region scores, not pixel-weighted Overall, and not Center_masked."
)
METRIC_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<key>Left|Center|Center_masked|Right|Overall)\s*:\s*n=(?P<n>\d+)\s+(?P<rest>.*)$"
)
SUMMARY_LINE = re.compile(
    r"^(?P<key>\S+)\s*:\s*n=(?P<n>\d+)\s+(?P<rest>.*)$"
)
CONSISTENCY_LINE = re.compile(
    r"^n=(?P<n>\d+)\s+CBSR=(?P<cbsr>[0-9.]+).*?\bPD=(?P<pd>[0-9.]+)"
)
REPORT_KINDS = (
    "photometric",
    "HM",
    # "consistency",  # CBSR and PD are paused and not part of measurement.
    "CRCS",
    "IPS",
)


def _metrics(text):
    return {
        name: float(value)
        for name, value in re.findall(r"([A-Z][A-Z0-9_]*)=([+-]?\d+(?:\.\d+)?)", text)
    }


def _format_row(key, count, metrics, order, indent=""):
    parts = [f"n={count}"]
    used = set()
    for name in order:
        if name not in metrics:
            continue
        parts.append(f"{name}={metrics[name]:.{DECIMALS.get(name, 4)}f}")
        used.add(name)
    for name in metrics:
        if name in used:
            continue
        parts.append(f"{name}={metrics[name]:.{DECIMALS.get(name, 4)}f}")
    return f"{indent}{key:20s}: {'  '.join(parts)}\n"


def _equal_mean_lines(rows, indent=""):
    """rows: key -> (n, metrics). Means are taken after any pooling."""
    by_key = rows
    lines = []
    groups = (("Mean_LR", ("Left", "Right")), ("Mean_LRC", ("Left", "Right", "Center")))
    for label, regions in groups:
        if any(region not in by_key for region in regions):
            continue
        shared = set(by_key[regions[0]][1])
        for region in regions[1:]:
            shared &= set(by_key[region][1])
        order = [name for name in PHOTOMETRIC_ORDER if name in shared]
        if not order:
            continue
        counts = [by_key[region][0] for region in regions]
        metrics = {}
        for name in order:
            metrics[name] = sum(by_key[region][1][name] for region in regions) / len(regions)
        count = counts[0] if len(set(counts)) == 1 else sum(counts) // len(counts)
        lines.append(_format_row(label, count, metrics, order, indent))
    return lines


def supplement_text(text):
    """Insert Mean_LR and Mean_LRC into one sparse photometric/HM report."""
    if "Mean_LR" in text or "Dataset: widedrive" in text or "Style: dense" in text:
        return text, False
    if not re.search(r"^Left\s*:", text, re.M):
        return text, False

    lines = text.splitlines(keepends=True)
    out = []
    block = []
    note_added = False

    def flush():
        nonlocal block
        if not block:
            return
        by_key = {item["key"]: item for item in block}
        for item in block:
            out.append(item["line"])
        if {"Left", "Center", "Right"} <= set(by_key):
            packed = {key: (item["n"], item["metrics"]) for key, item in by_key.items()}
            for extra in _equal_mean_lines(packed, indent=block[0]["indent"]):
                out.append(extra)
        block = []

    for line in lines:
        match = METRIC_LINE.match(line.rstrip("\n"))
        if match:
            raw = line if line.endswith("\n") else line + "\n"
            block.append({
                "key": match.group("key"),
                "n": int(match.group("n")),
                "metrics": _metrics(match.group("rest")),
                "indent": match.group("indent"),
                "line": raw,
            })
            continue
        flush()
        raw = line if line.endswith("\n") else line + "\n"
        if not note_added and raw.startswith("Scored frames:"):
            out.append(raw)
            out.append(MEAN_NOTE + "\n")
            note_added = True
            continue
        out.append(raw)
    flush()
    if not note_added:
        return text, False
    new = "".join(out)
    return new, new != text


def backfill_means(outputs):
    changed = []
    for kind in ("photometric", "HM"):
        for path in sorted(outputs.glob(f"*/unisplat_{kind}_*.txt")):
            if "widedrive" in path.parent.name or "lyft_combined" in path.name:
                continue
            text = path.read_text()
            new, did = supplement_text(text)
            if not did:
                continue
            path.write_text(new)
            changed.append(path)
    return changed


def latest_report(directory, kind):
    if not directory.is_dir():
        return None
    files = [
        path for path in directory.glob(f"unisplat_{kind}_*.txt")
        if "lyft_combined" not in path.name
    ]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def parse_summary(path):
    rows = []
    in_summary = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "Summary":
            in_summary = True
            continue
        if in_summary and stripped == "Per-scene":
            break
        if not in_summary:
            continue
        consistency = CONSISTENCY_LINE.match(stripped)
        if consistency:
            rows.append((
                "consistency",
                int(consistency.group("n")),
                {"CBSR": float(consistency.group("cbsr")), "PD": float(consistency.group("pd"))},
            ))
            continue
        match = SUMMARY_LINE.match(stripped)
        if not match or match.group("key") in {"Mean_LR", "Mean_LRC"}:
            continue
        rows.append((match.group("key"), int(match.group("n")), _metrics(match.group("rest"))))
    return rows


def pool_rows(left_rows, right_rows):
    right = {key: (count, metrics) for key, count, metrics in right_rows}
    pooled = []
    for key, count_a, metrics_a in left_rows:
        if key not in right:
            continue
        count_b, metrics_b = right[key]
        total = count_a + count_b
        if total <= 0:
            continue
        names = [name for name in metrics_a if name in metrics_b]
        metrics = {
            name: (metrics_a[name] * count_a + metrics_b[name] * count_b) / total
            for name in names
        }
        pooled.append((key, total, metrics))
    return pooled


def _order_for(kind, metrics):
    if kind in {"photometric", "HM"}:
        return PHOTOMETRIC_ORDER
    if kind == "consistency":
        return ("CBSR", "PD")
    if "VALUE" in metrics:
        return ("VALUE",)
    return tuple(metrics)


def write_combined(mode):
    suffix = "wide" if mode == "single" else "wide_multiframes"
    dir_1920 = OUTPUTS / f"lyft1920_{suffix}"
    dir_1224 = OUTPUTS / f"lyft1224_{suffix}"
    sections = []
    for kind in REPORT_KINDS:
        path_1920 = latest_report(dir_1920, kind)
        path_1224 = latest_report(dir_1224, kind)
        if path_1920 is None or path_1224 is None:
            print(f"[lyft] skip {mode} {kind}: missing 1920 or 1224 report")
            continue
        pooled = pool_rows(parse_summary(path_1920), parse_summary(path_1224))
        if not pooled:
            print(f"[lyft] skip {mode} {kind}: no shared summary rows")
            continue
        sections.append((kind, path_1920, path_1224, pooled))
    if not sections:
        print(f"[lyft] no combined file for {mode}; one subset has no reports yet")
        return None

    stamp = timestamp()
    out_path = dir_1920 / f"unisplat_lyft_combined_{stamp}.txt"
    lines = [
        "=== UniSplat Lyft combined (1920 + 1224) ===\n",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Mode: {mode}\n",
        "1920 and 1224 are one dataset. Each value is the frame-count-weighted mean\n",
        "(value_1920 * n_1920 + value_1224 * n_1224) / (n_1920 + n_1224),\n",
        "not the average of the two subset means.\n",
        "Mean_LR and Mean_LRC are then (Left+Right)/2 and (Left+Right+Center)/3 of the pooled region means.\n",
        "WideDrive is not included. Per-scene numbers stay in the subset reports.\n",
        "\n",
    ]
    for kind, path_1920, path_1224, pooled in sections:
        lines.append("=" * 80 + "\n")
        lines.append(f"{kind}\n")
        lines.append("=" * 80 + "\n")
        lines.append(f"1920: {path_1920.relative_to(REPO_ROOT)}\n")
        lines.append(f"1224: {path_1224.relative_to(REPO_ROOT)}\n")
        for key, count, metrics in pooled:
            lines.append(_format_row(key, count, metrics, _order_for(kind, metrics)))
        if kind in {"photometric", "HM"}:
            packed = {key: (count, metrics) for key, count, metrics in pooled}
            lines.extend(_equal_mean_lines(packed))
        lines.append("\n")
    out_path.write_text("".join(lines))
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multiframes", "both"), default="both")
    parser.add_argument(
        "--backfill-means",
        action="store_true",
        help="Insert Mean_LR and Mean_LRC into existing sparse photometric and HM reports.",
    )
    args = parser.parse_args()
    if args.backfill_means:
        changed = backfill_means(OUTPUTS)
        print(f"Updated {len(changed)} report(s)")
        for path in changed:
            print(f"  {path.relative_to(REPO_ROOT)}")
    modes = ("single", "multiframes") if args.mode == "both" else (args.mode,)
    for mode in modes:
        write_combined(mode)


if __name__ == "__main__":
    main()
