"""Score wide renders with CBSR and PD. No ground truth is required.

CBSR is the cross-band seam ratio (lower is better). PD is panel detail
(higher means more local contrast). The ranking number is the mean over
frames. Median and P90 are written too, but they are not the ranking numbers.

``metrics/eval_crcs.py`` and ``metrics/eval_ips.py`` are kept for comparison
and are not called by ``metrics/run_wide.sh``.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import (
    add_common_args,
    collect_renders,
    load_scene_ids,
    load_uint8,
    open_report,
    resolve,
    timestamp,
)
from consistency import score_image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


NAMES = ("cbsr", "pd")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser.parse_args()


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _write_stats(handle, stats, indent=""):
    if stats is None:
        handle.write(f"{indent}no frames\n")
        return
    handle.write(
        f"{indent}n={stats['cbsr']['n']}  "
        f"CBSR={stats['cbsr']['mean']:.4f} (median {stats['cbsr']['median']:.4f}, "
        f"p90 {stats['cbsr']['p90']:.4f})  "
        f"PD={stats['pd']['mean']:.4f} (median {stats['pd']['median']:.4f}, "
        f"p90 {stats['pd']['p90']:.4f})\n"
    )
    handle.write(f"{indent}Rank by the means. CBSR is lower-better. PD is higher-more-detail.\n")


def _bundle(rows):
    if not rows:
        return None
    return {name: _stats([row[name] for row in rows]) for name in NAMES}


def main():
    args = parse_args()
    _preset, render_root, gt_root, val_list = resolve(args)
    if not render_root.is_dir():
        raise SystemExit(f"render root not found: {render_root}")
    scenes, missing_scenes = load_scene_ids(render_root, val_list)
    if not scenes:
        raise SystemExit(f"no rendered scenes under {render_root}")
    jobs = collect_renders(render_root, scenes, args.image_dir)
    if not jobs:
        raise SystemExit(f"no {{frame}}_*_wide images under {render_root}/*/{args.image_dir}")

    global_rows = []
    scene_rows = {}
    for scene, _frame, path in tqdm(jobs, desc="consistency", unit="frame"):
        row = score_image(load_uint8(path))
        global_rows.append(row)
        scene_rows.setdefault(scene, []).append(row)

    out_path = render_root / f"unisplat_consistency_{timestamp()}.txt"
    meta = [
        f"Dataset: {args.dataset}",
        f"Render root: {render_root}",
        f"Val list: {val_list}",
        "No ground truth is used.",
        "CBSR = mean seam column score / (median ordinary-column score + 0.001).",
        "A column score is the absolute cross-band median of a 40px log-luminance step, times sign agreement.",
        "Seams are width/3 and 2*width/3. Ordinary columns are every 12px, excluding 80px around each seam.",
        "PD is the median interior horizontal gradient after a sigma-1 luminance blur.",
        "Rank by the mean. Median and P90 describe the frame distribution and are not ranking numbers.",
        "A lower CBSR with a much lower PD is contrast collapse, not a better seam.",
        "CRCS and IPS are not part of this report.",
        f"Val-list scenes without a render directory: {len(missing_scenes)}.",
        f"Rendered frames: {len(jobs)}.",
        f"GT root is unused: {gt_root}",
    ]
    handle = open_report(out_path, "UniSplat wide CBSR and PD", meta)
    with handle:
        handle.write("\n" + "=" * 80 + "\nSummary\n" + "=" * 80 + "\n")
        _write_stats(handle, _bundle(global_rows))
        handle.write("\n" + "=" * 80 + "\nPer-scene\n" + "=" * 80 + "\n")
        for scene in sorted(scene_rows):
            handle.write(f"\nScene {scene}:\n")
            _write_stats(handle, _bundle(scene_rows[scene]), indent="  ")
        if missing_scenes:
            handle.write("\nScenes in the val list with no render directory:\n")
            for scene in missing_scenes:
                handle.write(f"  {scene}\n")
    print(f"Wrote {out_path}", flush=True)


# Temporarily disabled. Wide measurement does not score CBSR and PD.
# if __name__ == "__main__":
#     main()
