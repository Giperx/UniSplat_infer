#!/usr/bin/env python3
"""Wide-FOV inference timing for UniSplat.

Single-frame: time one timestamp's Pi3 forward plus the wide render.
Multi-frame: history frames are run once, outside the timer, to fill the
history queue. Every measured iteration restores that queue and times only
the newest frame's input-to-output forward and wide render.

Data loading, checkpoint load, and image saving are not timed.
WARMUP / MEASURE default to 50 and can be overridden by env or CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import nuscenes_wide as nw  # noqa: E402
import inference_nuscenes_wide_multiframes as inf  # noqa: E402


def parse_args(argv=None, default_dataset="nuscenes", multi=False):
    parser = argparse.ArgumentParser(description="UniSplat wide-FOV inference benchmark.")
    parser.add_argument("--dataset", type=str, default=default_dataset, choices=sorted(nw.DATASETS))
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None, help="Newest frame for multi; the frame for single.")
    parser.add_argument("--num-frames", dest="num_frames", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "50")))
    parser.add_argument("--measure", type=int, default=int(os.environ.get("MEASURE", "50")))
    parser.add_argument("--config", type=str, default="configs/waymo.yaml")
    parser.add_argument("--load-from", dest="load_from", type=str, default="pretrained/model.safetensors")
    parser.add_argument("--data-root", dest="data_root", type=str, default=None)
    parser.add_argument("--scene-list", dest="scene_list", type=str, default=None)
    parser.add_argument("--cameras", type=str, default=None)
    parser.add_argument("--render-camera", dest="render_camera", type=int, default=None)
    parser.add_argument("--width-factor", dest="width_factor", type=float, default=3.0)
    parser.add_argument("--car-mask-root", dest="car_mask_root", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--disable-car-mask", dest="disable_car_mask", action="store_true")
    parser.add_argument("--mask-render-view", dest="mask_render_view", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", dest="output_json", type=str, default=None)
    parser.add_argument("--multi", action="store_true", default=multi)
    return parser.parse_args(argv)


def snapshot_queue(queue):
    state = {
        "scenes": None if queue.scenes is None else list(queue.scenes),
        "frames": None if queue.frames is None else list(queue.frames),
    }
    for field in queue.cache_fields:
        data = getattr(queue, field)
        if data is None:
            state[field] = None
            continue
        cloned = []
        for item in data:
            if item is None:
                cloned.append(None)
            elif hasattr(item, "detach"):
                cloned.append(item.detach().clone())
            else:
                cloned.append(item)
        state[field] = cloned
    return state


def restore_queue(queue, state):
    queue.scenes = None if state["scenes"] is None else list(state["scenes"])
    queue.frames = None if state["frames"] is None else list(state["frames"])
    for field in queue.cache_fields:
        data = state[field]
        if data is None:
            setattr(queue, field, None)
            continue
        restored = []
        for item in data:
            if item is None:
                restored.append(None)
            elif hasattr(item, "detach"):
                restored.append(item.detach().clone())
            else:
                restored.append(item)
        setattr(queue, field, restored)


def source_size(sdir, frame, cameras):
    src_wh = None
    for cam in cameras:
        path = sdir / "images" / f"{frame}_{cam}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing image {path}")
        with Image.open(path) as image:
            wh = image.size
        if src_wh is None:
            src_wh = wh
        elif wh != src_wh:
            raise ValueError(f"Camera {cam} image is {wh} but camera {cameras[0]} is {src_wh}.")
    return src_wh


def build_batch(args, sdir, scene, frame, cameras, render_index, out_wh, src_wh, is_newest, device):
    import torch

    packed = inf.prepare_frame(sdir, frame, cameras, out_wh, src_wh)
    out_w, out_h = out_wh
    s = len(cameras)
    mask_cam5 = nw.mask_render_camera(is_newest, args.mask_render_view)
    if args.disable_car_mask or args.mask_kind == "none":
        keep_np = np.ones((s, out_h, out_w), dtype=bool)
    else:
        camera_masks = nw.load_camera_keep_masks(
            args.car_mask_root, cameras, out_wh,
            mask_kind=args.mask_kind, scene=scene, mask_ext=args.mask_ext,
        )
        keep_np = nw.build_car_keep_mask(
            camera_masks, cameras, out_wh, args.render_camera,
            mask_render_view=mask_cam5, disable_car_mask=False,
        )
    if is_newest:
        wide_k, (wide_h, wide_w) = nw.make_wide_intrinsics(
            packed["Ks"][render_index], out_wh, args.width_factor
        )
        fovx, fovy = nw.intrinsics_to_fov(wide_k)
        output = {
            "rgb": torch.zeros((1, 1, 3, wide_h, wide_w), dtype=torch.float32),
            "c2w": torch.from_numpy(packed["c2ws"][render_index]).float()[None, None],
            "fovx": torch.tensor([[fovx]], dtype=torch.float32),
            "fovy": torch.tensor([[fovy]], dtype=torch.float32),
            "intrinsics": torch.from_numpy(wide_k).float()[None, None],
        }
        spatial = (out_h, out_w, wide_h, wide_w)
    else:
        fovx, fovy = inf._fovs(packed["Ks"])
        output = {
            "rgb": torch.zeros((1, s, 3, out_h, out_w), dtype=torch.float32),
            "c2w": torch.from_numpy(packed["c2ws"]).float()[None],
            "fovx": torch.tensor([fovx], dtype=torch.float32),
            "fovy": torch.tensor([fovy], dtype=torch.float32),
            "intrinsics": torch.from_numpy(packed["Ks"]).float()[None],
        }
        spatial = (out_h, out_w, out_h, out_w)
    images = packed["images"].to(device)
    batch = {
        "images": images,
        "input_dict_gs": {
            "rays_o": packed["rays_o"].to(device),
            "rays_d": packed["rays_d"].to(device),
            "intrinsics": torch.from_numpy(packed["Ks"]).float()[None].to(device),
            "camera2lidar": torch.from_numpy(packed["c2ws"]).float()[None].to(device),
            "sky_mask": torch.ones((1, s, out_h, out_w), dtype=images.dtype, device=device),
        },
        "output_dict_gs": {key: value.to(device) for key, value in output.items()},
        "source_keep_mask": torch.from_numpy(keep_np).to(device),
        "lidar2world": torch.from_numpy(packed["lidar2world"]).float()[None].to(device),
        "spatial": spatial,
    }
    return batch


def run_forward(model, batch, scene, frame, amp_dtype):
    import torch
    from torch.amp import autocast

    images = batch["images"]
    with torch.no_grad():
        if amp_dtype is not None:
            with autocast(device_type="cuda", dtype=amp_dtype):
                result = model(images)
        else:
            result = model(images)
        result["scene"] = [scene]
        result["frame"] = [frame]
        result["lidar2world"] = batch["lidar2world"]
        with autocast(device_type="cuda", enabled=False):
            rendered, _ = model.gaussian_head(
                result, images, 5,
                input_dict_gs=batch["input_dict_gs"],
                output_dict_gs=batch["output_dict_gs"],
                source_keep_mask=batch["source_keep_mask"],
            )
    return rendered


def pick_case(args, data_root, cameras):
    scenes = [args.scene] if args.scene else nw.read_scene_list(args.scene_list)
    for scene in scenes:
        frames = nw.enumerate_frames(data_root, scene, cameras, max_frames=None, frame=None)
        if args.multi:
            windows = nw.select_windows(frames, args.num_frames, frame=args.frame)
            windows = [window for window in windows if nw.frames_are_consecutive(window)]
            if windows:
                return scene, windows[0]
        elif frames:
            wanted = frames[0] if args.frame is None else nw.normalize_frame_id(args.frame, frames)
            if wanted in frames:
                return scene, (wanted,)
    raise SystemExit(f"No valid {'window' if args.multi else 'frame'} in {args.scene_list}.")


def summarize(times_ms):
    mean_ms = statistics.mean(times_ms)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "fps": 1000.0 / mean_ms,
    }


def main(argv=None, default_dataset="nuscenes", multi=False):
    import torch

    args = parse_args(argv, default_dataset=default_dataset, multi=multi)
    args.multi = multi or args.multi
    nw.fill_preset_args(args, multi=args.multi)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    inf._preload_torch_libs(torch)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    data_root = inf.resolve_local(args.data_root)
    args.scene_list = inf.resolve_local(args.scene_list)
    args.car_mask_root = inf.resolve_local(args.car_mask_root) if args.car_mask_root else args.car_mask_root
    cameras = list(nw.parse_cameras(args.cameras))
    render_index = cameras.index(int(args.render_camera))
    scene, window = pick_case(args, data_root, cameras)
    sdir = nw.scene_dir(data_root, scene)
    print(f"Loading model from {args.load_from} ...")
    model = inf.load_model(args, device)
    print("Model loaded.")

    batches = []
    for index, frame in enumerate(window):
        src_wh = source_size(sdir, frame, cameras)
        out_h, out_w = nw.plan_input_hw(src_wh[0], src_wh[1])
        batches.append(build_batch(
            args, sdir, scene, frame, cameras, render_index,
            (out_w, out_h), src_wh, index == len(window) - 1, device,
        ))
    timed = batches[-1]
    out_h, out_w, wide_h, wide_w = timed["spatial"]
    queue = model.gaussian_head.history_queue
    queue.scenes = None
    if args.multi and len(batches) > 1:
        print(f"Priming history {','.join(window[:-1])} outside the timer.")
        for frame, batch in zip(window[:-1], batches[:-1]):
            run_forward(model, batch, scene, frame, amp_dtype)
        torch.cuda.synchronize()
        primed = snapshot_queue(queue)
    else:
        primed = None

    def prepare():
        if primed is None:
            queue.scenes = None
        else:
            restore_queue(queue, primed)

    print(
        f"Scene {scene} frame {window[-1]} input={out_h}x{out_w} wide={wide_h}x{wide_w} "
        f"warmup={args.warmup} measure={args.measure}"
    )
    print(f"\n[1/2] Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        prepare()
        run_forward(model, timed, scene, window[-1], amp_dtype)
    torch.cuda.synchronize()

    print(f"[2/2] Measure ({args.measure} iters)...")
    times_ms = []
    for _ in range(args.measure):
        prepare()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_forward(model, timed, scene, window[-1], amp_dtype)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    stats = summarize(times_ms)
    title = f"{args.dataset} {'multi-frame' if args.multi else 'single-frame'} inference"
    scope = "last frame only; history primed outside the timer" if args.multi else "one frame, no history"
    print(f"\n{'=' * 64}")
    print(f" Benchmark Results — {title}")
    print(f"{'=' * 64}")
    print(f"  Scene:        {scene}")
    print(f"  Frame:        {window[-1]}")
    print(f"  Window:       {','.join(window)}")
    print(f"  Timed region: {scope}")
    print(f"  Input:        1 x {len(cameras)} x 3 x {out_h} x {out_w}")
    print(f"  Output:       1 x 3 x {wide_h} x {wide_w}  (cam{args.render_camera} wide-FOV)")
    print(f"  Warmup iters: {args.warmup}")
    print(f"  Measure iters:{args.measure}")
    print(f"{'─' * 64}")
    print(
        f"  Latency (ms): mean={stats['mean_ms']:8.2f}  median={stats['median_ms']:8.2f}  "
        f"min={stats['min_ms']:8.2f}  max={stats['max_ms']:8.2f}  stdev={stats['stdev_ms']:6.2f}"
    )
    print(f"  Throughput:   {stats['fps']:8.2f} FPS")
    print(f"{'=' * 64}")

    payload = {
        "dataset": args.dataset,
        "mode": "multi" if args.multi else "single",
        "scene": scene,
        "frame": window[-1],
        "window": list(window),
        "timed_region": scope,
        "input_hw": [out_h, out_w],
        "wide_hw": [wide_h, wide_w],
        "warmup": args.warmup,
        "measure": args.measure,
        **stats,
    }
    out_path = Path(args.output_json) if args.output_json else (
        REPO_ROOT / "outputs" / "benchmarks" / f"{args.dataset}_{'multi' if args.multi else 'single'}.json"
    )
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return payload


if __name__ == "__main__":
    main()
