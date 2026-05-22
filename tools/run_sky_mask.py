#!/usr/bin/env python3
# Run sky segmentation on the RGB images and write the masks alongside the
# depth/calib files so UniSplat/dataset/waymo.py can pick them up.
#
# Input layout (per scene):
#   {scene_root}/{scene}/images/{frame:06d}_{cam_minus_1}.png
#
# Output layout:
#   {scene_root}/{scene}/{frame:05d}_{cam}_moge_mask.png   # 255 = non-sky, 0 = sky
#
# Note the two naming conventions co-exist in UniSplat's dataset:
#   - PNG inputs: frame zero-padded to 6 chars, cam 0-indexed
#   - mask / depth / calib outputs: frame zero-padded to 5 chars, cam 1-indexed
#
# Sky segmentation model:
#   `skyseg.onnx` from https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing
#
# Multi-GPU: spawn one worker process per GPU; each scene is processed by a
# single worker (avoids re-loading the ONNX session frame-by-frame).

import os
import sys
import copy
import argparse
import multiprocessing as mp

import cv2
import numpy as np
import onnxruntime


INPUT_SIZE = (320, 320)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
SKY_THRESHOLD = 32


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", required=True, help="root containing per-scene dirs")
    parser.add_argument("--onnx", required=True, help="path to skyseg.onnx")
    parser.add_argument("--gpus", type=str, default="0", help="comma-separated GPU ids")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_skyseg(session, image_bgr):
    resized = cv2.resize(image_bgr, INPUT_SIZE)
    x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]  # (1, 3, H, W)

    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    out = session.run([out_name], {in_name: x.astype(np.float32)})[0]
    out = np.asarray(out).squeeze()
    out = (out - out.min()) / (out.max() - out.min() + 1e-6) * 255.0
    return out.astype(np.uint8)


def segment_one(session, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    raw = run_skyseg(session, image)
    raw = cv2.resize(raw, (image.shape[1], image.shape[0]))
    mask = np.zeros_like(raw)
    mask[raw < SKY_THRESHOLD] = 255  # invert: 255 = non-sky
    return mask


def parse_image_name(fname):
    # 'NNNNNN_C.png' where NNNNNN is 6-digit frame, C is 0..4 (0-indexed cam)
    if not fname.endswith(".png"):
        return None, None
    stem = fname[:-4]
    if "_" not in stem:
        return None, None
    frame_str, cam_str = stem.rsplit("_", 1)
    if len(frame_str) < 5 or not frame_str.isdigit() or cam_str not in "01234":
        return None, None
    return frame_str, cam_str


def output_name(frame_str, cam_zero_indexed):
    # frame: 6-char -> 5-char (strip the leading zero); cam: 0-indexed -> 1-indexed
    return f"{frame_str[1:]}_{int(cam_zero_indexed) + 1}_moge_mask.png"


def worker(gpu_id, scene_chunks, onnx_path, overwrite):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": 0}, {}])

    for scene_dir in scene_chunks:
        images_dir = os.path.join(scene_dir, "images")
        if not os.path.isdir(images_dir):
            continue
        files = sorted(os.listdir(images_dir))
        n_done = 0
        for fname in files:
            frame_str, cam_str = parse_image_name(fname)
            if frame_str is None:
                continue
            out_name = output_name(frame_str, cam_str)
            out_path = os.path.join(scene_dir, out_name)
            if os.path.isfile(out_path) and not overwrite:
                n_done += 1
                continue
            mask = segment_one(session, os.path.join(images_dir, fname))
            if mask is None:
                continue
            cv2.imwrite(out_path, mask)
            n_done += 1
        print(f"[gpu {gpu_id}] {os.path.basename(scene_dir)}: {n_done} files done")


def main():
    args = get_parser().parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    total_workers = len(gpus) * args.workers_per_gpu

    scenes = sorted(d for d in os.listdir(args.scene_root)
                    if os.path.isdir(os.path.join(args.scene_root, d)))
    scenes = [os.path.join(args.scene_root, s) for s in scenes]
    if not scenes:
        print("no scene dir found"); return

    # round-robin scenes across workers
    chunks = [[] for _ in range(total_workers)]
    for i, s in enumerate(scenes):
        chunks[i % total_workers].append(s)

    procs = []
    for wid in range(total_workers):
        gpu_id = gpus[wid // args.workers_per_gpu]
        p = mp.Process(target=worker, args=(gpu_id, chunks[wid], args.onnx, args.overwrite))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print(">> sky mask generation done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
