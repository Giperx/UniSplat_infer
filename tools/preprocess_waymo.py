#!/usr/bin/env python3
# Preprocess Waymo Perception v1.4.x tfrecords into the per-frame layout that
# UniSplat/dataset/waymo.py expects:
#
#   {output_dir}/{scene}/
#     ├── {frame:05d}_{cam}.exr       # sparse depth from LiDAR projection
#     └── {frame:05d}_{cam}.npz       # intrinsics / cam2world / cam2lidar / distortion
#

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import os.path as osp
import sys
import shutil
import json
import argparse
import numpy as np
import PIL.Image
from tqdm import tqdm
import cv2

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo_dir", required=True, help="dir containing *.tfrecord")
    parser.add_argument("--output_dir", required=True, help="output scene_root")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--keep_tmp", action="store_true",
                        help="keep the intermediate tmp/ dir (jpg + raw npz + calib.json)")
    return parser


def inv(mat):
    return np.linalg.inv(mat)


def geotrf(Trf, pts):
    """Apply a 4x4 transform to (N, 3) points."""
    pts = np.asarray(pts)
    h = np.concatenate([pts, np.ones_like(pts[:, :1])], axis=-1)  # (N, 4)
    return (h @ Trf.T)[:, :3]


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    return cv2.imread(path, options)


def _list_sequences(db_root):
    res = sorted(f for f in os.listdir(db_root) if f.endswith(".tfrecord"))
    print(f">> found {len(res)} sequences under {db_root}")
    return res


# --------- stage 1: tfrecord -> tmp/{scene}/{frame:05d}_{cam}.jpg + .npz + calib.json ---------

def extract_frames_one_seq(filename):
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils

    dataset = tf.data.TFRecordDataset(filename, compression_type="")
    calib = None
    frames = []

    for data in tqdm(dataset, leave=False, desc=osp.basename(filename)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        content = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _, range_image_top_pose = content

        views = {}
        frames.append((frame.context.name, views))

        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append((cam.name, dict(
                    width=cam.width, height=cam.height,
                    intrinsics=list(cam.intrinsic),
                    extrinsics=list(cam.extrinsic.transform))))

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose)
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        for image in frame.images:
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            cp_msk = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)),
                             dtype=tf.float32).numpy()
            pose = np.asarray(image.pose.transform).reshape(4, 4)
            rgb = tf.image.decode_jpeg(image.image).numpy()
            pix = cp_msk[..., 1:3].round().astype(np.int16)
            pts3d = points_all[mask.numpy()]
            views[image.name] = dict(img=rgb, pose=pose, pixels=pix, pts3d=pts3d)

    return calib, frames


def process_one_seq(db_root, tmp_dir, seq):
    out_dir = osp.join(tmp_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    calib_path = osp.join(out_dir, "calib.json")
    if osp.isfile(calib_path):
        return

    try:
        with tf.device("/CPU:0"):
            calib, frames = extract_frames_one_seq(osp.join(db_root, seq))
    except RuntimeError:
        print(f"/!\\ failed to extract {seq} /!\\", file=sys.stderr)
        return

    for f, (_, views) in enumerate(frames):
        for cam_idx, view in views.items():
            img = PIL.Image.fromarray(view.pop("img"))
            img.save(osp.join(out_dir, f"{f:05d}_{cam_idx}.jpg"))
            np.savez(osp.join(out_dir, f"{f:05d}_{cam_idx}.npz"), **view)
    with open(calib_path, "w") as f:
        json.dump(calib, f)


# --------- stage 2: tmp -> output_dir/{scene}/{frame:05d}_{cam}.exr + .npz ---------

def crop_one_seq(tmp_dir, output_dir, seq):
    seq_dir = osp.join(tmp_dir, seq)
    out_dir = osp.join(output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)

    try:
        with open(osp.join(seq_dir, "calib.json")) as f:
            calib = json.load(f)
    except IOError:
        print(f"/!\\ Error: Missing calib.json in sequence {seq} /!\\", file=sys.stderr)
        return

    axes_transformation = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    )

    cam_K = {}
    cam_distortion = {}
    cam_res = {}
    cam_to_car = {}
    for cam_idx, cam_info in calib:
        cam_idx = str(cam_idx)
        cam_res[cam_idx] = (W, H) = (cam_info["width"], cam_info["height"])
        f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info["intrinsics"]
        cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
        cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
        cam_to_car[cam_idx] = np.asarray(cam_info["extrinsics"]).reshape(4, 4)  # cam-to-vehicle

    frames = sorted(f[:-3] for f in os.listdir(seq_dir) if f.endswith(".jpg"))

    for frame in tqdm(frames, leave=False, desc=seq):
        cam_idx = frame[-2]  # cam index (last char before the trailing dot)
        assert cam_idx in "12345", f"bad {cam_idx=} in {frame=}"
        data = np.load(osp.join(seq_dir, frame + "npz"))
        car_to_world = data["pose"]
        W, H = cam_res[cam_idx]

        # load depthmap
        pos2d = data["pixels"].round().astype(np.uint16)
        pts3d = data["pts3d"]  # already in the car frame
        pts3d = geotrf(axes_transformation @ inv(cam_to_car[cam_idx]), pts3d)
        # X=LEFT_RIGHT y=ALTITUDE z=DEPTH

        # use image shape to catch corrupted data
        image = imread_cv2(osp.join(seq_dir, frame + "jpg"))
        H, W = image.shape[:2]
        if cam_idx in "123":
            assert W == 1920 and H == 1280, f"bad {H=} {W=} in {frame=}"
        elif cam_idx in "45":
            assert W == 1920 and H == 886, f"bad {H=} {W=} in {frame=}"

        depthmap = np.zeros((H, W), dtype=np.float32)
        x, y = pos2d.T
        depthmap[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = pts3d[:, 2]
        cv2.imwrite(osp.join(out_dir, frame + "exr"), depthmap)

        # save camera parameters
        cam2world = car_to_world @ cam_to_car[cam_idx] @ inv(axes_transformation)
        cam2lidar = cam_to_car[cam_idx] @ inv(axes_transformation)
        np.savez(
            osp.join(out_dir, frame + "npz"),
            intrinsics=cam_K[cam_idx],
            cam2world=cam2world,
            distortion=cam_distortion[cam_idx],
            cam2lidar=cam2lidar,
        )


def main():
    args = get_parser().parse_args()
    tmp_dir = osp.join(args.output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    sequences = _list_sequences(args.waymo_dir)[args.start:args.end]
    print(f">> processing {len(sequences)} sequences")

    # stage 1 (heavy: tfrecord decode + LiDAR projection)
    from multiprocessing import Pool
    jobs = [(args.waymo_dir, tmp_dir, seq) for seq in sequences]
    if args.workers <= 1:
        for j in jobs:
            process_one_seq(*j)
    else:
        with Pool(args.workers) as pool:
            for _ in tqdm(pool.imap_unordered(_proc1_star, jobs),
                          total=len(jobs), desc="extract"):
                pass

    # stage 2 (cheap: per-frame depth + calib write)
    jobs = [(tmp_dir, args.output_dir, seq) for seq in sequences]
    if args.workers <= 1:
        for j in jobs:
            crop_one_seq(*j)
    else:
        with Pool(args.workers) as pool:
            for _ in tqdm(pool.imap_unordered(_proc2_star, jobs),
                          total=len(jobs), desc="finalize"):
                pass

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f">> done. data at {args.output_dir}")


# pickleable wrappers for multiprocessing
def _proc1_star(args): return process_one_seq(*args)
def _proc2_star(args): return crop_one_seq(*args)


if __name__ == "__main__":
    main()
