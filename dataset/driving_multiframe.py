"""Consecutive multi-frame windows for nuScenes / Lyft fine-tuning.

A sample is one scene and ``num_frames`` adjacent timesteps (default 3).
Cameras stay in the inference order ``5, 4, 3``. Each frame is packed the same
way as ``scripts/inference_nuscenes_wide_multiframes.py``: resized RGB, scaled
intrinsics, cam2ego rays, and the ego pose repeated as lidar2world.

The newest frame is the supervision target. History frames are only there so
the gaussian head can cache and fuse their Gaussians.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import nuscenes_wide as nw
from dataset.waymo import get_ray_directions, get_rays


def _as_repo_path(path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    repo = Path(__file__).resolve().parents[1]
    return repo / path


def consecutive_windows(frame_ids: Sequence[str], num_frames: int) -> List[List[str]]:
    """Windows of ``num_frames`` ids whose integer values increase by 1."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
    ordered = sorted((int(frame), str(frame)) for frame in frame_ids)
    windows = []
    for end in range(num_frames - 1, len(ordered)):
        chunk = ordered[end - num_frames + 1 : end + 1]
        if all(chunk[i][0] == chunk[0][0] + i for i in range(num_frames)):
            windows.append([frame for _, frame in chunk])
    return windows


def _resize_mask(path: Path, out_wh, sky: bool) -> np.ndarray:
    """Nearest-resize a mask to ``out_wh=(w, h)``.

    Sky pngs are white on sky. The head treats 0 as sky, so those pixels are 0
    and everything else is 1. Dynamic pngs are white on dynamic objects.
    """
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (out_w, out_h):
            mask = mask.resize((out_w, out_h), Image.NEAREST)
        array = np.asarray(mask)
    if sky:
        return (array < 128).astype(np.float32)
    return (array >= 128).astype(np.float32)


class DrivingMultiFrameDataset(Dataset):
    def __init__(
        self,
        data_root,
        scene_list,
        cameras: str = "5,4,3",
        num_frames: int = 3,
        train_long: int = 518,
        max_scenes: Optional[int] = None,
        max_windows: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = _as_repo_path(data_root)
        self.scene_list_path = _as_repo_path(scene_list)
        self.cameras = list(nw.parse_cameras(cameras))
        self.num_frames = int(num_frames)
        self.train_long = int(train_long)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data root not found: {self.data_root}")
        scenes = nw.read_scene_list(self.scene_list_path)
        if max_scenes is not None and max_scenes >= 0:
            scenes = scenes[: int(max_scenes)]
        self.samples = []
        for scene in scenes:
            frames = nw.enumerate_frames(self.data_root, scene, self.cameras)
            for window in consecutive_windows(frames, self.num_frames):
                self.samples.append((scene, window))
                if max_windows is not None and len(self.samples) >= int(max_windows):
                    return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene, window = self.samples[index]
        return self.load_window(scene, window)

    def window_ending(self, scene, frame):
        """The consecutive window whose newest frame is ``frame``."""
        newest = nw.normalize_frame_id(frame)
        frames = nw.enumerate_frames(self.data_root, scene, self.cameras)
        by_int = {int(item): item for item in frames}
        newest_int = int(newest)
        window = []
        for offset in range(self.num_frames - 1, -1, -1):
            frame_int = newest_int - offset
            if frame_int not in by_int:
                raise FileNotFoundError(
                    f"scene {scene} has no {self.num_frames}-frame window ending at {newest}: "
                    f"missing frame {frame_int:03d}"
                )
            window.append(by_int[frame_int])
        return self.load_window(str(scene), window)

    def load_window(self, scene, window: Sequence[str]):
        frames = []
        for frame in window:
            frames.append(self._load_frame(scene, frame))
        out_hw = frames[0]["out_hw"]
        if any(item["out_hw"] != out_hw for item in frames):
            raise ValueError(f"scene {scene} window {window} changed resolution.")

        def stack(key):
            return torch.stack([item[key] for item in frames], dim=0)

        return {
            "scene": str(scene),
            "frames": [item["frame"] for item in frames],
            "out_hw": out_hw,
            "images": stack("images"),
            "rays_o": stack("rays_o"),
            "rays_d": stack("rays_d"),
            "intrinsics": stack("intrinsics"),
            "camera2lidar": stack("camera2lidar"),
            "lidar2world": stack("lidar2world"),
            "sky_mask": stack("sky_mask"),
            "dynamics_region": stack("dynamics_region"),
            "fovx": stack("fovx"),
            "fovy": stack("fovy"),
        }

    def _load_frame(self, scene, frame):
        scene_dir = nw.scene_dir(self.data_root, scene)
        src_wh = None
        for cam in self.cameras:
            path = scene_dir / "images" / f"{frame}_{cam}.jpg"
            if not path.is_file():
                raise FileNotFoundError(f"Missing image {path}")
            with Image.open(path) as image:
                size = image.size
            if src_wh is None:
                src_wh = size
            elif size != src_wh:
                raise ValueError(f"{path} is {size} but camera {self.cameras[0]} is {src_wh}.")
        out_h, out_w = nw.plan_input_hw(src_wh[0], src_wh[1], train_long=self.train_long)
        out_wh = (out_w, out_h)

        images, ks, c2ws, sky, dyn = [], [], [], [], []
        for cam in self.cameras:
            images.append(nw.load_rgb_resized(scene_dir / "images" / f"{frame}_{cam}.jpg", out_wh))
            intrinsic = nw.scale_intrinsics(
                nw.read_intrinsics(scene_dir / "intrinsics" / f"{cam}.txt"), src_wh, out_wh
            )
            ks.append(intrinsic)
            c2ws.append(nw.read_cam2ego(scene_dir / "cam2ego_extrinsics" / f"{cam}.txt"))
            sky_path = scene_dir / "sky_masks" / f"{frame}_{cam}.png"
            if sky_path.is_file():
                sky.append(_resize_mask(sky_path, out_wh, sky=True))
            else:
                sky.append(np.ones((out_h, out_w), dtype=np.float32))
            dyn_path = scene_dir / "fine_dynamic_masks" / "all" / f"{frame}_{cam}.png"
            if dyn_path.is_file():
                dyn.append(_resize_mask(dyn_path, out_wh, sky=False))
            else:
                dyn.append(np.zeros((out_h, out_w), dtype=np.float32))

        images_np = np.stack(images, axis=0)
        images_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).contiguous()
        ks_np = np.stack(ks, axis=0)
        c2ws_np = np.stack(c2ws, axis=0)
        intrinsics = torch.from_numpy(ks_np).float()
        camera2lidar = torch.from_numpy(c2ws_np).float()
        ego = nw.read_matrix(scene_dir / "ego_pose" / f"{frame}.txt", size=4)
        lidar2world = torch.from_numpy(np.repeat(ego[None], len(self.cameras), axis=0)).float()

        directions, fovx, fovy = [], [], []
        for intrinsic in ks_np:
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
            directions.append(
                get_ray_directions(out_h, out_w, focal=[float(fx), float(fy)], principal=[float(cx), float(cy)])
            )
            fovx.append(2.0 * np.arctan(cx / fx))
            fovy.append(2.0 * np.arctan(cy / fy))
        rays_o, rays_d = get_rays(torch.stack(directions, dim=0), camera2lidar, keepdim=True, normalize=False)
        return {
            "frame": str(frame),
            "out_hw": out_wh,
            "images": images_t,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "intrinsics": intrinsics,
            "camera2lidar": camera2lidar,
            "lidar2world": lidar2world,
            "sky_mask": torch.from_numpy(np.stack(sky, axis=0)),
            "dynamics_region": torch.from_numpy(np.stack(dyn, axis=0)),
            "fovx": torch.tensor(fovx, dtype=torch.float32),
            "fovy": torch.tensor(fovy, dtype=torch.float32),
        }


def collate_windows(samples):
    keys = (
        "images",
        "rays_o",
        "rays_d",
        "intrinsics",
        "camera2lidar",
        "lidar2world",
        "sky_mask",
        "dynamics_region",
        "fovx",
        "fovy",
    )
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
    batch["scene"] = [sample["scene"] for sample in samples]
    batch["frames"] = [sample["frames"] for sample in samples]
    batch["out_hw"] = samples[0]["out_hw"]
    return batch
