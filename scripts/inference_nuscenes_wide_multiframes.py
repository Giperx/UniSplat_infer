#!/usr/bin/env python3
"""3-frame nuScenes wide-FOV inference for UniSplat.

UniSplat fuses time through its history queue, one timestamp per forward.
A window is the causal triple ``[t-2, t-1, t]`` (default ``--num-frames 3``).
Each timestamp is encoded from cameras 5, 4, 3 in that frame's ego coordinates
(``cam2ego`` as cam2world). ``ego_pose`` is the ego-to-world pose so the queue
can move the previous frame's static Gaussians into the current ego frame.

Ego-car pixels are removed on every history view, including history cam5, and
on the current frame's cam4/cam3. Current cam5 is kept. Only the newest
frame's cam5 is rendered at 3x width and saved.

Output (kept separate from the single-frame run, same filename pattern)::

    outputs/nuscenes_wide_multiframes/<scene>/rgb/{newest}_{render_cam}_wide.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import nuscenes_wide as nw  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-frame nuScenes wide-FOV inference (UniSplat).")
    parser.add_argument("--config", type=str, default="configs/waymo.yaml")
    parser.add_argument("--load-from", dest="load_from", type=str, default="pretrained/model.safetensors")
    parser.add_argument("--data-root", dest="data_root", type=str, default="data/nuscenes/processed_10Hz/trainval2")
    parser.add_argument("--scene-list", dest="scene_list", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None, help="Render the window whose newest frame is this id.")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=-1)
    parser.add_argument("--num-frames", dest="num_frames", type=int, default=3)
    parser.add_argument("--cameras", type=str, default="5,4,3")
    parser.add_argument("--render-camera", dest="render_camera", type=int, default=5)
    parser.add_argument("--width-factor", dest="width_factor", type=float, default=3.0)
    parser.add_argument(
        "--car-mask-root", dest="car_mask_root", type=str,
        default="data/nuscenes/processed_10Hz/nuscenes_mask",
    )
    parser.add_argument("--mask-render-view", dest="mask_render_view", action="store_true")
    parser.add_argument("--disable-car-mask", dest="disable_car_mask", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="outputs/nuscenes_wide_multiframes")
    parser.add_argument("--save-inputs", dest="save_inputs", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_local(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _preload_torch_libs(torch) -> None:
    import ctypes

    libdir = Path(torch.__file__).resolve().parent / "lib"
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so", "libtorch.so"):
        path = libdir / name
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    nvidia_root = libdir.parent.parent / "nvidia"
    for path in sorted(nvidia_root.glob("cuda_runtime/lib/libcudart.so*")):
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        break


def save_rgb(tensor_chw, path: Path) -> None:
    from PIL import Image

    array = (
        tensor_chw.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).contiguous().numpy()
    )
    array = (array * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=95)


def _fovs(Ks):
    fovx, fovy = [], []
    for K in Ks:
        fx, fy = nw.intrinsics_to_fov(K)
        fovx.append(fx)
        fovy.append(fy)
    return fovx, fovy


def load_model(args, device):
    from omegaconf import OmegaConf
    import torch
    from safetensors.torch import load_file
    from pi3.models.pi3 import Pi3
    import model.gaussian_head as gaussian_head_class

    cfg = OmegaConf.load(resolve_local(args.config))
    model = Pi3()
    model_class = getattr(gaussian_head_class, cfg.Model.Gaussian_head.Name)
    model.gaussian_head = model_class(dim_in=2048, cfg=cfg.Model.Gaussian_head)
    model.gaussian_head.image_backbone.mask_token = None
    weight = load_file(str(resolve_local(args.load_from)), device="cpu")
    model.load_state_dict(weight, strict=False)
    model = model.to(device)
    model.eval()
    return model


def prepare_frame(sdir, frame, cameras, out_wh, src_wh):
    import torch
    from dataset.waymo import get_ray_directions, get_rays

    out_w, out_h = out_wh
    rgb_list = [nw.load_rgb_resized(sdir / "images" / f"{frame}_{cam}.jpg", out_wh) for cam in cameras]
    images_np = np.stack(rgb_list, axis=0)
    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).unsqueeze(0).contiguous()

    Ks, c2ws = [], []
    for cam in cameras:
        Ks.append(nw.scale_intrinsics(nw.read_intrinsics(sdir / "intrinsics" / f"{cam}.txt"), src_wh, out_wh))
        c2ws.append(nw.read_cam2ego(sdir / "cam2ego_extrinsics" / f"{cam}.txt"))
    Ks = np.stack(Ks, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    rays_o, rays_d = [], []
    for view in range(len(cameras)):
        fx, fy, cx, cy = Ks[view][0, 0], Ks[view][1, 1], Ks[view][0, 2], Ks[view][1, 2]
        direction = get_ray_directions(out_h, out_w, focal=[float(fx), float(fy)], principal=[float(cx), float(cy)])
        ro, rd = get_rays(direction, torch.from_numpy(c2ws[view]).float(), keepdim=True, normalize=False)
        rays_o.append(ro)
        rays_d.append(rd)
    ego = nw.read_matrix(sdir / "ego_pose" / f"{frame}.txt", size=4)
    lidar2world = np.repeat(ego[None], len(cameras), axis=0)
    return {
        "images": images,
        "rgb_list": rgb_list,
        "Ks": Ks,
        "c2ws": c2ws,
        "rays_o": torch.stack(rays_o, dim=0)[None],
        "rays_d": torch.stack(rays_d, dim=0)[None],
        "lidar2world": lidar2world,
    }


def main():
    args = parse_args()
    if args.num_frames < 1:
        raise SystemExit("--num-frames must be >= 1.")

    import torch
    from torch.amp import autocast
    from PIL import Image

    _preload_torch_libs(torch)
    data_root = resolve_local(args.data_root)
    scene_list = resolve_local(args.scene_list) if args.scene_list else (data_root / "nuScenes_Val2.txt")
    mask_root = resolve_local(args.car_mask_root)
    output_dir = resolve_local(args.output_dir)

    cameras = list(nw.parse_cameras(args.cameras))
    render_camera = int(args.render_camera)
    if render_camera not in cameras:
        raise SystemExit(f"--render-camera {render_camera} is not among --cameras {cameras}.")
    render_index = cameras.index(render_camera)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = load_model(args, device)
    scenes = [args.scene] if args.scene else nw.read_scene_list(scene_list)
    if not scenes:
        raise SystemExit(f"No scenes to process (scene list: {scene_list}).")

    for scene in scenes:
        frames = nw.enumerate_frames(data_root, scene, cameras, args.max_frames, frame=None)
        windows = nw.select_windows(frames, args.num_frames, frame=args.frame)
        if not windows:
            print(
                f"[warn] scene {scene}: fewer than --num-frames {args.num_frames} valid frames; skipping.",
                file=sys.stderr,
            )
            continue
        sdir = nw.scene_dir(data_root, scene)

        for window in windows:
            if not nw.frames_are_consecutive(window):
                print(
                    f"[warn] scene {scene} window {','.join(window)} is not consecutive; "
                    "history fusion will not link the gap.",
                    file=sys.stderr,
                )
            model.gaussian_head.history_queue.scenes = None
            newest = window[-1]
            saved = None
            history_dropped = 0
            current_dropped = 0

            for frame in window:
                is_newest = frame == newest
                src_wh = None
                for cam in cameras:
                    img_path = sdir / "images" / f"{frame}_{cam}.jpg"
                    if not img_path.is_file():
                        raise FileNotFoundError(f"Missing image {img_path}")
                    with Image.open(img_path) as im:
                        wh = im.size
                    if src_wh is None:
                        src_wh = wh
                    elif wh != src_wh:
                        raise ValueError(f"Camera {cam} image is {wh} but camera {cameras[0]} is {src_wh}.")
                out_h, out_w = nw.plan_input_hw(src_wh[0], src_wh[1])
                out_wh = (out_w, out_h)
                packed = prepare_frame(sdir, frame, cameras, out_wh, src_wh)
                S = len(cameras)

                mask_cam5 = nw.mask_render_camera(is_newest, args.mask_render_view)
                if args.disable_car_mask:
                    keep_np = np.ones((S, out_h, out_w), dtype=bool)
                else:
                    camera_masks = nw.load_camera_keep_masks(mask_root, cameras, out_wh)
                    keep_np = nw.build_car_keep_mask(
                        camera_masks, cameras, out_wh, render_camera,
                        mask_render_view=mask_cam5, disable_car_mask=False,
                    )
                dropped = int((~keep_np).sum())
                if is_newest:
                    current_dropped = dropped
                else:
                    history_dropped += dropped

                if is_newest:
                    wide_K, (wide_h, wide_w) = nw.make_wide_intrinsics(
                        packed["Ks"][render_index], out_wh, args.width_factor
                    )
                    fovx, fovy = nw.intrinsics_to_fov(wide_K)
                    output_dict_gs = {
                        "rgb": torch.zeros((1, 1, 3, wide_h, wide_w), dtype=torch.float32),
                        "c2w": torch.from_numpy(packed["c2ws"][render_index]).float()[None, None],
                        "fovx": torch.tensor([[fovx]], dtype=torch.float32),
                        "fovy": torch.tensor([[fovy]], dtype=torch.float32),
                        "intrinsics": torch.from_numpy(wide_K).float()[None, None],
                    }
                else:
                    fovx, fovy = _fovs(packed["Ks"])
                    output_dict_gs = {
                        "rgb": torch.zeros((1, S, 3, out_h, out_w), dtype=torch.float32),
                        "c2w": torch.from_numpy(packed["c2ws"]).float()[None],
                        "fovx": torch.tensor([fovx], dtype=torch.float32),
                        "fovy": torch.tensor([fovy], dtype=torch.float32),
                        "intrinsics": torch.from_numpy(packed["Ks"]).float()[None],
                    }

                images = packed["images"].to(device)
                input_dict_gs = {
                    "rays_o": packed["rays_o"].to(device),
                    "rays_d": packed["rays_d"].to(device),
                    "intrinsics": torch.from_numpy(packed["Ks"]).float()[None].to(device),
                    "camera2lidar": torch.from_numpy(packed["c2ws"]).float()[None].to(device),
                    "sky_mask": torch.ones((1, S, out_h, out_w), dtype=images.dtype, device=device),
                }
                for key in output_dict_gs:
                    output_dict_gs[key] = output_dict_gs[key].to(device)
                source_keep_mask = torch.from_numpy(keep_np).to(device)
                lidar2world = torch.from_numpy(packed["lidar2world"]).float()[None].to(device)

                with torch.no_grad():
                    if amp_dtype is not None:
                        with autocast(device_type="cuda", dtype=amp_dtype):
                            res = model(images)
                    else:
                        res = model(images)
                    res["scene"] = [scene]
                    res["frame"] = [frame]
                    res["lidar2world"] = lidar2world
                    with autocast(device_type="cuda", enabled=False):
                        render_pkg, _ = model.gaussian_head(
                            res, images, 5,
                            input_dict_gs=input_dict_gs,
                            output_dict_gs=output_dict_gs,
                            source_keep_mask=source_keep_mask,
                        )
                if is_newest:
                    save_path = output_dir / scene / "rgb" / f"{frame}_{render_camera}_wide.jpg"
                    save_rgb(render_pkg["image"][0], save_path)
                    saved = (out_h, out_w, wide_h, wide_w, save_path)
                    if args.save_inputs:
                        for view, cam in enumerate(cameras):
                            save_rgb(
                                torch.from_numpy(packed["rgb_list"][view]).permute(2, 0, 1),
                                output_dir / scene / "inputs" / f"{frame}_{cam}.jpg",
                            )

            print(
                f"scene={scene} window={','.join(window)} render={newest} "
                f"input={saved[0]}x{saved[1]} wide={saved[2]}x{saved[3]} "
                f"dropped_current={current_dropped} dropped_history={history_dropped}",
                flush=True,
            )
            if not args.disable_car_mask and history_dropped == 0:
                print(
                    f"[warn] scene={scene} window={','.join(window)}: history ego masks zeroed 0 pixel-gaussians.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
