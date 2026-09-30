#!/usr/bin/env python3
"""Single-frame nuScenes wide-FOV inference for UniSplat.

Builds the scene Gaussians from cameras ``5`` (CAM_BACK), ``4``
(CAM_BACK_RIGHT) and ``3`` (CAM_BACK_LEFT), drops the ego-car Gaussians from
cameras 4 and 3 by zeroing their pixel-Gaussian opacity, then renders the main
camera (5) at ``width_factor`` x width and the same height.

The network input is aspect-scaled with **no centre crop**: the long side
matches the training long side (518) and the short side is snapped to a multiple
of the DINOv2 patch (14).  For 1600x900 this gives ``294x518``.  The wide render
keeps the resized camera-5 focal length and ``cy`` and only moves ``cx`` to the
centre of the wide canvas.

Output layout (matching DepthSplat)::

    <output-dir>/<scene>/rgb/{frame}_{render_cam}_wide.jpg   # quality 95

with optional ``<output-dir>/<scene>/inputs/{frame}_{cam}.jpg``.
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
    parser = argparse.ArgumentParser(description="Single-frame nuScenes wide-FOV inference (UniSplat).")
    parser.add_argument("--config", type=str, default="configs/waymo.yaml")
    parser.add_argument("--load-from", dest="load_from", type=str, default="pretrained/model.safetensors")
    parser.add_argument("--data-root", dest="data_root", type=str, default="data/nuscenes/processed_10Hz/trainval2")
    parser.add_argument(
        "--scene-list", dest="scene_list", type=str, default=None,
        help="Scene id list (one per line). Default: <data-root>/nuScenes_Val2.txt.",
    )
    parser.add_argument("--scene", type=str, default=None, help="Process a single scene id instead of the list.")
    parser.add_argument("--frame", type=str, default=None, help="Process a single zero-padded frame id.")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=-1, help="Max valid frames per scene; -1 = all.")
    parser.add_argument("--cameras", type=str, default="5,4,3", help="Comma separated context camera ids (order preserved).")
    parser.add_argument("--render-camera", dest="render_camera", type=int, default=5)
    parser.add_argument("--width-factor", dest="width_factor", type=float, default=3.0)
    parser.add_argument(
        "--car-mask-root", dest="car_mask_root", type=str,
        default="data/nuscenes/processed_10Hz/nuscenes_mask",
    )
    parser.add_argument("--mask-render-view", dest="mask_render_view", action="store_true")
    parser.add_argument("--disable-car-mask", dest="disable_car_mask", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="outputs/nuscenes_wide")
    parser.add_argument("--save-inputs", dest="save_inputs", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _preload_torch_libs(torch) -> None:
    """Map torch and the bundled CUDA runtime into this process before importing extensions."""
    import ctypes

    libdir = Path(torch.__file__).resolve().parent / "lib"
    names = ("libc10.so", "libtorch_cpu.so", "libtorch_python.so", "libtorch.so")
    for name in names:
        path = libdir / name
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    nvidia_root = libdir.parent.parent / "nvidia"
    for path in sorted(nvidia_root.glob("cuda_runtime/lib/libcudart.so*")):
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        break


def resolve_local(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def save_rgb(tensor_chw, path: Path) -> None:
    """Save a [3,H,W] float tensor in [0,1] as a quality-95 JPEG (RGB)."""
    from PIL import Image

    array = (
        tensor_chw.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).contiguous().numpy()
    )
    array = (array * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=95)


def main():
    args = parse_args()

    from omegaconf import OmegaConf
    import torch
    from torch.amp import autocast
    from safetensors.torch import load_file

    # Editable CUDA extensions were built without an RPATH, so dlopen cannot
    # see torch/cuDNN unless those libs are already mapped into the process.
    _preload_torch_libs(torch)

    from pi3.models.pi3 import Pi3
    import model.gaussian_head as gaussian_head_class

    cfg = OmegaConf.load(resolve_local(args.config))
    data_root = resolve_local(args.data_root)
    scene_list = resolve_local(args.scene_list) if args.scene_list else (data_root / "nuScenes_Val2.txt")
    mask_root = resolve_local(args.car_mask_root)
    output_dir = resolve_local(args.output_dir)
    load_from = resolve_local(args.load_from)

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

    # --- model (mirrors demo.py) -------------------------------------------
    model = Pi3()
    model_class = getattr(gaussian_head_class, cfg.Model.Gaussian_head.Name)
    model.gaussian_head = model_class(dim_in=2048, cfg=cfg.Model.Gaussian_head)
    model.gaussian_head.image_backbone.mask_token = None
    weight = load_file(str(load_from), device="cpu")
    model.load_state_dict(weight, strict=False)
    model = model.to(device)
    model.eval()

    from PIL import Image  # RGB size probing
    from dataset.waymo import get_ray_directions, get_rays

    scenes = [args.scene] if args.scene else nw.read_scene_list(scene_list)
    if not scenes:
        raise SystemExit(f"No scenes to process (scene list: {scene_list}).")

    for scene in scenes:
        frames = nw.enumerate_frames(data_root, scene, cameras, args.max_frames, args.frame)
        if not frames:
            print(f"[warn] scene {scene}: no valid frames for cameras {cameras}.", file=sys.stderr)
            continue
        sdir = nw.scene_dir(data_root, scene)

        for frame in frames:
            # --- source resolution / plan ---------------------------------
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
                    raise ValueError(
                        f"Camera {cam} image is {wh} but camera {cameras[0]} is {src_wh}."
                    )

            out_h, out_w = nw.plan_input_hw(src_wh[0], src_wh[1])
            out_wh = (out_w, out_h)

            # --- RGB inputs (PIL LANCZOS, no crop) ------------------------
            rgb_list = [nw.load_rgb_resized(sdir / "images" / f"{frame}_{cam}.jpg", out_wh) for cam in cameras]
            images_np = np.stack(rgb_list, axis=0)  # [S,H,W,3]
            images = torch.from_numpy(images_np).permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # [1,S,3,H,W]

            # --- intrinsics / extrinsics ----------------------------------
            Ks, c2ws = [], []
            for cam in cameras:
                K = nw.scale_intrinsics(
                    nw.read_intrinsics(sdir / "intrinsics" / f"{cam}.txt"), src_wh, out_wh
                )
                Ks.append(K)
                c2ws.append(nw.read_cam2ego(sdir / "cam2ego_extrinsics" / f"{cam}.txt"))
            Ks = np.stack(Ks, axis=0)      # [S,3,3]
            c2ws = np.stack(c2ws, axis=0)  # [S,4,4]

            # --- rays (dataset.waymo, normalize=False) --------------------
            rays_o, rays_d = [], []
            for v in range(len(cameras)):
                fx, fy, cx, cy = Ks[v][0, 0], Ks[v][1, 1], Ks[v][0, 2], Ks[v][1, 2]
                direction = get_ray_directions(
                    out_h, out_w, focal=[float(fx), float(fy)], principal=[float(cx), float(cy)]
                )
                ro, rd = get_rays(
                    direction, torch.from_numpy(c2ws[v]).float(), keepdim=True, normalize=False
                )
                rays_o.append(ro)
                rays_d.append(rd)
            rays_o = torch.stack(rays_o, dim=0)[None]  # [1,S,H,W,3]
            rays_d = torch.stack(rays_d, dim=0)[None]

            S = len(cameras)
            # sky_mask: True = non-sky. nuScenes has no sky mask, so all True.
            sky_mask = torch.ones((1, S, out_h, out_w), dtype=torch.bool)
            input_dict_gs = {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "intrinsics": torch.from_numpy(Ks).float()[None],
                "camera2lidar": torch.from_numpy(c2ws).float()[None],
                "sky_mask": sky_mask.to(images.dtype),
            }

            # --- ego-car keep mask (True = keep) --------------------------
            if args.disable_car_mask:
                keep_np = np.ones((S, out_h, out_w), dtype=bool)
            else:
                masked_cameras = [
                    cam for cam in cameras if cam != render_camera or args.mask_render_view
                ]
                camera_masks = nw.load_camera_keep_masks(mask_root, masked_cameras, out_wh)
                keep_np = nw.build_car_keep_mask(
                    camera_masks, cameras, out_wh, render_camera,
                    mask_render_view=args.mask_render_view, disable_car_mask=False,
                )
            source_keep_mask = torch.from_numpy(keep_np)
            dropped = int((~keep_np).sum())

            # --- wide render target ---------------------------------------
            wide_K, (wide_h, wide_w) = nw.make_wide_intrinsics(
                Ks[render_index], out_wh, args.width_factor
            )
            fovx, fovy = nw.intrinsics_to_fov(wide_K)
            output_dict_gs = {
                "rgb": torch.zeros((1, 1, 3, wide_h, wide_w), dtype=torch.float32),
                "c2w": torch.from_numpy(c2ws[render_index]).float()[None, None],
                "fovx": torch.tensor([[fovx]], dtype=torch.float32),
                "fovy": torch.tensor([[fovy]], dtype=torch.float32),
                "intrinsics": torch.from_numpy(wide_K).float()[None, None],
            }

            # --- to device ------------------------------------------------
            images = images.to(device)
            for key in input_dict_gs:
                input_dict_gs[key] = input_dict_gs[key].to(device)
            for key in output_dict_gs:
                output_dict_gs[key] = output_dict_gs[key].to(device)
            source_keep_mask = source_keep_mask.to(device)

            # reset history so single frames never fuse with previous frames
            model.gaussian_head.history_queue.scenes = None

            with torch.no_grad():
                if amp_dtype is not None:
                    with autocast(device_type="cuda", dtype=amp_dtype):
                        res = model(images)
                else:
                    res = model(images)
                res["scene"] = [scene]
                res["frame"] = [frame]
                res["lidar2world"] = torch.eye(4, dtype=torch.float32, device=device)[None]
                with autocast(device_type="cuda", enabled=False):
                    render_pkg, _ = model.gaussian_head(
                        res, images, 5,
                        input_dict_gs=input_dict_gs,
                        output_dict_gs=output_dict_gs,
                        source_keep_mask=source_keep_mask,
                    )

            # --- save -----------------------------------------------------
            save_rgb(
                render_pkg["image"][0],
                output_dir / scene / "rgb" / f"{frame}_{render_camera}_wide.jpg",
            )
            if args.save_inputs:
                for v, cam in enumerate(cameras):
                    save_rgb(
                        torch.from_numpy(rgb_list[v]).permute(2, 0, 1),
                        output_dir / scene / "inputs" / f"{frame}_{cam}.jpg",
                    )

            print(
                f"scene={scene} frame={frame} input={out_h}x{out_w} "
                f"wide={wide_h}x{wide_w} dropped={dropped}",
                flush=True,
            )
            if not args.disable_car_mask and dropped == 0:
                print(
                    f"[warn] scene={scene} frame={frame}: ego-car mask zeroed 0 pixel-gaussians.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
