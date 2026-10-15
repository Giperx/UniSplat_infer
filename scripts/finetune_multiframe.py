#!/usr/bin/env python3
"""Fine-tune the gaussian head on 3-frame nuScenes or Lyft windows.

The Pi3 aggregator stays frozen. Each step reads one scene's consecutive
frames (default 3 x 3 cameras). Every frame rebuilds Gaussians and the history
queue fuses the previous frame, matching multiframe inference. The render loss
is only the newest frame's 3 original cameras, using the head's existing
rec / dyn / aux-voxel losses. History Gaussians are detached by the queue, so
the backward pass does not unroll through earlier frames.

Preview renders are written every ``preview_interval`` steps for the configured
scene/frame pairs: the 3 supervised cameras and the camera-5 wide view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset.driving_multiframe import DrivingMultiFrameDataset, collate_windows  # noqa: E402
from dataset import nuscenes_wide as nw  # noqa: E402
from dataset.utils import Logger  # noqa: E402
from pi3.models.pi3 import Pi3  # noqa: E402
import model.gaussian_head as gaussian_head_class  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-frame gaussian-head fine-tune.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--load-from", default=None, help="Full pretrained safetensors. Default: config or pretrained/model.safetensors.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--preview-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None, help="Debug cap on the number of training windows.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO / path


def freeze_aggregator(model):
    """Train gaussian_head only. Pi3 encoder/decoder/heads stay frozen."""
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("gaussian_head.")
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return trainable, frozen


def gaussian_param_groups(model, weight_decay, head_lr, backbone_lr):
    groups = [
        {"params": [], "weight_decay": 0.0, "lr": backbone_lr},
        {"params": [], "weight_decay": weight_decay, "lr": backbone_lr},
        {"params": [], "weight_decay": 0.0, "lr": head_lr},
        {"params": [], "weight_decay": weight_decay, "lr": head_lr},
    ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        backbone = "gaussian_head.image_backbone" in name
        bias = param.ndim == 1 or name.endswith(".bias")
        index = (0 if backbone else 2) + (0 if bias else 1)
        groups[index]["params"].append(param)
    return [group for group in groups if group["params"]]


def _take(batch, key, time_index, device, dtype=None):
    value = batch[key][:, time_index].to(device)
    if dtype is not None and value.is_floating_point():
        value = value.to(dtype)
    return value


def frame_inputs(batch, time_index, device):
    images = _take(batch, "images", time_index, device)
    input_dict = {
        "rays_o": _take(batch, "rays_o", time_index, device),
        "rays_d": _take(batch, "rays_d", time_index, device),
        "intrinsics": _take(batch, "intrinsics", time_index, device),
        "camera2lidar": _take(batch, "camera2lidar", time_index, device),
        "sky_mask": _take(batch, "sky_mask", time_index, device, images.dtype),
        "dynamics_region": _take(batch, "dynamics_region", time_index, device, images.dtype),
    }
    output_dict = {
        "rgb": images,
        "c2w": input_dict["camera2lidar"],
        "fovx": _take(batch, "fovx", time_index, device),
        "fovy": _take(batch, "fovy", time_index, device),
        "intrinsics": input_dict["intrinsics"],
        # Waymo leaves the current-view RGB mask empty and supervises dynamics
        # from the input mask. Same split here: rec loss sees every pixel.
        "dynamics_region": torch.zeros_like(input_dict["dynamics_region"]),
    }
    scenes = batch["scene"]
    frames = [window[time_index] for window in batch["frames"]]
    lidar2world = _take(batch, "lidar2world", time_index, device)
    return images, input_dict, output_dict, scenes, frames, lidar2world


def run_pi3(model, images, amp_dtype):
    with torch.no_grad():
        if amp_dtype is None:
            return model(images)
        with autocast(device_type="cuda", dtype=amp_dtype):
            return model(images)


def attach_pose(res, scenes, frames, lidar2world):
    res["scene"] = list(scenes)
    res["frame"] = list(frames)
    res["lidar2world"] = lidar2world
    return res


def forward_window(model, batch, device, stage, amp_dtype, with_loss):
    """Replay a window oldest to newest. Loss is computed only on the last frame."""
    head = model.gaussian_head
    head.history_queue.scenes = None
    steps = batch["images"].shape[1]
    newest = steps - 1
    last_render = None
    loss = None
    for time_index in range(steps):
        images, input_dict, output_dict, scenes, frames, lidar2world = frame_inputs(batch, time_index, device)
        is_newest = time_index == newest
        res = attach_pose(run_pi3(model, images, amp_dtype), scenes, frames, lidar2world)
        if is_newest and with_loss:
            with autocast(device_type="cuda", enabled=False):
                last_render, loss = head(
                    res, images, 5,
                    input_dict_gs=input_dict,
                    output_dict_gs=output_dict,
                    test=False,
                    stage=stage,
                )
        else:
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=False):
                    last_render, loss = head(
                        res, images, 5,
                        input_dict_gs=input_dict,
                        output_dict_gs=output_dict,
                        test=True,
                    )
    return last_render, loss


def loss_total(loss_dict):
    total = None
    values = {}
    for key, value in loss_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        values[key] = float(value.detach())
        if value.requires_grad:
            total = value if total is None else total + value
    if total is None:
        raise RuntimeError(f"No differentiable loss in {list(loss_dict)}")
    values["total"] = float(total.detach())
    return total, values


def save_rgb(tensor_chw, path: Path):
    from PIL import Image
    import numpy as np

    array = tensor_chw.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).contiguous().numpy()
    array = (array * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=95)


def _render_views(model, batch, device, amp_dtype, output_override=None):
    """Replay one window. ``output_override`` replaces only the newest render targets."""
    head = model.gaussian_head
    head.history_queue.scenes = None
    steps = batch["images"].shape[1]
    rendered = None
    for time_index in range(steps):
        images, input_dict, output_dict, scenes, frames, lidar2world = frame_inputs(batch, time_index, device)
        if output_override is not None and time_index == steps - 1:
            output_dict = {key: value.to(device) for key, value in output_override.items()}
        res = attach_pose(run_pi3(model, images, amp_dtype), scenes, frames, lidar2world)
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=False):
                rendered, _ = head(
                    res, images, 5,
                    input_dict_gs=input_dict,
                    output_dict_gs=output_dict,
                    test=True,
                )
    return rendered


def preview_window(model, dataset, spec, device, amp_dtype, out_dir, width_factor, render_camera):
    scene, frame = spec
    sample = dataset.window_ending(scene, frame)
    batch = collate_windows([sample])
    was_training = model.training
    model.eval()
    render = _render_views(model, batch, device, amp_dtype)
    images = render["image"]
    cameras = dataset.cameras
    if images.shape[0] != len(cameras):
        raise RuntimeError(f"Expected {len(cameras)} preview views, got {images.shape[0]}.")
    gt = sample["images"][-1]
    for view, cam in enumerate(cameras):
        stem = f"{scene}_{frame}_{cam}"
        save_rgb(images[view], out_dir / f"{stem}.jpg")
        save_rgb(gt[view], out_dir / f"{stem}_gt.jpg")

    render_index = cameras.index(int(render_camera))
    out_w, out_h = sample["out_hw"]
    view_k, view_c2w, fovx, fovy, _, _ = nw.render_views(
        sample["intrinsics"][-1, render_index].numpy(),
        sample["camera2lidar"][-1, render_index].numpy(),
        (out_w, out_h),
        multiplane=False,
        width_factor=width_factor,
    )
    panel_h, panel_w = nw.wide_render_size((out_w, out_h), width_factor)
    wide = {
        "rgb": torch.zeros((1, 1, 3, panel_h, panel_w), dtype=torch.float32),
        "c2w": torch.from_numpy(view_c2w).float()[None],
        "fovx": torch.from_numpy(fovx).float()[None],
        "fovy": torch.from_numpy(fovy).float()[None],
        "intrinsics": torch.from_numpy(view_k).float()[None],
    }
    wide_render = _render_views(model, batch, device, amp_dtype, output_override=wide)
    save_rgb(wide_render["image"][0], out_dir / f"{scene}_{frame}_{render_camera}_wide.jpg")
    if was_training:
        model.train()


def parse_preview_samples(value):
    samples = []
    for item in value or []:
        text = str(item).replace(",", " ").split()
        if len(text) != 2:
            raise ValueError(f"preview sample must be 'scene frame', got {item!r}")
        samples.append((text[0], nw.normalize_frame_id(text[1])))
    return samples


def save_checkpoint(model, optimizer, scaler, scheduler, step, work_dir, logging):
    save_dir = work_dir / f"model_step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    weight_path = save_dir / "model.safetensors"
    save_file(model.state_dict(), str(weight_path))
    state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, work_dir / "training_state.pth")
    logging.info(f"Saved {weight_path}")


def main():
    args = parse_args()
    cfg = OmegaConf.load(resolve(args.config))
    work_dir = resolve(args.work_dir or cfg.Train.get("work_dir", f"work_dirs/{Path(args.config).stem}"))
    work_dir.mkdir(parents=True, exist_ok=True)
    logging = Logger(True, work_dir=str(work_dir))
    logging.info(cfg)

    data_cfg = cfg.Dataset
    dataset = DrivingMultiFrameDataset(
        data_root=data_cfg.data_root,
        scene_list=data_cfg.scene_list,
        cameras=data_cfg.get("cameras", "5,4,3"),
        num_frames=int(data_cfg.get("num_frames", 3)),
        train_long=int(data_cfg.get("train_long", 518)),
        max_scenes=data_cfg.get("max_scenes", None),
        max_windows=args.max_windows if args.max_windows is not None else data_cfg.get("max_windows", None),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No {dataset.num_frames}-frame windows under {dataset.data_root}")

    batch_size = int(args.batch_size or cfg.Train.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.Train.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_windows,
        drop_last=False,
    )
    if len(loader) == 0:
        raise SystemExit(f"Dataloader is empty (dataset={len(dataset)}, batch_size={batch_size}).")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = Pi3()
    head_cfg = cfg.Model.Gaussian_head
    model.gaussian_head = getattr(gaussian_head_class, head_cfg.Name)(dim_in=2048, cfg=head_cfg)
    if hasattr(model.gaussian_head, "image_backbone"):
        model.gaussian_head.image_backbone.mask_token = None

    load_from = args.load_from or cfg.Model.get("pretrained", "pretrained/model.safetensors")
    load_from = resolve(load_from)
    if not load_from.is_file():
        raise SystemExit(f"pretrained weights not found: {load_from}")
    missing, unexpected = model.load_state_dict(load_file(str(load_from)), strict=False)
    logging.info(f"Loaded {load_from} missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        logging.info(f"Unexpected keys (first 8): {unexpected[:8]}")

    trainable, frozen = freeze_aggregator(model)
    logging.info(f"Trainable gaussian_head params: {trainable}. Frozen aggregator params: {frozen}.")
    model.to(device)
    model.train()

    opt_cfg = cfg.Optimizer
    optimizer = torch.optim.AdamW(
        gaussian_param_groups(
            model,
            weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
            head_lr=float(opt_cfg.get("head_lr", 5.0e-5)),
            backbone_lr=float(opt_cfg.get("backbone_lr", 1.5e-5)),
        ),
        betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    max_steps = int(args.max_steps or cfg.Train.get("max_steps", 10000))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        total_steps=max_steps,
        pct_start=0.01,
        anneal_strategy="cos",
    )

    start_step = 0
    state_path = work_dir / "training_state.pth"
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu")
        start_step = int(state["step"])
        weight = work_dir / f"model_step_{start_step}" / "model.safetensors"
        if weight.is_file():
            model.load_state_dict(load_file(str(weight)), strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            if "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            logging.info(f"Resumed from step {start_step}")
        else:
            logging.info(f"training_state exists but {weight} is missing; starting from pretrained weights.")
            start_step = 0

    stage = int(cfg.Train.get("forward_stage", 3))
    log_interval = int(args.log_interval or cfg.Train.get("log_interval", 20))
    save_interval = int(args.save_interval or cfg.Train.get("save_interval", 2000))
    preview_interval = int(args.preview_interval or cfg.Train.get("preview_interval", 2000))
    preview_samples = parse_preview_samples(cfg.Train.get("preview_samples", []))
    width_factor = float(cfg.Train.get("width_factor", 3.0))
    render_camera = int(cfg.Train.get("render_camera", 5))
    if render_camera not in dataset.cameras:
        raise SystemExit(f"render_camera {render_camera} is not in cameras {dataset.cameras}")
    for scene, frame in preview_samples:
        dataset.window_ending(scene, frame)
    logging.info(
        f"Windows: {len(dataset)}. Batch: {batch_size}. Steps: {start_step} -> {max_steps}. "
        f"Preview every {preview_interval}: {preview_samples}"
    )

    step = start_step
    running = {}
    seen = 0
    while step < max_steps:
        for batch in loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            _, loss_dict = forward_window(model, batch, device, stage, amp_dtype, with_loss=True)
            total, values = loss_total(loss_dict)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad], 10.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for key, value in values.items():
                running[key] = running.get(key, 0.0) + value
            seen += 1
            if step % log_interval == 0:
                mean = {key: value / seen for key, value in running.items()}
                loss_str = ", ".join(f"{key}: {value:.6f}" for key, value in mean.items())
                lr = optimizer.param_groups[-1]["lr"]
                logging.info(f"step {step}/{max_steps} lr {lr:.7f} {loss_str}")
                running = {}
                seen = 0
            if preview_samples and step % preview_interval == 0:
                out_dir = work_dir / "preview" / f"step_{step:07d}"
                for spec in preview_samples:
                    preview_window(
                        model, dataset, spec, device, amp_dtype, out_dir, width_factor, render_camera
                    )
                model.train()
                logging.info(f"Preview written to {out_dir}")
            if step % save_interval == 0 or step == max_steps:
                save_checkpoint(model, optimizer, scaler, scheduler, step, work_dir, logging)
            if step >= max_steps:
                break
    logging.info(f"Finished at step {step}")


if __name__ == "__main__":
    main()
