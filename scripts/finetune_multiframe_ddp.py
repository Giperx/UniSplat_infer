#!/usr/bin/env python3
"""Multi-GPU fine-tune. The single-GPU script is unchanged.

Each rank owns one shard of the windows and keeps ``Train.batch_size`` (default 1).
Launch with torchrun, from the repo root:

    torchrun --nproc_per_node=2 scripts/finetune_multiframe_ddp.py \
        --config configs/finetune_nuscenes_multiframe.yaml

The loss is computed inside the DDP forward. History frames still run under
no_grad; only the newest frame's gaussian-head loss is synchronized.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset.driving_multiframe import DrivingMultiFrameDataset, collate_windows  # noqa: E402
from dataset.utils import Logger  # noqa: E402
from finetune_multiframe import (  # noqa: E402
    freeze_aggregator,
    forward_window,
    gaussian_param_groups,
    loss_total,
    parse_args,
    parse_preview_samples,
    preview_window,
    resolve,
    save_checkpoint,
)
from pi3.models.pi3 import Pi3  # noqa: E402
import model.gaussian_head as gaussian_head_class  # noqa: E402


class FineTuneStep(torch.nn.Module):
    """One window in, one loss out, so DDP can all-reduce the gaussian head."""

    def __init__(self, model, stage, amp_dtype):
        super().__init__()
        self.model = model
        self.stage = stage
        self.amp_dtype = amp_dtype
        self.last_values = {}

    def forward(self, batch):
        device = batch["images"].device
        _, loss_dict = forward_window(
            self.model, batch, device, self.stage, self.amp_dtype, with_loss=True
        )
        total, values = loss_total(loss_dict)
        self.last_values = values
        return total


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise SystemExit(
            "Multi-GPU fine-tune must be launched with torchrun, for example:\n"
            "  torchrun --nproc_per_node=2 scripts/finetune_multiframe_ddp.py "
            "--config configs/finetune_nuscenes_multiframe.yaml"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def average_metrics(values, device, world_size):
    keys = sorted(values)
    packed = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= world_size
    return {key: float(packed[index]) for index, key in enumerate(keys)}


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0
    cfg = OmegaConf.load(resolve(args.config))
    if args.work_dir:
        work_dir = resolve(args.work_dir)
    else:
        # Keep this off the single-GPU work_dir stored in the yaml.
        configured = str(cfg.Train.get("work_dir", f"work_dirs/{Path(args.config).stem}"))
        work_dir = resolve(configured + "_ddp")
    if is_main:
        work_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(device_ids=[local_rank])
    logging = Logger(is_main, work_dir=str(work_dir))
    if is_main:
        logging.info(cfg)
        logging.info(f"DDP world_size={world_size}")

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
    if len(dataset) < world_size:
        raise SystemExit(
            f"Need at least one window per rank, got {len(dataset)} windows and world_size={world_size}."
        )

    batch_size = int(args.batch_size or cfg.Train.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.Train.get("num_workers", 4))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_windows,
        drop_last=True,
    )
    if len(loader) == 0:
        raise SystemExit(
            f"Dataloader is empty (windows={len(dataset)}, batch_size={batch_size}, world_size={world_size})."
        )

    device = torch.device(f"cuda:{local_rank}")
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    torch.manual_seed(42)
    model = Pi3()
    head_cfg = cfg.Model.Gaussian_head
    model.gaussian_head = getattr(gaussian_head_class, head_cfg.Name)(dim_in=2048, cfg=head_cfg)
    if hasattr(model.gaussian_head, "image_backbone"):
        model.gaussian_head.image_backbone.mask_token = None

    load_from = resolve(args.load_from or cfg.Model.get("pretrained", "pretrained/model.safetensors"))
    if not load_from.is_file():
        raise SystemExit(f"pretrained weights not found: {load_from}")
    missing, unexpected = model.load_state_dict(load_file(str(load_from)), strict=False)
    if is_main:
        logging.info(f"Loaded {load_from} missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            logging.info(f"Unexpected keys (first 8): {unexpected[:8]}")

    trainable, frozen = freeze_aggregator(model)
    if is_main:
        logging.info(f"Trainable gaussian_head params: {trainable}. Frozen aggregator params: {frozen}.")
    model.to(device)

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
    scaler = GradScaler(enabled=True)
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
            if is_main:
                logging.info(f"Resumed from step {start_step}")
        else:
            if is_main:
                logging.info(f"training_state exists but {weight} is missing; starting from pretrained weights.")
            start_step = 0

    stage = int(cfg.Train.get("forward_stage", 3))
    step_module = FineTuneStep(model, stage, amp_dtype).to(device)
    ddp = DDP(
        step_module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

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
    if is_main:
        logging.info(
            f"Windows: {len(dataset)}. Per-GPU batch: {batch_size}. World size: {world_size}. "
            f"Steps: {start_step} -> {max_steps}. Preview every {preview_interval}: {preview_samples}"
        )

    step = start_step
    running = {}
    seen = 0
    epoch = 0
    ddp.train()
    while step < max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            total = ddp(batch)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad], 10.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            averaged = average_metrics(step_module.last_values, device, world_size)
            for key, value in averaged.items():
                running[key] = running.get(key, 0.0) + value
            seen += 1
            if step % log_interval == 0:
                if is_main:
                    mean = {key: value / seen for key, value in running.items()}
                    loss_str = ", ".join(f"{key}: {value:.6f}" for key, value in mean.items())
                    lr = optimizer.param_groups[-1]["lr"]
                    logging.info(f"step {step}/{max_steps} lr {lr:.7f} {loss_str}")
                running = {}
                seen = 0
            if preview_samples and step % preview_interval == 0:
                if is_main:
                    out_dir = work_dir / "preview" / f"step_{step:07d}"
                    raw = step_module.model
                    for spec in preview_samples:
                        preview_window(
                            raw, dataset, spec, device, amp_dtype, out_dir, width_factor, render_camera
                        )
                    raw.train()
                    logging.info(f"Preview written to {out_dir}")
                dist.barrier(device_ids=[local_rank])
            if step % save_interval == 0 or step == max_steps:
                if is_main:
                    save_checkpoint(model, optimizer, scaler, scheduler, step, work_dir, logging)
                dist.barrier(device_ids=[local_rank])
            if step >= max_steps:
                break
        epoch += 1
    if is_main:
        logging.info(f"Finished at step {step}")
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
