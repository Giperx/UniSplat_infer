import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
from pathlib import Path
from datetime import timedelta
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from accelerate.utils import set_seed
from safetensors.torch import load_file

import dataset as waymo_dataset
from dataset.utils import Logger
import dataset.utils as train_utils
from dataset.samplers.distributed_group_in_batch_sampler import DistributedGroupInBatchSampler, get_dist_info
from pi3.models.pi3 import Pi3
import model.gaussian_head as gaussian_head_class


# only these gaussian_head sub-modules are involved in scale & shift prediction
SCALE_SHIFT_KEYS = (
    'gaussian_head.scale_head',
    'gaussian_head.shift_head',
    'gaussian_head.point_decoder',
)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument("--config", type=str, default='configs/waymo_stage1.yaml', help="Path to the YAML config file")
    parser.add_argument("--work_dir_root", type=str, default="./work_dirs", help="Root directory for storing outputs")
    parser.add_argument("--data_path", type=str, default='data/waymo', help="data load from")
    parser.add_argument("--pi3_ckpt", type=str, default=None, help="Pi3 pretrained safetensors")
    parser.add_argument("--dinov2_ckpt", type=str, default=None, help="DINOv2 pretrained weight for image_backbone")
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--tcp_port', type=int, default=18888)
    return parser.parse_args()


def init_distributed_mode(args):
    if args.launcher == 'none':
        return
    if args.launcher != 'pytorch':
        raise NotImplementedError(f"launcher {args.launcher} is not supported")

    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    tcp_port = os.environ.get('MASTER_PORT', args.tcp_port)
    dist_url = f'tcp://{master_addr}:{tcp_port}'
    machine_rank = int(os.environ.get('NODE_RANK', 0))
    machine_num = int(os.environ.get('WORLD_SIZE_NODE', 1))
    num_gpus = torch.cuda.device_count()
    dist.init_process_group(
        backend='nccl', init_method=dist_url,
        rank=args.local_rank + machine_rank * num_gpus,
        world_size=num_gpus * machine_num,
        timeout=timedelta(minutes=60))
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    print(f'| distributed init (rank {args.rank}): {dist_url}')
    dist.barrier()


def freeze_pi3_backbone(model, logging):
    for tmp in [model.encoder, model.decoder, model.point_decoder, model.conf_decoder, model.camera_decoder,
                model.point_head, model.conf_head, model.camera_head]:
        for name, param in tmp.named_parameters():
            param.requires_grad = False
    model.register_token.requires_grad = False
    logging.info("Pi3 backbone parameters frozen")


def apply_stage_freeze(model, stage, logging):
    """Freeze parameters according to the training stage:
        stage 1: only train scale_head / shift_head / point_decoder of gaussian_head
        stage 2: freeze scale_head / shift_head / point_decoder, train the rest of
                 gaussian_head using GT-aligned scale/shift
        stage 3: same freezing as stage 2, but the rendering path is driven by the
                 (frozen) predicted scale/shift instead of the GT alignment
    """
    if stage == 1:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not any(key in name for key in SCALE_SHIFT_KEYS):
                param.requires_grad = False
        logging.info(f"[stage 1] trainable: {SCALE_SHIFT_KEYS}")
    elif stage in (2, 3):
        for name, param in model.named_parameters():
            if any(key in name for key in SCALE_SHIFT_KEYS):
                param.requires_grad = False
        logging.info(f"[stage {stage}] frozen: {SCALE_SHIFT_KEYS}")
    else:
        raise ValueError(f"unknown training stage {stage}")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    config_name = Path(args.config).stem
    work_dir = os.path.join(args.work_dir_root, config_name)
    os.makedirs(work_dir, exist_ok=True)

    args.distributed = False
    args.rank = 0
    args.world_size = 1
    init_distributed_mode(args)
    is_main_process = (args.rank == 0)
    set_seed(42)
    logging = Logger(is_main_process, work_dir=work_dir)
    logging.info(f"Distributed: {args.distributed}, rank: {args.rank}, local_rank: {args.local_rank}, world_size: {args.world_size}")
    logging.info(cfg)

    # resume from latest checkpoint in work_dir if any
    latest_path, latest_epoch = train_utils.find_latest_checkpoint(work_dir)
    is_resume = latest_path is not None
    if is_resume:
        logging.info(f"Found latest checkpoint: {latest_path} at epoch {latest_epoch}")
    else:
        logging.info("No previous checkpoint found, starting from scratch")

    # dataset & dataloader
    dataset_name = cfg.Dataset.get('name', 'WaymoDataset')
    dataset_class = getattr(waymo_dataset, dataset_name)
    dataset = dataset_class(scene_root=args.data_path, is_train=True, cfg=cfg.Dataset)
    rank, world_size = get_dist_info()
    batch_size = cfg.Train.get('batch_size', 2)
    sampler = DistributedGroupInBatchSampler(dataset, batch_size=batch_size, seed=None,
                                             rank=rank, world_size=world_size, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=cfg.Train.get('num_workers', 8), pin_memory=True)

    # build model
    model = Pi3()
    pi3_ckpt = args.pi3_ckpt or cfg.Model.get('pi3_ckpt', None)
    if pi3_ckpt is not None:
        weight = load_file(pi3_ckpt)
        model.load_state_dict(weight)
        logging.info(f"Loaded Pi3 pretrained weights from {pi3_ckpt}")

    model_name = cfg.Model.Gaussian_head.Name
    model_class = getattr(gaussian_head_class, model_name)
    model.gaussian_head = model_class(dim_in=2048, cfg=cfg.Model.Gaussian_head)

    dinov2_ckpt = args.dinov2_ckpt or cfg.Model.get('dinov2_ckpt', None)
    if hasattr(model.gaussian_head, 'image_backbone') and dinov2_ckpt is not None:
        ckpts = torch.load(dinov2_ckpt, map_location='cpu')
        model.gaussian_head.image_backbone.load_state_dict(ckpts, strict=True)
        logging.info(f"Loaded DINOv2 weights into gaussian_head.image_backbone from {dinov2_ckpt}")
    if hasattr(model.gaussian_head, 'image_backbone'):
        model.gaussian_head.image_backbone.mask_token = None

    # pretrained weights from a previous stage
    pretrained = cfg.Model.get('pretrained', None)
    if pretrained is not None and not is_resume:
        prtrained_weight = load_file(pretrained)
        missing, unexpected = model.load_state_dict(prtrained_weight, strict=False)
        logging.info(f"Loaded pretrained weights from {pretrained} (missing={len(missing)}, unexpected={len(unexpected)})")

    if is_resume:
        checkpoint = load_file(f'{latest_path}/model.safetensors')
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Resumed model weights from {latest_path}")

    # apply freezing: Pi3 backbone always frozen, then stage-specific
    freeze_pi3_backbone(model, logging)
    stage = cfg.Train.get('stage', 3)
    apply_stage_freeze(model, stage, logging)

    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # optimizer
    backbone_cfg = cfg.Optimizer.get('backbone_cfg')
    finetype = backbone_cfg.get('type', 'get_parameter_groups_v3')
    lr_cfg = backbone_cfg.get('lr_cfg')
    finetune_func = getattr(train_utils, finetype)
    param_groups = finetune_func(model, **lr_cfg)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    scaler = GradScaler()

    # scheduler
    scheduler = None
    if cfg.get('Scheduler', None) is not None:
        assert cfg.Scheduler['name'] == 'OneCycleLR', "only OneCycleLR is supported"
        total_steps = len(dataloader) * cfg['Train']['epoch']
        base_lr = [g['lr'] for g in optimizer.param_groups if 'lr' in g]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=base_lr, total_steps=total_steps,
            pct_start=0.01, anneal_strategy='cos')

    if is_resume:
        training_state_path = os.path.join(work_dir, 'training_state.pth')
        checkpoint_data = torch.load(training_state_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        start_epoch = checkpoint_data['epoch']
        assert start_epoch == latest_epoch
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        logging.info(f"Resumed training state from epoch {start_epoch}")
    else:
        start_epoch = 0

    if args.distributed and args.launcher == 'pytorch':
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logging.info(f"Dataset loaded, {len(dataset)} samples, {len(dataloader)} iterations / epoch")

    num_epochs = cfg['Train']['epoch']
    log_interval = args.log_interval
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    running_losses = {}
    batch_count = 0

    for epoch in range(start_epoch, num_epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        for batch_idx in range(len(dataloader)):
            batch = next(dataloader_iter)
            for key in batch.keys():
                if key in ('input_dict_gs', 'output_dict_gs'):
                    continue
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            images = batch['images'].to(device)
            unwrapped_model = model.module if args.distributed else model

            # Pi3 is always frozen, so run it under no_grad + amp
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=amp_dtype):
                    res = model(images)

            loss_all = {}
            with autocast(device_type='cuda', enabled=False):
                input_dict_gs = batch['input_dict_gs']
                input_dict_gs['sky_mask'] = batch['sky_mask'].to(images.dtype).to(device)
                input_dict_gs['single_depthmaps'] = batch['single_depthmaps'].to(device)
                input_dict_gs['intrinsics'] = batch['intrinsics'].to(device)
                input_dict_gs['camera2lidar'] = batch['camera2lidar'].to(device)
                input_dict_gs['dynamics_region'] = batch['dynamics_region'].to(device)
                res['scene'] = batch['scene']
                res['frame'] = batch['frame']
                res['lidar2world'] = batch['camera_pose'] @ batch['camera2lidar'].inverse()

                _, loss_gaussian = unwrapped_model.gaussian_head(
                    res, images, 5,
                    input_dict_gs=batch['input_dict_gs'],
                    output_dict_gs=batch['output_dict_gs'],
                    test=False, stage=stage)
                loss_all.update(loss_gaussian)

            loss_sum = 0.0
            for k, v in loss_all.items():
                if not isinstance(v, torch.Tensor) or not v.requires_grad:
                    continue
                loss_sum = loss_sum + v
                running_losses[k] = running_losses.get(k, 0.0) + v.item()
            running_losses['total'] = running_losses.get('total', 0.0) + float(loss_sum)

            scaler.scale(loss_sum).backward()
            scaler.unscale_(optimizer)
            max_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            if max_norm > 100:
                print(f"Gradient clipping: {max_norm} {batch['scene']} {batch['frame']}")
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            batch_count += 1
            if batch_count % log_interval == 0:
                gathered = {k: v / log_interval for k, v in running_losses.items()}
                if is_main_process:
                    lr_str = ", ".join([f"{g['lr']:.7f}" for g in optimizer.param_groups])
                    loss_str = ", ".join([f"{k}: {v:.6f}" for k, v in gathered.items()])
                    progress = f"[{batch_idx + 1}/{len(dataloader)}]"
                    logging.info(f"Epoch {epoch + 1}/{num_epochs} {progress}, LR: {lr_str}, Loss: {loss_str}")
                running_losses = {}

        if args.distributed:
            dist.barrier()
        if is_main_process:
            logging.info(f"Completed Epoch {epoch + 1}/{num_epochs}")

        # checkpoint
        if is_main_process:
            save_hook = None
            for hook in cfg.get('Hook', []):
                if hook['type'] == 'ckpthook':
                    save_hook = hook
                    break
            if save_hook and ((epoch + 1) % save_hook['interval'] == 0 or epoch == num_epochs - 1):
                unwrapped_model = model.module if args.distributed else model
                save_dir = os.path.join(work_dir, f"model_epoch_{epoch + 1}")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped_model.save_pretrained(save_dir)
                logging.info(f"Model checkpoint saved at {save_dir}")

                training_state = {
                    'epoch': epoch + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }
                if scheduler is not None:
                    training_state['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(training_state, os.path.join(work_dir, 'training_state.pth'))
                logging.info(f"Training state saved at {work_dir}/training_state.pth")
        if args.distributed:
            dist.barrier()


if __name__ == "__main__":
    main()
