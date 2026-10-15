"""CPU checks for the 3-frame fine-tune window sampler."""

from pathlib import Path

import torch

from dataset.driving_multiframe import (
    DrivingMultiFrameDataset,
    collate_windows,
    consecutive_windows,
)
from scripts.finetune_multiframe import frame_inputs, parse_preview_samples


def test_consecutive_windows_drop_gaps():
    windows = consecutive_windows(["000", "001", "002", "004", "005", "006"], 3)
    assert windows == [["000", "001", "002"], ["004", "005", "006"]]


def test_parse_preview_samples():
    assert parse_preview_samples(["002 002", "078,013"]) == [("002", "002"), ("078", "013")]


def test_nuscenes_window_matches_inference_packing():
    root = Path("data/nuscenes/processed_10Hz/trainval2")
    if not (root / "000" / "images").is_dir():
        return
    dataset = DrivingMultiFrameDataset(
        data_root=root,
        scene_list=root / "nuScenes_Train.txt",
        max_scenes=1,
        max_windows=1,
    )
    sample = dataset[0]
    assert sample["frames"][0] != sample["frames"][-1]
    assert [int(frame) for frame in sample["frames"]] == list(range(int(sample["frames"][0]), int(sample["frames"][0]) + 3))
    assert sample["images"].shape[0] == 3
    assert sample["images"].shape[1] == 3
    assert sample["images"].shape[-2:] == (294, 518)
    assert sample["sky_mask"].shape == (3, 3, 294, 518)
    assert sample["dynamics_region"].shape == sample["sky_mask"].shape
    # White sky pngs become 0. The top of camera 5 is sky.
    assert sample["sky_mask"][0, 0, :8].mean() < 0.2
    assert sample["sky_mask"][0, 0, -8:].mean() > 0.8
    # All cameras of one frame share one ego pose.
    pose = sample["lidar2world"][0]
    assert torch.allclose(pose[0], pose[1]) and torch.allclose(pose[0], pose[2])

    batch = collate_windows([sample])
    images, input_dict, output_dict, scenes, frames, _ = frame_inputs(batch, 2, torch.device("cpu"))
    assert scenes == [sample["scene"]]
    assert frames == [sample["frames"][-1]]
    assert torch.equal(images, sample["images"][-1].unsqueeze(0))
    assert torch.equal(output_dict["rgb"], images)
    assert torch.equal(output_dict["c2w"], input_dict["camera2lidar"])
    assert torch.count_nonzero(output_dict["dynamics_region"]) == 0
    assert input_dict["dynamics_region"].shape[-2:] == (294, 518)
