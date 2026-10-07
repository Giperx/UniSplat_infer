"""PSNR, MAE, RMSE, SSIM, and LPIPS, following the DWSplat wide evaluators.

Sparse datasets use the v2 split: Left/Right score the GT mask with sparse
luminance SSIM, Center uses dense window SSIM, and Center_masked also applies
the ego-car mask. Dense WideDrive scores Full/Left/Center/Right on the resized
camera-2 image. UniSplat does not write a render alpha mask, so masked dense
rows stay empty unless a GT mask is present.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import car_mask_path, dense_bounds, load_binary_mask, strip_bounds


def sparse_ssim(rendered, gt, mask, k1=0.01):
    """Per-pixel luminance SSIM on valid pixels. No sliding window."""
    if int(mask.sum()) == 0:
        return 0.0
    pred = rendered[mask] * 255.0
    ref = gt[mask] * 255.0
    c1 = (k1 * 255.0) ** 2
    term = (2.0 * pred * ref + c1) / (pred ** 2 + ref ** 2 + c1)
    return float(term.mean())


def _gaussian_kernel(win_size, sigma, channels, device):
    coords = torch.arange(win_size, dtype=torch.float32, device=device) - win_size // 2
    weights = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :]).expand(channels, 1, win_size, win_size)
    return kernel.contiguous()


def dense_ssim_map(rendered, gt, device, win_size=11, data_range=1.0):
    left = torch.from_numpy(np.ascontiguousarray(rendered)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    right = torch.from_numpy(np.ascontiguousarray(gt)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    channels = left.shape[1]
    kernel = _gaussian_kernel(win_size, 1.5, channels, device)
    pad = win_size // 2
    mu_x = F.conv2d(left, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(right, kernel, padding=pad, groups=channels)
    sigma_x = F.conv2d(left ** 2, kernel, padding=pad, groups=channels) - mu_x ** 2
    sigma_y = F.conv2d(right ** 2, kernel, padding=pad, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(left * right, kernel, padding=pad, groups=channels) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean(dim=1).squeeze(0).detach().cpu().numpy()


def scalar_errors(rendered, gt, mask):
    if mask is None or int(mask.sum()) == 0:
        return None
    diff = rendered - gt
    mae = float(np.abs(diff)[mask].mean())
    mse = float(np.square(diff)[mask].mean())
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    return {"mae": mae, "rmse": rmse, "psnr": psnr, "n_pixels": int(mask.sum())}


def match_valid(rendered, gt, mask):
    """CDF-align rendered pixels selected by mask. Invalid pixels stay zero."""
    from skimage.exposure import match_histograms

    if int(mask.sum()) == 0:
        return None
    matched = np.zeros_like(rendered)
    aligned = match_histograms(rendered[mask][None, ...], gt[mask][None, ...], channel_axis=-1)[0]
    matched[mask] = np.clip(aligned, 0.0, 1.0)
    return matched


def load_lpips(device):
    import lpips

    model = lpips.LPIPS(net="alex", spatial=True).to(device)
    model.eval()
    return model


def spatial_lpips(lpips_fn, rendered, gt, device):
    pred = torch.from_numpy(np.ascontiguousarray(rendered)).float().permute(2, 0, 1).unsqueeze(0)
    ref = torch.from_numpy(np.ascontiguousarray(gt)).float().permute(2, 0, 1).unsqueeze(0)
    pred = pred.to(device) * 2.0 - 1.0
    ref = ref.to(device) * 2.0 - 1.0
    with torch.no_grad():
        lpips_map = lpips_fn(pred, ref)
    return np.squeeze(lpips_map.detach().float().cpu().numpy())


def resize_map(lpips_map, shape_hw):
    if lpips_map.shape == tuple(shape_hw):
        return lpips_map
    tensor = torch.from_numpy(np.asarray(lpips_map)).float()[None, None]
    resized = F.interpolate(tensor, size=tuple(shape_hw), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def score_region(rendered, gt, mask, device, lpips_fn, ssim_kind, want_lpips, histogram_match):
    if mask is None or int(np.asarray(mask).sum()) == 0:
        return None
    work = rendered
    if histogram_match:
        work = match_valid(rendered, gt, mask)
        if work is None:
            return None
    errors = scalar_errors(work, gt, mask)
    if errors is None:
        return None
    if ssim_kind == "sparse":
        errors["ssim"] = sparse_ssim(work, gt, mask)
    else:
        ssim_map = dense_ssim_map(work, gt, device)
        errors["ssim"] = float(ssim_map[mask].mean()) if mask.any() else 0.0
    if histogram_match:
        errors["hm_ssim"] = sparse_ssim(work, gt, mask)
    if want_lpips:
        lpips_map = resize_map(spatial_lpips(lpips_fn, work, gt, device), mask.shape)
        errors["lpips"] = float(lpips_map[mask].mean()) if mask.any() else float("nan")
    else:
        errors["lpips"] = float("nan")
    return errors


def _overall(results):
    parts = [results[name] for name in ("Left", "Center", "Right") if name in results]
    if not parts:
        return None
    total = sum(part["n_pixels"] for part in parts)
    if total <= 0:
        return None
    overall = {"n_pixels": total, "lpips": float("nan")}
    for name in ("mae", "rmse", "psnr", "ssim"):
        overall[name] = sum(part[name] * part["n_pixels"] for part in parts) / total
    return overall


def score_sparse(render, gt, gt_mask, preset, scene, device, lpips_fn, histogram_match):
    """v2 regions. Returns (metrics, car_mask_missing)."""
    bounds = strip_bounds(render.shape[1])
    results = {}
    car_missing = False
    for name, (x0, x1) in bounds.items():
        region_mask = gt_mask[:, x0:x1]
        if name == "Center":
            errors = score_region(
                render[:, x0:x1], gt[:, x0:x1], region_mask, device, lpips_fn,
                "dense", True, histogram_match,
            )
            if errors is not None:
                results["Center"] = errors
            strip_w = x1 - x0
            path = car_mask_path(preset, scene)
            car = None
            if path is not None and path.is_file():
                car = load_binary_mask(path, (strip_w, render.shape[0]))
            else:
                car_missing = True
            masked = region_mask if car is None else (region_mask & car)
            masked_errors = score_region(
                render[:, x0:x1], gt[:, x0:x1], masked, device, lpips_fn,
                "dense", True, histogram_match,
            )
            if masked_errors is not None:
                results["Center_masked"] = masked_errors
        else:
            errors = score_region(
                render[:, x0:x1], gt[:, x0:x1], region_mask, device, lpips_fn,
                "sparse", False, histogram_match,
            )
            if errors is not None:
                results[name] = errors
    overall = _overall(results)
    if overall is not None:
        results["Overall"] = overall
    return results, car_missing


def score_dense(render, gt, gt_mask, device, lpips_fn, histogram_match):
    bounds = dense_bounds(render.shape[1])
    results = {}
    for name, (x0, x1) in bounds.items():
        crop_r = render[:, x0:x1]
        crop_g = gt[:, x0:x1]
        full = np.ones(crop_r.shape[:2], dtype=bool)
        unmasked = score_region(crop_r, crop_g, full, device, lpips_fn, "dense", True, histogram_match)
        if unmasked is not None:
            results[f"{name}_unmasked"] = unmasked
        if gt_mask is not None:
            masked = score_region(
                crop_r, crop_g, gt_mask[:, x0:x1], device, lpips_fn, "dense", True, histogram_match,
            )
            if masked is not None:
                results[f"{name}_masked"] = masked
    return results
