"""numpy-only geometry/data helpers for single-frame nuScenes wide inference.

This module deliberately has **no torch dependency** so it can be imported (and
unit tested) without a GPU stack.  Image decoding uses PIL lazily, so importing
the module only needs :mod:`numpy`.

Two input conventions are supported by ``UniSplat`` single-frame inference:

* The network input is aspect-scaled so its long side matches the training long
  side (``train_long``) and the short side is snapped to a multiple of the
  DINOv2 patch size.  There is **no centre crop**.
* The wide render view keeps the resized render-camera focal length and
  principal ``cy`` and only moves ``cx`` to the centre of the wider canvas.

See ``scripts/inference_nuscenes_wide.py`` for the driver.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Per-camera nuScenes ego-car mask file names.  nuScenes camera ids 0..5 are
# CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK_RIGHT,
# CAM_BACK; only the three cameras used by wide inference are mapped here.
CAMERA_MASK_FILES: Dict[int, str] = {
    3: "CAM_BACK_LEFT_mask.png",
    4: "CAM_BACK_RIGHT_mask.png",
    5: "CAM_BACK_mask.png",
}

# Mask polarity: black (< threshold) is the ego car to remove, white (>=) kept.
CAR_MASK_KEEP_THRESHOLD = 128

DEFAULT_TRAIN_LONG = 518
DEFAULT_PATCH = 14
DEFAULT_WIDTH_FACTOR = 3.0


@dataclass(frozen=True)
class DatasetPreset:
    """Where one driving dataset keeps scenes, cameras, and ego-car masks.

    Poses are the same for every preset: single-frame uses ``cam2ego`` as
    OpenCV cam2world, and multi-frame uses that rig in the ego frame with
    ``ego_pose`` as ego-to-world. Only the mask layout differs.
    """

    name: str
    data_root: str
    scene_list_name: str
    cameras: str = "5,4,3"
    render_camera: int = 5
    mask_kind: str = "nuscenes"  # nuscenes | lyft | ddad | none
    mask_root: str = ""
    mask_ext: str = "png"


DATASETS: Dict[str, DatasetPreset] = {
    "nuscenes": DatasetPreset(
        name="nuscenes",
        data_root="data/nuscenes/processed_10Hz/trainval2",
        scene_list_name="nuScenes_Val.txt",
        mask_kind="nuscenes",
        mask_root="data/nuscenes/processed_10Hz/nuscenes_mask",
        mask_ext="png",
    ),
    "ddad": DatasetPreset(
        name="ddad",
        data_root="data/ddad/valid",
        scene_list_name="valid.txt",
        mask_kind="ddad",
        mask_root="data/ddad/valid",
        mask_ext="jpg",
    ),
    "lyft1920": DatasetPreset(
        name="lyft1920",
        data_root="data/lyft/lyft_val1920_3cams",
        scene_list_name="lyft_val1920.txt",
        mask_kind="lyft",
        mask_root="data/lyft/lyft_val1920_3cams/ego_car_masks",
        mask_ext="jpg",
    ),
    "lyft1224": DatasetPreset(
        name="lyft1224",
        data_root="data/lyft/lyft_val1224_3cams",
        scene_list_name="lyft_val1224.txt",
        mask_kind="lyft",
        mask_root="data/lyft/lyft_val1224_3cams/ego_car_masks",
        mask_ext="jpg",
    ),
    "widedrive": DatasetPreset(
        name="widedrive",
        data_root="data/WideDrive_processed/WideDriveVal",
        scene_list_name="val.txt",
        mask_kind="none",
        mask_root="",
        mask_ext="",
    ),
}


REPO_ROOT = Path(__file__).resolve().parents[1]

# On-disk names that refer to the same split. The first existing candidate wins,
# with the name the caller asked for preferred over the aliases.
DIR_ALIAS_GROUPS = (
    ("ddad", "ddad_process"),
    ("valid", "val", "validation"),
    ("trainval", "trainval2"),
    ("nuscenes_mask", "corrected_masks"),
)
LIST_ALIAS_GROUPS = (
    ("valid.txt", "valid2.txt", "val.txt", "validation.txt"),
    ("nuScenes_Val.txt", "nuScenes_Val2.txt"),
    ("nuScenes_Train.txt", "nuScenes_Train2.txt"),
    ("lyft_val1920.txt", "lyft_val19202.txt"),
    ("lyft_val1224.txt", "lyft_val12242.txt"),
    ("val.txt", "valid.txt", "valid2.txt"),
)


def _as_repo_path(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _alias_names(name: str, groups) -> List[str]:
    ordered = [name]
    for group in groups:
        if name not in group:
            continue
        for item in group:
            if item not in ordered:
                ordered.append(item)
    return ordered


def locate_dir(path) -> Optional[Path]:
    """Return ``path`` or the same path with a renamed parent folder.

    ``data/ddad/valid`` is accepted when the checkout stores it as
    ``data/ddad_process/valid``. An existing directory is never rewritten.
    """
    path = _as_repo_path(path)
    if path.is_dir():
        return path
    if path.is_absolute():
        current = Path(path.anchor)
        parts = path.parts[1:]
    else:
        current = Path()
        parts = path.parts
    for part in parts:
        exact = current / part
        if exact.is_dir():
            current = exact
            continue
        found = None
        for alias in _alias_names(part, DIR_ALIAS_GROUPS):
            cand = current / alias
            if cand.is_dir():
                found = cand
                break
        if found is None:
            return None
        current = found
    return current if current.is_dir() else None


def _remember_dir(dirs: List[Path], directory: Path) -> None:
    if directory.is_dir() and directory not in dirs:
        dirs.append(directory)


def locate_file(path, extra_dirs=()) -> Optional[Path]:
    """Find a scene list under renamed split folders or aliased filenames.

    Search order is the exact path, then known filenames in the resolved
    directory, then the same names in a sibling split (``trainval2`` <->
    ``trainval``, ``valid`` <-> ``val``).
    """
    path = _as_repo_path(path)
    if path.is_file():
        return path
    names = _alias_names(path.name, LIST_ALIAS_GROUPS)
    dirs: List[Path] = []
    parent = locate_dir(path.parent)
    if parent is not None:
        _remember_dir(dirs, parent)
    bases = []
    if parent is not None:
        bases.append(parent)
    if path.parent != parent:
        bases.append(path.parent)
    for base in bases:
        grand = base.parent
        located_grand = grand if grand.is_dir() else locate_dir(grand)
        if located_grand is None:
            continue
        for alias in _alias_names(base.name, DIR_ALIAS_GROUPS):
            _remember_dir(dirs, located_grand / alias)
        _remember_dir(dirs, located_grand)
    for extra in extra_dirs:
        extra_path = Path(extra)
        _remember_dir(dirs, extra_path)
        if extra_path.parent.is_dir():
            for alias in _alias_names(extra_path.name, DIR_ALIAS_GROUPS):
                _remember_dir(dirs, extra_path.parent / alias)
    for directory in dirs:
        for name in names:
            cand = directory / name
            if cand.is_file():
                return cand
    return None


def locate_mask_root(path, data_root, mask_kind: str) -> Optional[Path]:
    """Find an ego-car mask directory when its folder was renamed."""
    if not path:
        return None
    found = locate_dir(path)
    if found is not None:
        return found
    root = locate_dir(data_root) or _as_repo_path(data_root)
    if mask_kind == "nuscenes":
        candidates = (
            root / "nuscenes_mask",
            root.parent / "nuscenes_mask",
            root.parent.parent / "nuscenes_mask",
            root.parent / "corrected_masks",
        )
    elif mask_kind == "lyft":
        candidates = (root / "ego_car_masks", root.parent / "ego_car_masks")
    elif mask_kind == "ddad":
        candidates = (root,)
    else:
        candidates = ()
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def _note_relocated(kind: str, requested, found: Path) -> None:
    requested_path = _as_repo_path(requested)
    if requested_path.resolve() == found.resolve():
        return
    print(f"[data] {kind}: {requested_path} -> {found}", file=sys.stderr)


def get_preset(name: str) -> DatasetPreset:
    key = str(name).strip().lower()
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; known are {sorted(DATASETS)}.")
    return DATASETS[key]


def fill_preset_args(args, multi: bool = False):
    """Fill unset paths from ``args.dataset``. WideDrive forces masks off."""
    preset = get_preset(args.dataset)
    if not getattr(args, "data_root", None):
        args.data_root = preset.data_root
    if not getattr(args, "scene_list", None):
        args.scene_list = str(Path(args.data_root) / preset.scene_list_name)
    if not getattr(args, "cameras", None):
        args.cameras = preset.cameras
    if getattr(args, "render_camera", None) is None:
        args.render_camera = preset.render_camera
    if not getattr(args, "car_mask_root", None):
        if preset.mask_kind == "lyft":
            args.car_mask_root = str(Path(args.data_root) / "ego_car_masks")
        elif preset.mask_kind == "ddad":
            args.car_mask_root = args.data_root
        else:
            args.car_mask_root = preset.mask_root
    if not getattr(args, "output_dir", None):
        suffix = "_wide_multiframes" if multi else "_wide"
        args.output_dir = f"outputs/{preset.name}{suffix}"
    args.mask_kind = preset.mask_kind
    args.mask_ext = preset.mask_ext or "png"
    if preset.mask_kind == "none":
        args.disable_car_mask = True
    root = locate_dir(args.data_root)
    if root is not None:
        _note_relocated("data_root", args.data_root, root)
        args.data_root = str(root)
    listed = locate_file(args.scene_list, extra_dirs=(args.data_root,))
    if listed is not None:
        _note_relocated("scene_list", args.scene_list, listed)
        args.scene_list = str(listed)
    if getattr(args, "car_mask_root", None) and not args.disable_car_mask:
        mask = locate_mask_root(args.car_mask_root, args.data_root, args.mask_kind)
        if mask is not None:
            _note_relocated("car_mask_root", args.car_mask_root, mask)
            args.car_mask_root = str(mask)
    return args


# ---------------------------------------------------------------------------
# Resolution / intrinsics planning
# ---------------------------------------------------------------------------


def _round_half_up(value: float) -> int:
    """Round to nearest integer, half away from zero (predictable, unlike round())."""
    return int(math.floor(value + 0.5)) if value >= 0 else -int(math.floor(-value + 0.5))


def plan_input_hw(
    src_w: int, src_h: int, train_long: int = DEFAULT_TRAIN_LONG, patch: int = DEFAULT_PATCH
) -> Tuple[int, int]:
    """Plan the network input ``(out_h, out_w)`` for a ``src_w`` x ``src_h`` image.

    The long side is matched to ``train_long`` (already a multiple of ``patch``
    for the Waymo training canvas ``518``).  The short side is aspect-scaled and
    snapped to the nearest multiple of ``patch`` (at least one patch step).  No
    cropping is performed and the principal point is preserved by
    :func:`scale_intrinsics`.

    Reference: ``1600x900 -> (294, 518)`` because
    ``round(900 * 518 / 1600 / 14) * 14 == 294``.
    """
    src_w, src_h = int(src_w), int(src_h)
    patch = int(patch)
    train_long = int(train_long)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source size {src_w}x{src_h}.")
    if patch <= 0:
        raise ValueError(f"patch must be positive, got {patch}.")
    long_side = max(patch, train_long // patch * patch)

    if src_w >= src_h:
        out_w = long_side
        out_h = max(patch, _round_half_up(src_h * (out_w / src_w) / patch) * patch)
    else:
        out_h = long_side
        out_w = max(patch, _round_half_up(src_w * (out_h / src_h) / patch) * patch)
    return out_h, out_w


def intrinsics_from_params(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3x3 pinhole intrinsics matrix with last row ``[0, 0, 1]``."""
    return np.array(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_intrinsics(
    K: np.ndarray, src_wh: Sequence[int], out_wh: Sequence[int]
) -> np.ndarray:
    """Independently scale ``K`` for a pure resize from ``src_wh`` to ``out_wh``.

    ``src_wh`` and ``out_wh`` are ``(width, height)``.  ``fx``/``cx`` scale by
    ``out_w / src_w`` and ``fy``/``cy`` by ``out_h / src_h``.  Because there is
    no crop, the principal point is only scaled, never shifted.  The returned
    matrix keeps a ``[0, 0, 1]`` last row.
    """
    K = np.asarray(K, dtype=np.float64).copy()
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K.shape}.")
    src_w, src_h = int(src_wh[0]), int(src_wh[1])
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source size {src_w}x{src_h}.")
    scale_x = out_w / src_w
    scale_y = out_h / src_h
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y
    K[2] = (0.0, 0.0, 1.0)
    return K


def iround_hw(value: float) -> int:
    """Round a float dimension to an int (half away from zero)."""
    return _round_half_up(float(value))


def wide_render_size(
    out_wh: Sequence[int], width_factor: float = DEFAULT_WIDTH_FACTOR
) -> Tuple[int, int]:
    """Return ``(wide_h, wide_w)`` for the resized input ``out_wh=(out_w, out_h)``.

    ``wide_h == out_h`` and ``wide_w == round(out_w * width_factor)`` (default 3x).
    """
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if float(width_factor) <= 0:
        raise ValueError(f"width_factor must be positive, got {width_factor}.")
    wide_w = iround_hw(out_w * float(width_factor))
    return out_h, wide_w


def make_wide_intrinsics(
    K_resized: np.ndarray,
    out_wh: Sequence[int],
    width_factor: float = DEFAULT_WIDTH_FACTOR,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Build the wide-view pixel K from the resized render-camera pixel K.

    ``fx``, ``fy`` and ``cy`` are preserved; ``cx`` is moved to the centre of the
    wide canvas ``wide_w / 2``.  Returns ``(K_wide, (wide_h, wide_w))``.
    """
    K_resized = np.asarray(K_resized, dtype=np.float64)
    if K_resized.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K_resized.shape}.")
    wide_h, wide_w = wide_render_size(out_wh, width_factor)
    K_wide = K_resized.copy()
    K_wide[0, 2] = wide_w / 2.0
    K_wide[2] = (0.0, 0.0, 1.0)
    return K_wide, (wide_h, wide_w)


def intrinsics_to_fov(K: np.ndarray) -> Tuple[float, float]:
    """Return ``(fovx, fovy)`` from pixel intrinsics (Waymo convention).

    ``fovx = 2 * atan(cx / fx)`` and ``fovy = 2 * atan(cy / fy)``.  For the wide
    view ``cx == wide_w / 2``, so ``fovx`` is the true widened horizontal FOV.
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fovx = 2.0 * math.atan(cx / fx)
    fovy = 2.0 * math.atan(cy / fy)
    return fovx, fovy


# ---------------------------------------------------------------------------
# Ego-car keep-mask indexing
# ---------------------------------------------------------------------------


def nonsky_drop_mask(keep: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Return a 1-D drop mask over NON-sky pixels in flatten order.

    ``keep`` and ``sky`` are bool ``[S, H, W]``; ``sky=True`` marks a sky pixel.
    The pixel Gaussians produced by ``refine_guassians`` for the non-sky branch
    are ordered by the flattened ``~sky`` pixels, so this helper reproduces that
    ordering exactly (a sky pixel earlier in the image shifts the index of all
    following non-sky pixels).  ``True`` means "drop" (i.e. ``keep`` is False).
    """
    keep = np.asarray(keep, dtype=bool)
    sky = np.asarray(sky, dtype=bool)
    if keep.shape != sky.shape:
        raise ValueError(f"keep and sky must share a shape, got {keep.shape} and {sky.shape}.")
    return ~keep.reshape(-1)[~sky.reshape(-1)]


def sky_drop_mask(keep: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Return a 1-D drop mask over SKY pixels in flatten order (mirror helper)."""
    keep = np.asarray(keep, dtype=bool)
    sky = np.asarray(sky, dtype=bool)
    if keep.shape != sky.shape:
        raise ValueError(f"keep and sky must share a shape, got {keep.shape} and {sky.shape}.")
    sky_flat = sky.reshape(-1)
    return ~keep.reshape(-1)[sky_flat]


def build_car_keep_mask(
    camera_masks: Dict[int, np.ndarray],
    cameras: Sequence[int],
    out_wh: Sequence[int],
    render_camera: int,
    mask_render_view: bool = False,
    disable_car_mask: bool = False,
) -> np.ndarray:
    """Build the ``[S, H, W]`` per-view keep mask (``True`` = keep).

    Default policy: every camera except the render camera uses its own mask;
    the render camera is fully preserved unless ``mask_render_view`` is set.
    ``disable_car_mask`` keeps every Gaussian for every view.  A required but
    missing camera mask is an error.
    """
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    cameras = [int(cam) for cam in cameras]
    render_camera = int(render_camera)
    keep = np.ones((len(cameras), out_h, out_w), dtype=bool)
    if disable_car_mask:
        return keep
    for view_index, cam in enumerate(cameras):
        if cam == render_camera and not mask_render_view:
            continue
        mask = camera_masks.get(cam)
        if mask is None:
            raise KeyError(f"No ego-car mask was loaded for camera {cam}.")
        if mask.shape != (out_h, out_w):
            raise ValueError(
                f"Ego-car mask for camera {cam} has shape {mask.shape} but the "
                f"model input is {out_h}x{out_w}."
            )
        keep[view_index] = mask
    return keep


# ---------------------------------------------------------------------------
# File / text parsing
# ---------------------------------------------------------------------------


def parse_floats(text: str) -> List[float]:
    return [float(tok) for tok in text.replace(",", " ").split()]


def read_scene_list(path) -> List[str]:
    """One scene id per non-empty line."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def parse_intrinsics_text(text: str) -> Tuple[float, float, float, float]:
    """Parse the 9-value intrinsics file, returning ``(fx, fy, cx, cy)``.

    The on-disk file is **not** a 3x3 matrix: it stores nine floats whose first
    four are ``fx, fy, cx, cy`` and whose remaining entries are zeros.
    """
    values = parse_floats(text)
    if len(values) < 4:
        raise ValueError(f"Expected at least 4 intrinsic values, found {len(values)}.")
    return values[0], values[1], values[2], values[3]


def read_intrinsics(path) -> np.ndarray:
    """Read an intrinsics txt file into a 3x3 pixel K matrix."""
    fx, fy, cx, cy = parse_intrinsics_text(Path(path).read_text())
    return intrinsics_from_params(fx, fy, cx, cy)


def read_matrix(path, size: int = 4) -> np.ndarray:
    """Read a flat ``size*size`` txt file into a ``size`` x ``size`` matrix."""
    values = parse_floats(Path(path).read_text())
    if len(values) != size * size:
        raise ValueError(
            f"Expected {size * size} values in {path}, found {len(values)}."
        )
    return np.asarray(values, dtype=np.float64).reshape(size, size)


def read_cam2ego(path) -> np.ndarray:
    """Read a 4x4 row-major OpenCV camera-to-ego transform."""
    return read_matrix(path, size=4)


def normalize_frame_id(frame, available: Optional[Iterable[str]] = None) -> str:
    """Normalise a frame id to the zero-padded on-disk form (``0 -> 000``)."""
    text = str(frame).strip()
    if available is not None and text in set(available):
        return text
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def parse_cameras(text) -> Tuple[int, ...]:
    cameras = tuple(int(tok) for tok in str(text).replace(",", " ").split())
    if not cameras:
        raise ValueError("At least one camera id is required.")
    if len(set(cameras)) != len(cameras):
        raise ValueError(f"Duplicate camera ids in {cameras}.")
    return cameras


# ---------------------------------------------------------------------------
# Scene / frame enumeration and image IO (PIL imported lazily)
# ---------------------------------------------------------------------------


def scene_dir(data_root, scene) -> Path:
    return Path(data_root) / str(scene)


def enumerate_frames(
    data_root,
    scene,
    cameras: Sequence[int],
    max_frames: Optional[int] = None,
    frame=None,
) -> List[str]:
    """Frames that have ``images/{frame:03d}_{cam}.jpg`` for every camera."""
    images_dir = scene_dir(data_root, scene) / "images"
    if not images_dir.is_dir():
        return []
    cameras = [int(cam) for cam in cameras]
    suffixes = {f"_{cam}.jpg" for cam in cameras}
    all_frames = set()
    for entry in images_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        for suffix in suffixes:
            if name.endswith(suffix):
                all_frames.add(name[: -len(suffix)])
                break
    frames = sorted(
        frame_id
        for frame_id in all_frames
        if all((images_dir / f"{frame_id}_{cam}.jpg").is_file() for cam in cameras)
    )
    if frame is not None:
        wanted = normalize_frame_id(frame, frames)
        return [wanted] if wanted in frames else []
    if max_frames is not None and max_frames >= 0:
        frames = frames[:max_frames]
    return frames


def load_rgb_resized(path, out_wh: Sequence[int]) -> np.ndarray:
    """Load a jpg as RGB, resize with LANCZOS to ``out_wh=(w,h)``, ``[0,1]`` float32 HWC."""
    from PIL import Image

    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (out_w, out_h):
            image = image.resize((out_w, out_h), Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def load_keep_mask_resized(path, out_wh: Sequence[int]) -> np.ndarray:
    """Load an ego-car mask, NEAREST-resize to ``out_wh``, return bool keep.

    Same source scale as the RGB images and no crop; ``True`` = keep (>= 128).
    """
    from PIL import Image

    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ego-car mask: {path}")
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (out_w, out_h):
            mask = mask.resize((out_w, out_h), Image.NEAREST)
        array = np.asarray(mask)
    return array >= CAR_MASK_KEEP_THRESHOLD


def camera_mask_path(
    mask_root,
    cam: int,
    mask_kind: str = "nuscenes",
    scene=None,
    mask_ext: str = "png",
) -> Path:
    """Resolve one camera's ego-car mask.

    * ``nuscenes``: ``<mask_root>/CAM_*_mask.png`` (dataset-level).
    * ``lyft``: ``<mask_root>/{cam}.jpg`` (split-level ``ego_car_masks``).
    * ``ddad``: ``<mask_root>/<scene>/ego_car_masks/{cam}.jpg`` (per scene).
    * ``none``: WideDrive has no ego-car mask; callers must keep every pixel.
    """
    cam = int(cam)
    kind = str(mask_kind)
    root = Path(mask_root)
    if kind == "nuscenes":
        if cam not in CAMERA_MASK_FILES:
            raise KeyError(
                f"No nuScenes ego-car mask defined for camera {cam}; known are "
                f"{sorted(CAMERA_MASK_FILES)}."
            )
        return root / CAMERA_MASK_FILES[cam]
    if kind == "lyft":
        return root / f"{cam}.{mask_ext}"
    if kind == "ddad":
        if scene is None:
            raise ValueError("DDAD ego-car masks are per-scene; pass scene.")
        return root / str(scene) / "ego_car_masks" / f"{cam}.{mask_ext}"
    if kind == "none":
        raise ValueError("This dataset has no ego-car mask; keep every pixel.")
    raise KeyError(f"Unknown mask kind {mask_kind!r}.")


def load_camera_keep_masks(
    mask_root,
    cameras: Sequence[int],
    out_wh: Sequence[int],
    mask_kind: str = "nuscenes",
    scene=None,
    mask_ext: str = "png",
) -> Dict[int, np.ndarray]:
    """Load every selected camera's keep mask, failing loudly when missing."""
    masks: Dict[int, np.ndarray] = {}
    for cam in cameras:
        cam = int(cam)
        path = camera_mask_path(
            mask_root, cam, mask_kind=mask_kind, scene=scene, mask_ext=mask_ext
        )
        masks[cam] = load_keep_mask_resized(path, out_wh)
    return masks


def enumerate_windows(frames: Sequence[str], num_frames: int) -> List[Tuple[str, ...]]:
    """Causal windows of ``num_frames`` consecutive listed frames, oldest first.

    The last element is the frame that should be rendered. Fewer than
    ``num_frames`` valid frames yields no partial window.
    """
    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
    frames = list(frames)
    if len(frames) < num_frames:
        return []
    return [
        tuple(frames[start : start + num_frames])
        for start in range(len(frames) - num_frames + 1)
    ]


def select_windows(
    frames: Sequence[str], num_frames: int, frame=None
) -> List[Tuple[str, ...]]:
    """Windows whose newest frame matches ``frame``, or every window if unset."""
    windows = enumerate_windows(frames, num_frames)
    if frame is None:
        return windows
    wanted = normalize_frame_id(frame, frames)
    return [window for window in windows if window[-1] == wanted]


def frames_are_consecutive(window: Sequence[str]) -> bool:
    """True when integer frame ids increase by 1, which the history queue requires."""
    ids = [int(frame) for frame in window]
    return all(newer == older + 1 for older, newer in zip(ids, ids[1:]))


def mask_render_camera(is_newest: bool, mask_render_view: bool = False) -> bool:
    """History frames always mask the render camera; the newest frame does not.

    ``mask_render_view`` forces the newest render camera to be masked too.
    """
    if not is_newest:
        return True
    return bool(mask_render_view)
