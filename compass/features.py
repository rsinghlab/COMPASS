import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, convex_hull_image
from typing import Dict, List, Optional, Tuple


def extract_box_crop(img_tensor: torch.Tensor, box: torch.Tensor, target_size: int = 64) -> torch.Tensor:
    """
    Crop the region defined by `box` from `img_tensor` and resize to `target_size`.
    Returns a tensor of shape [C, target_size, target_size].
    """
    if img_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor of shape [C,H,W], got {img_tensor.shape}.")
    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    C, H, W = img_tensor.shape
    x1 = int(torch.clamp(box[0], 0, W - 1).item())
    y1 = int(torch.clamp(box[1], 0, H - 1).item())
    x2 = int(torch.clamp(box[2], x1 + 1, W).item())
    y2 = int(torch.clamp(box[3], y1 + 1, H).item())

    crop = img_tensor[:, y1:y2, x1:x2].contiguous()
    if crop.numel() == 0:
        return torch.zeros((C, target_size, target_size), dtype=img_tensor.dtype)

    crop = crop.unsqueeze(0)
    resized = F.interpolate(
        crop,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).contiguous()


HU_FEATURE_COUNT = 7
TOTAL_FEATURE_COUNT = HU_FEATURE_COUNT + 2  # Hu moments + solidity + eccentricity

def _to_grayscale_01(img: torch.Tensor) -> torch.Tensor:
    """Return float32 grayscale in [0,1]."""
    if img.ndim == 3:
        gray = img.mean(dim=0)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError("img must be shaped (H,W) or (C,H,W)")
    g = gray.to(torch.float32)
    # normalize if it looks like 0..255
    if torch.isfinite(g).any() and g.max() > 1.5:
        g = g / 255.0
    # clip mild outliers to reduce hot pixels
    lo, hi = torch.quantile(g.flatten(), torch.tensor([0.001, 0.999], device=g.device))
    g = g.clamp(min=float(lo), max=float(hi))
    g = (g - float(lo)) / (float(hi - lo) + 1e-12)
    return g.clamp(0.0, 1.0)


def _to_grayscale_unit(img: torch.Tensor) -> torch.Tensor:
    """
    Return grayscale float32 in [0,1] without per-crop contrast stretching.
    This better mirrors OpenComet thresholding behavior.
    """
    if img.ndim == 3:
        gray = img.mean(dim=0)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError("img must be shaped (H,W) or (C,H,W)")
    g = gray.to(torch.float32)
    if torch.isfinite(g).any() and g.max() > 1.5:
        g = g / 255.0
    return g.clamp(0.0, 1.0)


def _huang_threshold_u8(img_u8: np.ndarray) -> int:
    """ImageJ-compatible Huang threshold on a uint8 image."""
    hist = np.bincount(img_u8.reshape(-1).astype(np.int64), minlength=256).astype(np.int64)
    nonzero = np.where(hist > 0)[0]
    if nonzero.size == 0:
        return 0
    first_bin = int(nonzero[0])
    last_bin = int(nonzero[-1])
    if last_bin <= first_bin:
        return first_bin

    term = 1.0 / float(last_bin - first_bin)
    mu_0 = np.zeros((256,), dtype=np.float64)
    sum_pix = 0
    num_pix = 0
    for ih in range(first_bin, 256):
        sum_pix += ih * int(hist[ih])
        num_pix += int(hist[ih])
        if num_pix > 0:
            mu_0[ih] = float(sum_pix) / float(num_pix)

    mu_1 = np.zeros((256,), dtype=np.float64)
    sum_pix = 0
    num_pix = 0
    for ih in range(last_bin, 0, -1):
        sum_pix += ih * int(hist[ih])
        num_pix += int(hist[ih])
        if num_pix > 0:
            mu_1[ih - 1] = float(sum_pix) / float(num_pix)

    threshold = 0
    min_ent = float("inf")
    for it in range(256):
        ent = 0.0
        for ih in range(0, it + 1):
            mu_x = 1.0 / (1.0 + term * abs(float(ih) - float(mu_0[it])))
            if 1e-6 < mu_x < 0.999999:
                ent += float(hist[ih]) * (
                    -mu_x * float(np.log(mu_x)) - (1.0 - mu_x) * float(np.log(1.0 - mu_x))
                )
        for ih in range(it + 1, 256):
            mu_x = 1.0 / (1.0 + term * abs(float(ih) - float(mu_1[it])))
            if 1e-6 < mu_x < 0.999999:
                ent += float(hist[ih]) * (
                    -mu_x * float(np.log(mu_x)) - (1.0 - mu_x) * float(np.log(1.0 - mu_x))
                )
        if ent < min_ent:
            min_ent = ent
            threshold = it
    return int(threshold)


@torch.no_grad()
def find_comet_pixels(
    crop: torch.Tensor,
) -> torch.Tensor:
    """
    Return (y,x) coords of foreground comet pixels from the crop using an
    OpenComet-inspired local thresholding and component-selection pipeline.
    `crop` can be [H,W] or [C,H,W]; values may be 0..255 or 0..1.
    """
    I = _to_grayscale_unit(crop)
    H, W = int(I.shape[0]), int(I.shape[1])
    if H < 3 or W < 3:
        return torch.empty((0, 2), dtype=torch.long, device=crop.device)

    I_u8 = np.clip(np.rint(I.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    if int(I_u8.max()) <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=crop.device)
    thresh = float(_huang_threshold_u8(I_u8))
    if thresh <= 0.0 and bool((I_u8 > 0).any()):
        thresh = 1.0

    mask = I_u8 >= thresh
    morph_fp = np.ones((3, 3), dtype=bool)
    for _ in range(3):
        mask = binary_dilation(mask, footprint=morph_fp)
    for _ in range(3):
        mask = binary_erosion(mask, footprint=morph_fp)

    if not bool(mask.any()):
        return torch.empty((0, 2), dtype=torch.long, device=crop.device)

    min_area = max(64, min(400, int(0.02 * H * W)))
    labeled = label(mask.astype(np.uint8), connectivity=2)
    components = regionprops(labeled)
    if len(components) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=crop.device)

    def _front_centroid(mask_bool: np.ndarray, x0: int, y0: int, width: int, height: int) -> int:
        limit = int(np.ceil(float(x0) + float(width) * 0.1))
        limit = max(x0 + 1, min(int(mask_bool.shape[1]), limit))
        y_sum = 0.0
        cnt = 0
        for x in range(int(x0), int(limit)):
            for y in range(int(y0), int(y0 + height)):
                if bool(mask_bool[y, x]):
                    y_sum += float(y)
                    cnt += 1
        if cnt <= 0:
            ys_tmp, _ = np.where(mask_bool)
            return int(np.mean(ys_tmp)) if ys_tmp.size > 0 else int(y0 + height // 2)
        return int(y_sum / float(cnt))

    def _y_symmetry(
        mask_bool: np.ndarray,
        x0: int,
        y0: int,
        width: int,
        height: int,
        y_front_centroid: int,
    ) -> float:
        if width <= 0:
            return 1.0
        absdy = 0.0
        valid_cols = 0
        for x in range(int(x0), int(x0 + width)):
            col = mask_bool[int(y0): int(y0 + height), int(x)]
            cnt = int(col.sum())
            if cnt <= 0:
                continue
            ys_local = np.where(col)[0] + int(y0)
            sum1 = int((ys_local < int(y_front_centroid)).sum())
            sum2 = int((ys_local >= int(y_front_centroid)).sum())
            absdy += abs(sum2 - sum1) / float(cnt)
            valid_cols += 1
        if valid_cols <= 0:
            return 1.0
        return float(absdy / float(valid_cols))

    center_y = (float(H) - 1.0) / 2.0
    center_x = (float(W) - 1.0) / 2.0
    candidate_rows: List[Dict[str, object]] = []
    for comp in components:
        area = int(comp.area)
        if area < min_area:
            continue
        comp_mask = labeled == int(comp.label)
        ys_c, xs_c = np.where(comp_mask)
        if ys_c.size == 0:
            continue
        x0 = int(xs_c.min())
        x1 = int(xs_c.max())
        y0 = int(ys_c.min())
        y1 = int(ys_c.max())
        width = int(x1 - x0 + 1)
        height = int(y1 - y0 + 1)
        if width <= 0 or height <= 0:
            continue

        hull_mask = convex_hull_image(comp_mask).astype(bool)
        hull_area = int(hull_mask.sum())
        convexity = float(area) / float(max(1, hull_area))
        hratio = float(height) / float(max(1, width))
        y_front_centroid = _front_centroid(comp_mask, x0, y0, width, height)
        symmetry = _y_symmetry(comp_mask, x0, y0, width, height, y_front_centroid)
        roi_center_y = float(y0) + float(height) / 2.0
        centerline_diff = abs(float(y_front_centroid) - roi_center_y) / float(max(1, height))
        valid = (
            convexity >= 0.85
            and symmetry <= 0.5
            and hratio <= 1.05
            and centerline_diff <= 0.2
        )

        centroid_y = float(comp.centroid[0])
        centroid_x = float(comp.centroid[1])
        dist2 = (centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2
        mean_intensity = float(I_u8[comp_mask].mean()) if area > 0 else 0.0
        candidate_rows.append(
            {
                "mask": comp_mask,
                "area": area,
                "dist2": dist2,
                "mean_intensity": mean_intensity,
                "valid": bool(valid),
            }
        )

    if len(candidate_rows) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=crop.device)

    valid_rows = [row for row in candidate_rows if bool(row["valid"])]
    rows_to_rank = valid_rows if len(valid_rows) > 0 else candidate_rows
    rows_sorted = sorted(
        rows_to_rank,
        key=lambda row: (
            float(row["dist2"]),
            -int(row["area"]),
            -float(row["mean_intensity"]),
        ),
    )
    best_mask = rows_sorted[0]["mask"]
    ys_np, xs_np = np.where(best_mask)
    if ys_np.size == 0:
        return torch.empty((0, 2), dtype=torch.long, device=crop.device)

    ys = torch.from_numpy(ys_np).to(device=crop.device, dtype=torch.long)
    xs = torch.from_numpy(xs_np).to(device=crop.device, dtype=torch.long)
    coords = torch.stack([ys, xs], dim=1)
    return coords


@torch.no_grad()
def find_comet_pixels_from_mask(
    mask: torch.Tensor,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Given a full-image SAM mask, return (y,x) coords of comet pixels within [y1:y2, x1:x2].

    The input `mask` can be [H,W] or [1,H,W] and may contain probabilities or logits.
    """
    if mask.ndim == 3:
        # assume [1,H,W] or [C,H,W]; use first channel
        mask2d = mask[0]
    elif mask.ndim == 2:
        mask2d = mask
    else:
        raise ValueError("mask must be shaped (H,W) or (1,H,W)")

    # If values look like logits, convert to probabilities
    if mask2d.dtype.is_floating_point:
        # Heuristic: if values are not already in [0,1], apply sigmoid
        if mask2d.min() < 0.0 or mask2d.max() > 1.0:
            mask_prob = torch.sigmoid(mask2d)
        else:
            mask_prob = mask2d
    else:
        # treat non-float as binary mask
        mask_prob = mask2d.to(torch.float32)

    mask_bin = mask_prob >= float(threshold)

    H, W = mask_bin.shape
    # Clamp box coordinates to valid range
    x1_clamped = max(0, min(int(x1), W - 1))
    y1_clamped = max(0, min(int(y1), H - 1))
    x2_clamped = max(0, min(int(x2), W))
    y2_clamped = max(0, min(int(y2), H))
    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
        return torch.empty((0, 2), dtype=torch.long, device=mask.device)

    mask_crop = mask_bin[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    if not mask_crop.any():
        return torch.empty((0, 2), dtype=torch.long, device=mask.device)

    ys, xs = torch.where(mask_crop)
    coords = torch.stack([ys.to(mask.device), xs.to(mask.device)], dim=1).to(torch.long)
    return coords

def create_comet_profile(
    idx,
    x1,
    y1,
    x2,
    y2,
    score,
    damage_class=-1,
    overlap=0,
    cutoff=0,
    empty=0,
    tail_percent=float("nan"),
    tail_moment=float("nan"),
    olive_moment=float("nan"),
    measurements: Optional[Dict[str, float]] = None,
):
    # map detection idx to its profile
    profile = {}
    coords = (x1, y1, x2, y2)
    if overlap or cutoff or empty:
        label = 0
    else:
        label = 1
    profile[idx] = (
        coords,
        score,
        label,
        damage_class,
        overlap,
        cutoff,
        empty,
        tail_percent,
        tail_moment,
        olive_moment,
        measurements or {},
    )
    return profile

@torch.no_grad()
def compute_box_ml_features(
    img_tensor: torch.Tensor,
    box: torch.Tensor,
    score: float,
    idx: int,
    mask: Optional[torch.Tensor] = None,
    use_masks: bool = False,
    image_id: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute comet-assay heuristics from an image crop defined by `box`.

    Returns [pct_DNA_tail, tail_moment, olive_tail_moment].
    """
    # convert image to greyscale 
    if img_tensor.ndim == 3:
        gray = img_tensor.mean(dim=0)
    elif img_tensor.ndim == 2:
        gray = img_tensor
    else:
        raise ValueError("img_tensor must be shaped (H,W) or (C,H,W)")

    def conv_opencomet(signal: torch.Tensor, kernel: torch.Tensor, pad: bool) -> torch.Tensor:
        """
        Match OpenComet's Java convFilter behavior.
        - pad=True: centered kernel with truncated borders, output length = len(signal)
        - pad=False: valid convolution, output length = len(signal) + 1 - len(kernel)
        """
        if signal.ndim != 1 or kernel.ndim != 1:
            raise ValueError("conv_opencomet expects 1D signal and 1D kernel.")
        w = int(signal.numel())
        n = int(kernel.numel())
        if n <= 0:
            raise ValueError("kernel must be non-empty.")
        if not pad and n > w:
            return torch.empty((0,), dtype=signal.dtype, device=signal.device)

        out_len = w if pad else (w + 1 - n)
        out = torch.zeros(out_len, dtype=signal.dtype, device=signal.device)
        kern_radius = int(n // 2)
        for i in range(out_len):
            kern_start = (i - kern_radius) if pad else i
            if pad:
                j0 = max(0, -kern_start)
                j1 = min(n, w - kern_start)
                x0 = max(0, kern_start)
                x1 = x0 + (j1 - j0)
                if j1 > j0:
                    out[i] = (kernel[j0:j1] * signal[x0:x1]).sum()
            else:
                out[i] = (kernel * signal[kern_start:kern_start + n]).sum()
        return out

    def _smooth_column_background(bg_col: torch.Tensor, radius: int = 2) -> torch.Tensor:
        """OpenComet-like local smoothing using a +/-2 column window."""
        if bg_col.numel() == 0:
            return bg_col
        n = int(bg_col.numel())
        out = torch.zeros_like(bg_col)
        for i in range(n):
            j0 = max(0, i - radius)
            j1 = min(n, i + radius + 1)
            out[i] = bg_col[j0:j1].mean()
        return out

    def _conv_filter_np(x: np.ndarray, kernel: np.ndarray, pad: bool) -> np.ndarray:
        if x.ndim != 1 or kernel.ndim != 1:
            raise ValueError("_conv_filter_np expects 1D inputs.")
        if kernel.size == 0:
            return np.zeros((int(x.size),), dtype=np.float64) if pad else np.zeros((0,), dtype=np.float64)
        out_len = int(x.size) if pad else int(x.size + 1 - kernel.size)
        if out_len <= 0:
            return np.zeros((0,), dtype=np.float64)
        y = np.zeros((out_len,), dtype=np.float64)
        kern_radius = int(np.floor(kernel.size / 2.0))
        for i in range(out_len):
            kern_start = int(i - kern_radius) if pad else int(i)
            acc = 0.0
            for j in range(int(kernel.size)):
                idx = kern_start + j
                if 0 <= idx < int(x.size):
                    acc += float(kernel[j]) * float(x[idx])
            y[i] = acc
        return y

    def _get_local_thresh(hist: np.ndarray, n_bins: int = 255, percent: float = 0.95) -> int:
        total = float(np.sum(hist[: int(n_bins)]))
        if total <= 0.0:
            return 0
        running = 0.0
        for i in range(int(n_bins)):
            running += float(hist[i])
            if running > float(percent) * total:
                return int(i - 1)
        return 0

    def _binary_bound_rect(mask_bin: np.ndarray) -> Tuple[int, int, int, int]:
        ys_b, xs_b = np.where(mask_bin)
        if ys_b.size == 0:
            return (0, 0, 0, 0)
        min_x = int(xs_b.min())
        max_x = int(xs_b.max())
        min_y = int(ys_b.min())
        max_y = int(ys_b.max())
        # Mirror OpenComet's Rectangle width/height convention from max-min.
        return (min_x, min_y, int(max_x - min_x), int(max_y - min_y))

    def _front_centroid(mask_bool: np.ndarray, x0: int, y0: int, width: int, height: int) -> int:
        limit = int(np.ceil(float(x0) + float(width) * 0.1))
        limit = max(x0 + 1, min(int(mask_bool.shape[1]), limit))
        y_sum = 0.0
        cnt = 0
        for x in range(int(x0), int(limit)):
            for y in range(int(y0), int(y0 + height)):
                if 0 <= y < int(mask_bool.shape[0]) and 0 <= x < int(mask_bool.shape[1]) and bool(mask_bool[y, x]):
                    y_sum += float(y)
                    cnt += 1
        if cnt <= 0:
            ys_tmp, _ = np.where(mask_bool)
            return int(np.mean(ys_tmp)) if ys_tmp.size > 0 else int(y0 + height // 2)
        return int(y_sum / float(cnt))

    def _head_height(mask_crop: np.ndarray, xcenter: int) -> int:
        if mask_crop.size == 0 or xcenter < 0 or xcenter >= int(mask_crop.shape[1]):
            return 0
        return int(mask_crop[:, int(xcenter)].sum())

    def _mask_width_height(mask_bool: np.ndarray) -> Tuple[int, int]:
        ys_m, xs_m = np.where(mask_bool)
        if ys_m.size == 0:
            return (0, 0)
        w = int(xs_m.max() - xs_m.min() + 1)
        h = int(ys_m.max() - ys_m.min() + 1)
        return (w, h)

    def _x_intensity_centroid_opencomet(intensity_img: np.ndarray, roi_mask: np.ndarray) -> float:
        ys_m, xs_m = np.where(roi_mask)
        if ys_m.size == 0:
            return 0.0
        x0_m = int(xs_m.min())
        x1_m = int(xs_m.max())
        total_intensity = float(intensity_img[roi_mask].sum())
        if total_intensity <= 0.0:
            return float(x0_m)
        half_intensity = total_intensity / 2.0
        running = 0.0
        for x in range(x0_m, x1_m + 1):
            col_mask = roi_mask[:, x]
            if bool(col_mask.any()):
                running += float(intensity_img[col_mask, x].sum())
            if running > half_intensity:
                return float(x)
        return float(x1_m)

    def _column_avg_opencomet(
        img: np.ndarray,
        comet_mask_bool: np.ndarray,
        roi_bounds: Tuple[int, int, int, int],
        profile_rect: Tuple[int, int, int, int],
    ) -> np.ndarray:
        roi_x, roi_y, roi_w, roi_h = roi_bounds
        px, py, pw, ph = profile_rect
        col_avg = np.zeros((int(pw),), dtype=np.float64)
        if pw <= 0 or ph <= 0:
            return col_avg

        for i in range(int(pw)):
            col_sum = 0.0
            for j in range(int(ph)):
                img_x = int(i + px)
                img_y = int(j + py)
                mask_x = int(i + px - roi_x)
                mask_y = int(j + py - roi_y)
                in_img = 0 <= img_x < int(img.shape[1]) and 0 <= img_y < int(img.shape[0])
                in_mask = 0 <= mask_x < int(roi_w) and 0 <= mask_y < int(roi_h)
                if in_img and in_mask and bool(comet_mask_bool[roi_y + mask_y, roi_x + mask_x]):
                    col_sum += float(img[img_y, img_x])
            # OpenComet divides by profile rectangle height (not masked count).
            col_avg[i] = col_sum / float(ph)
        return col_avg

    def _head_edge_opencomet(
        img: np.ndarray,
        comet_mask_bool: np.ndarray,
        roi_bounds: Tuple[int, int, int, int],
        circularity: float,
    ) -> int:
        x0, y0, w0, h0 = roi_bounds
        yc = _front_centroid(comet_mask_bool, x0, y0, w0, h0)
        if float(circularity) < 0.9:
            profile_rect = (int(x0), int(yc - 5), int(w0), 10)
        else:
            profile_rect = (int(x0), int(y0), int(w0), int(h0))

        comet_profile = _column_avg_opencomet(img, comet_mask_bool, roi_bounds, profile_rect)
        kernel_width = int(w0 // 10)
        if kernel_width <= 0 or comet_profile.size == 0:
            return int(w0)
        smooth_kernel = np.full((kernel_width,), 1.0 / float(kernel_width), dtype=np.float64)
        diff_kernel = np.array([-1.0, 1.0], dtype=np.float64)

        y1 = _conv_filter_np(comet_profile, smooth_kernel, pad=True)
        y2 = _conv_filter_np(y1, diff_kernel, pad=False)
        y3 = _conv_filter_np(y2, smooth_kernel, pad=True)
        y4 = _conv_filter_np(y3, diff_kernel, pad=False)
        dd_profile = _conv_filter_np(y4, smooth_kernel, pad=True)

        zcross = 0
        for i in range(max(0, int(dd_profile.size - 1))):
            if float(dd_profile[i + 1]) > 0.0 and float(dd_profile[i]) < 0.0:
                zcross = int(i)
                break

        ddmax = 0
        for i in range(1, max(1, int(dd_profile.size - 1))):
            if i < zcross:
                continue
            if float(dd_profile[i + 1]) <= float(dd_profile[i]) and float(dd_profile[i - 1]) <= float(dd_profile[i]):
                ddmax = int(i)
                break
        if ddmax == 0:
            ddmax = int(w0)
        return int(ddmax)

    if gray.dtype.is_floating_point:
        g = gray.clone()
        if g.max() > 1.5:
            g = g / 255.0
    else:
        g = gray.to(torch.float32) / 255.0

    H, W = int(g.shape[-2]), int(g.shape[-1])

    # 1) cast to scalars BEFORE any `if`
    x1 = float(box[0]); y1 = float(box[1]); x2 = float(box[2]); y2 = float(box[3])

    # 2) expand bounding boxes by 20% for purpose of determining if cutoff
    bw = abs(x2 - x1); bh = abs(y2 - y1)
    dx20 = bw * 0.20; dy20 = bh * 0.20
    x1_20 = x1 - dx20/2; x2_20 = x2 + dx20/2
    y1_20 = y1 - dy20/2; y2_20 = y2 + dy20/2

    cutoff_flag = int(
        x1_20 < 0.0 or y1_20 < 0.0 or
        x2_20 > float(W) or y2_20 > float(H)
    )
    # 3) expand the bounding boxes by 10% for downstream analysis
    dx = abs(x2 - x1) * 0.1
    dy = abs(y2 - y1) * 0.1
    x1 -= dx/2; x2 += dx/2
    y1 -= dy/2; y2 += dy/2

    # 4) clamp the EXPANDED coords, then ints for slicing
    x1 = max(0.0, min(x1, W - 1.0))
    y1 = max(0.0, min(y1, H - 1.0))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))
    if x2 <= x1: x2 = min(float(W), x1 + 1.0)
    if y2 <= y1: y2 = min(float(H), y1 + 1.0)

    import math
    x1 = int(x1); y1 = int(y1)
    x2 = int(math.ceil(x2)); y2 = int(math.ceil(y2))

    crop = g[y1:y2, x1:x2].contiguous()
    if crop.numel() == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return (
            torch.full((3,), float("nan"), dtype=crop.dtype, device=crop.device),
            create_comet_profile(idx, x1, y1, x2, y2, float(score), empty=1),
        )

    # Keep head finding and profile analysis on an explicit grayscale crop.
    I = _to_grayscale_unit(crop)

    using_predicted_mask = use_masks and mask is not None
    if using_predicted_mask:
        coords = find_comet_pixels_from_mask(mask, y1, y2, x1, x2)
    else:
        coords = find_comet_pixels(I)  # [N,2] (y,x) within the crop
    # Ensure coords live on the same device as the intensity image before indexing
    coords = coords.to(I.device)

    if coords.numel() == 0:
        return (
            torch.full((3,), float("nan"), dtype=crop.dtype, device=crop.device),
            create_comet_profile(idx, x1, y1, x2, y2, float(score), empty=1),
        )
    else:
        # Normalize ROI to OpenComet-style convex hull before head finding/metrics.
        comet_mask = torch.zeros_like(I, dtype=torch.bool)
        ys_raw = coords[:, 0].clamp(0, I.shape[0] - 1)
        xs_raw = coords[:, 1].clamp(0, I.shape[1] - 1)
        comet_mask[ys_raw, xs_raw] = True
        comet_mask_np = comet_mask.detach().cpu().numpy().astype(bool)
        if bool(comet_mask_np.any()):
            comet_mask_np = convex_hull_image(comet_mask_np).astype(bool)
        ys_np, xs_np = np.where(comet_mask_np)
        ys = torch.from_numpy(ys_np).to(device=I.device, dtype=torch.long)
        xs = torch.from_numpy(xs_np).to(device=I.device, dtype=torch.long)
        vals = I[ys, xs]  # [N]

        # OpenComet setupHead mirrored in Python (AUTO mode):
        # Stage 1: brightest-region head + geometric validity checks.
        # Stage 2: if invalid, profile-based head edge fallback.
        head_mask_np = None
        if bool(comet_mask_np.any()):
            I_np = I.detach().cpu().numpy()
            x0_roi = int(xs_np.min())
            x1_roi = int(xs_np.max())
            y0_roi = int(ys_np.min())
            y1_roi = int(ys_np.max())
            bw_roi = int(x1_roi - x0_roi + 1)
            bh_roi = int(y1_roi - y0_roi + 1)
            roi_bounds = (x0_roi, y0_roi, bw_roi, bh_roi)

            # Circularity from normalized comet ROI.
            roi_props = regionprops(comet_mask_np.astype(np.uint8))
            circularity = 1.0
            if roi_props:
                p = roi_props[0]
                perim = float(p.perimeter)
                area = float(p.area)
                if perim > 1e-8:
                    circularity = float((4.0 * np.pi * area) / (perim * perim))

            comet_u8 = np.clip(np.rint(I_np * 255.0), 0, 255).astype(np.uint8)
            mask_crop = comet_mask_np[y0_roi:y1_roi + 1, x0_roi:x1_roi + 1]
            ip_comet = np.zeros_like(mask_crop, dtype=np.uint8)
            ip_comet[mask_crop] = comet_u8[y0_roi:y1_roi + 1, x0_roi:x1_roi + 1][mask_crop]
            hist = np.bincount(ip_comet[mask_crop].astype(np.int64), minlength=256) if bool(mask_crop.any()) else np.zeros((256,), dtype=np.int64)
            threshbin = _get_local_thresh(hist, n_bins=255, percent=0.95)

            # Brightest-region thresholding (binary) within comet ROI crop.
            ip_comet_bin = (ip_comet >= int(threshbin)).astype(np.uint8) * 255
            _, _, b_w, b_h = _binary_bound_rect(ip_comet_bin > 0)
            head_valid = True
            if float(circularity) < 0.9 and int(b_w) > int(2 * b_h):
                head_valid = False

            white_ys, white_xs = np.where(ip_comet_bin > 0)
            if white_xs.size == 0:
                head_valid = False
                xc_local = 0
            else:
                # OpenComet uses binary brightest-region center-of-mass.
                xc_local = int(float(np.mean(white_xs)))
                yc_local = int(float(np.mean(white_ys)))
                xc_local = int(max(0, min(mask_crop.shape[1] - 1, xc_local)))
                yc_local = int(max(0, min(mask_crop.shape[0] - 1, yc_local)))
                head_radius = int(_head_height(mask_crop, xc_local) // 2)
                head_gap = int(xc_local - head_radius)
                if head_gap > 0:
                    head_valid = False

                head_x = int(x0_roi)
                head_center_y = int(_front_centroid(comet_mask_np, x0_roi, y0_roi, bw_roi, bh_roi))
                cx = float(head_x + head_radius)
                cy = float(head_center_y)
                yy, xx = np.ogrid[:int(I.shape[0]), :int(I.shape[1])]
                roi_head_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(head_radius ** 2)
                candidate_head = roi_head_circle & comet_mask_np
                h_w, h_h = _mask_width_height(candidate_head)
                if int(h_w) * int(h_h) == 0:
                    head_valid = False

                if head_valid:
                    head_mask_np = candidate_head

            # AUTO fallback to profile-based head.
            if not head_valid:
                head_edge = int(_head_edge_opencomet(I_np, comet_mask_np, roi_bounds, circularity))
                head_radius = int(head_edge // 2)
                head_x = int(x0_roi)
                head_center_y = int(_front_centroid(comet_mask_np, x0_roi, y0_roi, bw_roi, bh_roi))
                cx = float(head_x + head_radius)
                cy = float(head_center_y)
                yy, xx = np.ogrid[:int(I.shape[0]), :int(I.shape[1])]
                roi_head_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(head_radius ** 2)
                candidate_head = roi_head_circle & comet_mask_np
                if bool(candidate_head.any()):
                    head_mask_np = candidate_head

        # OpenComet-style per-column background correction: estimate a strip above
        # (or below) the comet, smooth the column averages, and subtract by x.
        x_min = int(xs.min().item())
        x_max = int(xs.max().item())
        y_min = int(ys.min().item())
        y_max = int(ys.max().item())
        comet_height = max(1, y_max - y_min + 1)
        bg_height = max(int(comet_height / 5), 10)
        bg_height = min(bg_height, int(I.shape[0]))

        if y_min - bg_height >= 0:
            bg_y0 = y_min - bg_height
        else:
            bg_y0 = y_max + 1
            bg_y0 = min(bg_y0, int(I.shape[0]) - bg_height)
            bg_y0 = max(0, bg_y0)
        bg_y1 = min(int(I.shape[0]), bg_y0 + bg_height)

        bg_by_x = torch.zeros(int(I.shape[1]), dtype=I.dtype, device=I.device)
        if bg_y1 > bg_y0 and x_max >= x_min:
            bg_roi = I[bg_y0:bg_y1, x_min:x_max + 1]
            if bg_roi.numel() > 0 and bg_roi.shape[0] > 0:
                bg_col = bg_roi.mean(dim=0)
                bg_col = _smooth_column_background(bg_col, radius=2)
                bg_by_x[x_min:x_max + 1] = bg_col
        I_corr = (I - bg_by_x.view(1, -1)).clamp_min(0.0)
        vals = (vals - bg_by_x[xs]).clamp_min(0.0)

        total_I = vals.sum()
        if total_I < 5.0:
            return (
                torch.full((3,), float("nan"), dtype=crop.dtype, device=crop.device),
                create_comet_profile(idx, x1, y1, x2, y2, float(score), empty=1),
            )
        profile_width = int(I.shape[1])
        major_centers = xs.to(vals.dtype) + 0.5
        major_idx = major_centers.floor().to(torch.long).clamp(0, profile_width - 1)

        axis_sum = torch.zeros(profile_width, dtype=I.dtype, device=I.device)
        axis_sum.scatter_add_(0, major_idx, vals)

    # OpenComet getColumnAvg normalizes each column by fixed profile height.
    profile_height = max(1.0, float(I.shape[0]))
    profile = axis_sum / profile_height

    # Smooth and differentiate the profile to locate the head-tail transition.
    # define Ks relative to the width of the bounding box
    width = profile.shape[0]
    # OpenComet uses integer width/10 with no odd-length constraint.
    smooth_len = max(1, int(width // 10))

    # Ks
    smooth_kernel = torch.ones(smooth_len, dtype=I.dtype, device=I.device) / float(smooth_len)
    # Kd
    diff_kernel = torch.tensor([-1.0, 1.0], dtype=I.dtype, device=I.device)

    # OpenComet sequence:
    # y1 = smooth(profile, pad)
    # y2 = diff(y1, valid)
    # y3 = smooth(y2, pad)
    # y4 = diff(y3, valid)
    # ydd = smooth(y4, pad)
    profile_smoothed = conv_opencomet(profile, smooth_kernel, pad=True)
    profile_d = conv_opencomet(profile_smoothed, diff_kernel, pad=False)
    profile_d_smoothed = conv_opencomet(profile_d, smooth_kernel, pad=True)
    profile_dd_raw = conv_opencomet(profile_d_smoothed, diff_kernel, pad=False)
    profile_dd = conv_opencomet(profile_dd_raw, smooth_kernel, pad=True)

    # Border between head and tail: first maximum point in pdd after a
    # crossing from negative to positive.
    pd_cpu = profile_d.detach().cpu()
    pdd_cpu = profile_dd.detach().cpu()
    split_idx_profile = None
    for k in range(1, pdd_cpu.shape[0]):
        if pdd_cpu[k-1] < 0.0 and pdd_cpu[k] >= 0.0:
            peak_idx = None
            for j in range(k, pdd_cpu.shape[0]):
                left  = pdd_cpu[j-1] if j-1 >= 0 else pdd_cpu[j]
                right = pdd_cpu[j+1] if j+1 < pdd_cpu.shape[0] else pdd_cpu[j]
                if pdd_cpu[j] > left and pdd_cpu[j] > right:
                    peak_idx = j; break
            split_idx_profile = (int(torch.argmax(profile_dd[k:]).item()) + k) if peak_idx is None else peak_idx
            break

    # OpenComet fallback: if no transition is found, place the edge at full width.
    if split_idx_profile is None:
        split_idx_profile = int(width)

    if split_idx_profile >= width:
        split_boundary_profile = float(width) - 0.5
    else:
        split_boundary_profile = min(float(split_idx_profile) + 0.5, float(width) - 0.5)

    # OpenComet split assignment is ROI-based:
    #   head ROI = circle ∩ comet ROI
    #   tail ROI = comet ROI \ head ROI
    head_roi_np = np.zeros_like(comet_mask_np, dtype=bool)
    if head_mask_np is not None:
        head_roi_np = head_mask_np & comet_mask_np
    tail_roi_np = comet_mask_np & (~head_roi_np)

    # OpenComet-style post-correction measurements from comet/head/tail ROIs.
    I_corr_np = I_corr.detach().cpu().numpy()
    comet_area = float(comet_mask_np.sum())
    head_area = float(head_roi_np.sum())
    tail_area = float(tail_roi_np.sum())

    comet_dna = float(I_corr_np[comet_mask_np].sum()) if comet_area > 0 else 0.0
    head_dna = float(I_corr_np[head_roi_np].sum()) if head_area > 0 else 0.0
    tail_dna = float(I_corr_np[tail_roi_np].sum()) if tail_area > 0 else 0.0
    tail_percent_raw = tail_dna / (comet_dna + 1e-12)
    head_percent_raw = head_dna / (comet_dna + 1e-12)

    comet_width, _ = _mask_width_height(comet_mask_np)
    head_width, _ = _mask_width_height(head_roi_np)
    tail_width, _ = _mask_width_height(tail_roi_np)
    tail_length = float(max(int(comet_width) - int(head_width), 0))

    if head_area > 0:
        head_centroid_x = _x_intensity_centroid_opencomet(I_corr_np, head_roi_np)
    else:
        head_centroid_x = _x_intensity_centroid_opencomet(I_corr_np, comet_mask_np)

    tail_w, tail_h = _mask_width_height(tail_roi_np)
    if int(tail_w) * int(tail_h) == 0:
        tail_centroid_x = float(head_centroid_x)
    else:
        tail_centroid_x = _x_intensity_centroid_opencomet(I_corr_np, tail_roi_np)

    tail_percent = float(max(0.0, min(1.0, tail_percent_raw)))
    head_percent = float(max(0.0, min(1.0, head_percent_raw)))
    tail_moment = float(tail_length * tail_percent)
    olive_tail_moment = float(abs(float(tail_centroid_x) - float(head_centroid_x)) * tail_percent)
    measurements = {
        "comet_area": comet_area,
        "comet_length": float(comet_width),
        "comet_dna_content": comet_dna,
        "comet_average_intensity": float(comet_dna / comet_area) if comet_area > 0 else float("nan"),
        "head_area": head_area,
        "head_diameter": float(head_width),
        "head_dna_content": head_dna,
        "head_average_intensity": float(head_dna / head_area) if head_area > 0 else float("nan"),
        "head_dna_percent": head_percent,
        "tail_area": tail_area,
        "tail_length": tail_length,
        "tail_dna_content": tail_dna,
        "tail_dna_percent": tail_percent,
        "tail_moment": tail_moment,
        "olive_moment": olive_tail_moment,
        "tail_bounding_length": float(tail_width),
    }

    feat = torch.tensor(
        [
            tail_percent,
            tail_moment,
            olive_tail_moment,
        ],
        dtype=torch.float32,
    )
    return feat, create_comet_profile(
        idx,
        x1,
        y1,
        x2,
        y2,
        float(score),
        cutoff=cutoff_flag,
        tail_percent=tail_percent,
        tail_moment=tail_moment,
        olive_moment=olive_tail_moment,
        measurements=measurements,
    )


@torch.no_grad()
def compute_box_features(
    img_tensor: torch.Tensor,
    box: torch.Tensor,
    score: float,
    idx: int,
    mask: Optional[torch.Tensor] = None,
    use_masks: bool = False,
    image_id: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Return the comet feature vector and profile metadata for one detection."""
    return compute_box_ml_features(
        img_tensor,
        box,
        score,
        idx,
        mask,
        use_masks=use_masks,
        image_id=image_id,
    )


def _signed_log_compress(values, eps=1e-12):
    abs_vals = torch.abs(values) + eps
    return torch.sign(values) * torch.log10(abs_vals)


def compute_mask_features(mask):
    mask_np = mask.cpu().numpy().astype(np.uint8)

    region = regionprops(mask_np)[0]

    solidity = float(region.solidity)
    eccentricity = float(region.eccentricity)

    hu = torch.tensor(region.moments_hu, dtype=torch.float32)
    hu_compressed = _signed_log_compress(hu)

    return torch.cat((hu_compressed, torch.tensor([solidity, eccentricity], dtype=hu.dtype)))


def select_features(
    features: torch.Tensor,
    hu_1: bool = False,
    hu_2: bool = False,
    hu_3: bool = False,
    hu_4: bool = False,
    hu_5: bool = False,
    hu_6: bool = False,
    hu_7: bool = False,
    solidity: bool = False,
    eccentricity: bool = False,
) -> torch.Tensor:
    if features.shape[-1] != 9:
        raise ValueError(f"Expected feature vector of length 9, got shape {features.shape}.")

    mask = torch.tensor(
        [hu_1, hu_2, hu_3, hu_4, hu_5, hu_6, hu_7, solidity, eccentricity],
        dtype=torch.bool,
        device=features.device,
    )
    if mask.all():
        return features

    return features[..., mask]
