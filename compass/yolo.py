"""
Ultralytics YOLO helpers for JRComet.

This is a minimal wrapper used by `src/run.py` to obtain:
  - boxes (xyxy)
  - scores
  - class ids (mapped 0/1 -> 1/2 to match the historical AB code)
  - instance masks (if the loaded weights are a -seg model)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import time 
import numpy as np
import torch

MASK_CROP_INFLATE = 0.20
_MASK_CROP_INFLATE = float(MASK_CROP_INFLATE)
_MASK_CROP_PATCHED = False
_ORIG_PROCESS_MASK = None
_ORIG_PROCESS_MASK_NATIVE = None


def _inflate_boxes(bboxes: torch.Tensor, shape: Tuple[int, int], inflate: float) -> torch.Tensor:
    if bboxes is None or bboxes.numel() == 0 or inflate <= 0.0:
        return bboxes
    h = float(shape[0])
    w = float(shape[1])
    if h <= 1.0 or w <= 1.0:
        return bboxes
    boxes = bboxes.to(torch.float32)
    x1, y1, x2, y2 = boxes.unbind(1)
    bw = (x2 - x1).clamp(min=0.0)
    bh = (y2 - y1).clamp(min=0.0)
    dx = bw * inflate
    dy = bh * inflate
    x1 = x1 - dx / 2.0
    y1 = y1 - dy / 2.0
    x2 = x2 + dx / 2.0
    y2 = y2 + dy / 2.0
    x1 = x1.clamp(0.0, w - 1.0)
    y1 = y1.clamp(0.0, h - 1.0)
    x2 = x2.clamp(0.0, w)
    y2 = y2.clamp(0.0, h)
    x2 = torch.max(x2, x1 + 1.0)
    y2 = torch.max(y2, y1 + 1.0)
    x2 = x2.clamp(0.0, w)
    y2 = y2.clamp(0.0, h)
    return torch.stack([x1, y1, x2, y2], dim=1).to(bboxes.dtype)


def _patch_ultralytics_mask_cropping(inflate: float) -> None:
    global _MASK_CROP_PATCHED, _MASK_CROP_INFLATE, _ORIG_PROCESS_MASK, _ORIG_PROCESS_MASK_NATIVE
    _MASK_CROP_INFLATE = float(inflate)
    if _MASK_CROP_PATCHED:
        return
    from ultralytics.utils import ops  # type: ignore

    _ORIG_PROCESS_MASK = ops.process_mask
    _ORIG_PROCESS_MASK_NATIVE = ops.process_mask_native

    def _process_mask_patched(protos, masks_in, bboxes, shape, upsample: bool = False):
        if _MASK_CROP_INFLATE > 0.0:
            bboxes = _inflate_boxes(bboxes, shape, _MASK_CROP_INFLATE)
        return _ORIG_PROCESS_MASK(protos, masks_in, bboxes, shape, upsample=upsample)

    def _process_mask_native_patched(protos, masks_in, bboxes, shape):
        if _MASK_CROP_INFLATE > 0.0:
            bboxes = _inflate_boxes(bboxes, shape, _MASK_CROP_INFLATE)
        return _ORIG_PROCESS_MASK_NATIVE(protos, masks_in, bboxes, shape)

    ops.process_mask = _process_mask_patched
    ops.process_mask_native = _process_mask_native_patched
    _MASK_CROP_PATCHED = True


def load_yolo_model(weights_path: str):
    from ultralytics import YOLO  # type: ignore

    w = os.path.expanduser(weights_path)
    if not os.path.isfile(w):
        raise FileNotFoundError(f"YOLO weights not found at {w}")
    return YOLO(w)


def yolo_predict_batch(
    yolo_model,
    imgs_t: List[torch.Tensor],
    conf: float = 0.25,
    iou: Optional[float] = None,
    max_det: Optional[int] = None,
    agnostic_nms: Optional[bool] = None,
    device: Optional[str] = None,
    imgsz: Optional[Union[int, Tuple[int, int]]] = None,
    retina_masks: Optional[bool] = None,
) -> List[Dict[str, torch.Tensor]]:
    preds: List[Dict[str, torch.Tensor]] = []

    _patch_ultralytics_mask_cropping(_MASK_CROP_INFLATE)

    imgs_np = [
        (img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8) for img in imgs_t
    ]
    predict_kwargs = {"conf": float(conf), "verbose": False}
    if iou is not None:
        predict_kwargs["iou"] = float(iou)
    if max_det is not None and int(max_det) > 0:
        predict_kwargs["max_det"] = int(max_det)
    if agnostic_nms is not None:
        predict_kwargs["agnostic_nms"] = bool(agnostic_nms)
    if device:
        predict_kwargs["device"] = device
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    if retina_masks is not None:
        predict_kwargs["retina_masks"] = bool(retina_masks)

    start_time = time.perf_counter()
    results = yolo_model.predict(imgs_np, **predict_kwargs)
    # print("inference time: ", time.perf_counter() - start_time)

    for idx, res in enumerate(results):
        if res is None or getattr(res, "boxes", None) is None or len(res.boxes) == 0:
            preds.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "masks": torch.zeros((0, 1, 0, 0), dtype=torch.float32),
                }
            )
            continue

        boxes = res.boxes.xyxy.detach().cpu().to(torch.float32)
        if _MASK_CROP_INFLATE > 0.0 and boxes.numel() > 0:
            shape = getattr(res, "orig_shape", None)
            if not shape:
                img_np = imgs_np[idx]
                shape = (int(img_np.shape[0]), int(img_np.shape[1]))
            boxes = _inflate_boxes(boxes, shape, _MASK_CROP_INFLATE)
        scores = res.boxes.conf.detach().cpu().to(torch.float32)
        labels = (res.boxes.cls.detach().cpu().to(torch.int64) + 1)  # map 0/1 -> 1/2 (Non-Ghost/Ghost)

        out: Dict[str, torch.Tensor] = {"boxes": boxes, "scores": scores, "labels": labels}

        masks = getattr(res, "masks", None)
        if masks is not None and getattr(masks, "data", None) is not None:
            m = masks.data.detach().cpu().to(torch.float32)
            if m.ndim == 3:
                m = m.unsqueeze(1)  # [N,1,H,W]
            out["masks"] = m
        else:
            out["masks"] = torch.zeros((int(boxes.shape[0]), 1, 0, 0), dtype=torch.float32)

        preds.append(out)

    return preds
