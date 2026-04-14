from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry
from typing import Dict, List, Optional, Tuple, Union

from .yolo import load_yolo_model, yolo_predict_batch


PROMPT_MIN_AREA_RATIO = 100.0 / (960.0 * 960.0)
MRCNN_NUM_CLASSES = 3


def parse_imgsz(value: Optional[str]) -> Optional[Union[int, Tuple[int, int]]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid image size '{value}'. Use a single integer or H,W.")
        return (int(parts[0]), int(parts[1]))
    return int(text)


def image_size_for_model(model_name: str) -> Tuple[int, int]:
    return (512, 720) if str(model_name).lower() == "mrcnn" else (2048, 2880)


def load_model(model_name: str, weights_path: str, device: str):
    model_key = str(model_name).strip().lower()
    if model_key == "yolo":
        return load_yolo_model(weights_path)
    if model_key == "sam":
        return _load_sam_predictor(weights_path, device)
    if model_key == "mrcnn":
        return _load_mrcnn_model(weights_path, device)
    raise ValueError(f"Unsupported model: {model_name}")


def predict_batch(
    model_name: str,
    model_runner,
    imgs_t: List[torch.Tensor],
    *,
    device: str,
    yolo_conf: float,
    yolo_iou: Optional[float],
    yolo_max_det: Optional[int],
    yolo_agnostic_nms: bool,
    yolo_imgsz: Optional[Union[int, Tuple[int, int]]],
    retina_masks: bool,
    mrcnn_conf: float,
) -> List[Dict[str, torch.Tensor]]:
    model_key = str(model_name).strip().lower()
    if model_key == "yolo":
        return yolo_predict_batch(
            model_runner,
            imgs_t,
            conf=float(yolo_conf),
            iou=yolo_iou,
            max_det=yolo_max_det,
            agnostic_nms=bool(yolo_agnostic_nms),
            device=device,
            imgsz=yolo_imgsz,
            retina_masks=bool(retina_masks),
        )
    if model_key == "sam":
        return sam_predict_batch(model_runner, imgs_t)
    if model_key == "mrcnn":
        return mrcnn_predict_batch(model_runner, imgs_t, device=device, conf=float(mrcnn_conf))
    raise ValueError(f"Unsupported model: {model_name}")


def _load_sam_predictor(checkpoint_path: str, device: str) -> SamPredictor:
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    return SamPredictor(sam)


def _prompt_points_from_image(image_gray: np.ndarray) -> List[np.ndarray]:
    h, w = image_gray.shape
    min_contour_area = PROMPT_MIN_AREA_RATIO * float(h * w)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prompt_points: List[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) <= min_contour_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue
        roi = image_gray[by : by + bh, bx : bx + bw]
        if roi.size == 0:
            continue
        _, _, _, max_loc = cv2.minMaxLoc(roi)
        head_pt = [bx + max_loc[0], by + max_loc[1]]
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            body_pt = [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]
        else:
            body_pt = head_pt
        prompt_points.append(np.asarray([head_pt, body_pt], dtype=np.float32))
    return prompt_points


def _boxes_from_masks_np(masks_bool: np.ndarray) -> np.ndarray:
    boxes = np.zeros((int(masks_bool.shape[0]), 4), dtype=np.float32)
    for i in range(int(masks_bool.shape[0])):
        ys, xs = np.where(masks_bool[i])
        if xs.size == 0 or ys.size == 0:
            continue
        boxes[i] = np.asarray([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    return boxes


def sam_predict_batch(predictor: SamPredictor, imgs_t: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    preds: List[Dict[str, torch.Tensor]] = []
    imgs_np = [
        (img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        for img in imgs_t
    ]

    for img_np in imgs_np:
        image_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        predictor.set_image(img_np)
        pred_masks: List[np.ndarray] = []
        pred_scores: List[float] = []
        for points in _prompt_points_from_image(image_gray):
            sam_masks, iou_predictions, _ = predictor.predict(
                point_coords=points,
                point_labels=np.array([1, 1], dtype=np.int32),
                multimask_output=False,
            )
            if sam_masks is None or len(sam_masks) == 0:
                continue
            mask_bool = sam_masks[0] > 0
            if not bool(mask_bool.any()):
                continue
            pred_masks.append(mask_bool.astype(np.float32))
            score = float(iou_predictions[0]) if iou_predictions is not None and len(iou_predictions) else 0.0
            pred_scores.append(score)

        if not pred_masks:
            preds.append(empty_prediction())
            continue
        masks_np = np.stack(pred_masks, axis=0)
        boxes_np = _boxes_from_masks_np(masks_np > 0)
        preds.append(
            {
                "boxes": torch.from_numpy(boxes_np).to(torch.float32),
                "scores": torch.from_numpy(np.asarray(pred_scores, dtype=np.float32)),
                "labels": torch.ones((masks_np.shape[0],), dtype=torch.int64),
                "masks": torch.from_numpy(masks_np[:, None, :, :]).to(torch.float32),
            }
        )
    return preds


def _build_deepcomet_maskrcnn(num_classes: int) -> torch.nn.Module:
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    return model


def _load_mrcnn_model(checkpoint_path: str, device: str):
    model = _build_deepcomet_maskrcnn(num_classes=MRCNN_NUM_CLASSES)
    state_obj = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = state_obj["model_state"] if isinstance(state_obj, dict) and "model_state" in state_obj else state_obj
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported Mask R-CNN checkpoint format: {checkpoint_path}")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if all(str(k).startswith("module.") for k in state_dict.keys()):
            model.load_state_dict({str(k)[7:]: v for k, v in state_dict.items()}, strict=True)
        else:
            raise
    if hasattr(model, "rpn") and hasattr(model.rpn, "nms_thresh"):
        model.rpn.nms_thresh = 0.5
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "nms_thresh"):
        model.roi_heads.nms_thresh = 0.2
    model.to(device=device)
    model.eval()
    return model


@torch.no_grad()
def mrcnn_predict_batch(mrcnn_model, imgs_t: List[torch.Tensor], *, device: str, conf: float):
    preds_device = mrcnn_model([img.to(device) for img in imgs_t])
    preds: List[Dict[str, torch.Tensor]] = []
    for pred in preds_device:
        boxes = pred.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().to(torch.float32)
        scores = pred.get("scores", torch.zeros((int(boxes.shape[0]),), dtype=torch.float32)).detach().cpu()
        labels = pred.get("labels", torch.zeros((int(boxes.shape[0]),), dtype=torch.int64)).detach().cpu()
        masks = pred.get("masks", torch.zeros((int(boxes.shape[0]), 1, 0, 0), dtype=torch.float32)).detach().cpu()
        if float(conf) > 0.0 and int(boxes.shape[0]) > 0:
            keep = scores >= float(conf)
            boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
        preds.append({"boxes": boxes, "scores": scores, "labels": labels, "masks": masks})
    return preds


def empty_prediction() -> Dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, 1, 0, 0), dtype=torch.float32),
    }
