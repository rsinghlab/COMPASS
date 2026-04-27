from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .datasets import ImageFolderDataset, collate_images
from .features import compute_box_features
from .models import image_size_for_model, load_model, parse_imgsz, predict_batch
from .paths import default_weight_path, resolve_output_dir
from .scoring import (
    PROFILE_COORDS_IDX,
    PROFILE_CUTOFF_IDX,
    PROFILE_DAMAGE_CLASS_IDX,
    PROFILE_EMPTY_IDX,
    PROFILE_LABEL_IDX,
    PROFILE_MEASUREMENTS_IDX,
    PROFILE_OLIVE_MOMENT_IDX,
    PROFILE_OVERLAP_IDX,
    PROFILE_SCORE_IDX,
    PROFILE_TAIL_MOMENT_IDX,
    PROFILE_TAIL_PERCENT_IDX,
    apply_overlap_suppression,
    assign_damage_classes,
    filter_enclosed_boxes,
)
from .viz import save_prediction_overlay
from .workbook import (
    DETECTION_COLUMNS,
    MEASUREMENT_COLUMNS,
    SUMMARY_COLUMNS,
    finite_or_none,
    summarize_image_rows,
    write_csv,
    write_measurements_xlsx,
)


BOX_FEATURE_DIM = 3


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    model: str = "yolo"
    weights: Optional[Path] = None
    treatment: str = "infer"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_images: Optional[int] = None
    batch_size: int = 1
    num_workers: int = 0
    yolo_conf: float = 0.3
    yolo_iou: Optional[float] = 0.5
    yolo_max_det: Optional[int] = None
    yolo_agnostic_nms: bool = True
    yolo_imgsz: Optional[str] = "960"
    retina_masks: bool = True
    mrcnn_conf: float = 0.05
    mask_thresh: float = 0.5


def build_config(args) -> PipelineConfig:
    model = str(args.model).strip().lower()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = resolve_output_dir(args.output, model, input_dir)
    arg_weights = getattr(args, "weights", None)
    weights = Path(arg_weights).expanduser().resolve() if arg_weights else default_weight_path(model)
    return PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        weights=weights,
        treatment=str(args.treatment).strip().lower(),
        device=str(args.device),
        max_images=args.max_images,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        yolo_conf=float(args.yolo_conf),
        yolo_iou=args.yolo_iou,
        yolo_max_det=args.yolo_max_det,
        yolo_agnostic_nms=bool(args.yolo_agnostic_nms),
        yolo_imgsz=args.yolo_imgsz,
        retina_masks=bool(args.retina_masks),
        mrcnn_conf=float(args.mrcnn_conf),
        mask_thresh=float(args.mask_thresh),
    )


def run_pipeline(config: PipelineConfig) -> Path:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(config.output_dir / "run.log")
    logger.info("Starting COMPASS")
    logger.info("Input: %s", config.input_dir)
    logger.info("Output: %s", config.output_dir)
    logger.info("Model: %s", config.model)
    logger.info("Weights: %s", config.weights)

    dataset = ImageFolderDataset(
        config.input_dir,
        size_hw=image_size_for_model(config.model),
        treatment=config.treatment,
        max_images=config.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_images,
    )
    model_runner = load_model(config.model, str(config.weights), config.device)
    yolo_imgsz = parse_imgsz(config.yolo_imgsz)

    per_image_rows: List[Tuple[str, List[Dict[str, object]]]] = []
    all_detection_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    overlays_dir = config.output_dir / "overlays"

    image_index = 0
    with torch.no_grad():
        for imgs, targets in loader:
            img_list = [img.detach().cpu() for img in imgs]
            preds = predict_batch(
                config.model,
                model_runner,
                img_list,
                device=config.device,
                yolo_conf=config.yolo_conf,
                yolo_iou=config.yolo_iou,
                yolo_max_det=config.yolo_max_det,
                yolo_agnostic_nms=config.yolo_agnostic_nms,
                yolo_imgsz=yolo_imgsz,
                retina_masks=config.retina_masks,
                mrcnn_conf=config.mrcnn_conf,
            )

            for img, target, pred in zip(imgs, targets, preds):
                rows, boxes, labels, profiles = _process_prediction(img.detach().cpu(), target, pred, config)
                image_name = str(target.get("file_name", f"image_{image_index:05d}"))
                per_image_rows.append((image_name, rows))
                all_detection_rows.extend(rows)
                summary_rows.append(_summarize_target_rows(target, image_name, rows))
                save_prediction_overlay(img.detach().cpu(), boxes, labels, target, overlays_dir, profiles)
                logger.info(
                    "%s: detections=%d selected_good=%d",
                    target.get("relative_path", image_name),
                    len(rows),
                    sum(1 for row in rows if row.get("good_bad") == "Good"),
                )
                image_index += 1

    write_measurements_xlsx(config.output_dir / "measurements.xlsx", per_image_rows)
    write_csv(config.output_dir / "detections.csv", all_detection_rows, DETECTION_COLUMNS)
    write_csv(config.output_dir / "run_summary.csv", summary_rows, SUMMARY_COLUMNS)
    _write_run_config(config)
    logger.info("Finished COMPASS")
    return config.output_dir


def _validate_config(config: PipelineConfig) -> None:
    if config.model not in {"yolo", "sam", "mrcnn"}:
        raise ValueError("Model must be one of: yolo, sam, mrcnn")
    if config.treatment not in {"infer", "baseline", "damage", "repair"}:
        raise ValueError("Treatment must be one of: infer, baseline, damage, repair")
    if config.weights is None or not config.weights.is_file():
        raise FileNotFoundError(
            f"Weight file not found: {config.weights}\n"
            "Download the pretrained weights into $COMPASS_HOME/weights before running COMPASS."
        )


def _process_prediction(
    img: torch.Tensor,
    target: Dict[str, object],
    pred: Dict[str, torch.Tensor],
    config: PipelineConfig,
) -> Tuple[List[Dict[str, object]], torch.Tensor, torch.Tensor, List[Dict]]:
    boxes = pred.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().to(torch.float32)
    scores = pred.get("scores", torch.zeros((int(boxes.shape[0]),), dtype=torch.float32)).detach().cpu().to(torch.float32)
    labels = pred.get("labels", torch.zeros((int(boxes.shape[0]),), dtype=torch.int64)).detach().cpu().to(torch.int64)
    masks = _resize_masks(pred.get("masks"), int(img.shape[-2]), int(img.shape[-1]))
    masks_bin = masks >= float(config.mask_thresh)

    original_count = int(boxes.shape[0])
    keep_idx = filter_enclosed_boxes(boxes, scores)
    if keep_idx.numel() != original_count:
        boxes = boxes[keep_idx]
        scores = scores[keep_idx] if scores.numel() == original_count else scores
        labels = labels[keep_idx] if labels.numel() == original_count else labels
        masks_bin = masks_bin[keep_idx] if int(masks_bin.shape[0]) == original_count else masks_bin

    num_detections = int(boxes.shape[0])
    if labels.numel() != num_detections:
        labels = torch.ones((num_detections,), dtype=torch.int64)
    if scores.numel() != num_detections:
        scores = torch.zeros((num_detections,), dtype=torch.float32)

    box_features = torch.zeros((num_detections, BOX_FEATURE_DIM), dtype=torch.float32)
    box_profiles: List[Dict] = [{} for _ in range(num_detections)]

    for det_idx in range(num_detections):
        mask_for_box = None
        if masks_bin.numel() > 0 and det_idx < int(masks_bin.shape[0]):
            mask_for_box = masks_bin[det_idx].squeeze(0).to(torch.float32)
        feature_vec, profile = compute_box_features(
            img,
            boxes[det_idx],
            float(scores[det_idx]),
            det_idx,
            mask=mask_for_box,
            use_masks=mask_for_box is not None,
            image_id=str(target.get("file_name", "")),
        )
        if feature_vec.numel() == BOX_FEATURE_DIM:
            box_features[det_idx] = feature_vec.to(torch.float32)
        box_profiles[det_idx] = profile

    treatment = str(target.get("treatment", "baseline"))
    assign_damage_classes(boxes, box_features, box_profiles, labels, treatment)
    apply_overlap_suppression(boxes, box_profiles, labels)
    rows = _measurement_rows(target, boxes, labels, scores, box_profiles)
    return rows, boxes, labels, box_profiles


def _resize_masks(masks: Optional[torch.Tensor], out_h: int, out_w: int) -> torch.Tensor:
    if masks is None:
        return torch.zeros((0, 1, out_h, out_w), dtype=torch.float32)
    m = masks.detach().cpu().to(torch.float32)
    if m.ndim == 3:
        m = m.unsqueeze(1)
    if m.ndim != 4 or m.shape[1] != 1:
        return torch.zeros((0, 1, out_h, out_w), dtype=torch.float32)
    if m.numel() == 0 or m.shape[-2] == 0 or m.shape[-1] == 0:
        return torch.zeros((int(m.shape[0]), 1, out_h, out_w), dtype=torch.float32)
    if int(m.shape[-2]) == out_h and int(m.shape[-1]) == out_w:
        return m
    return F.interpolate(m, size=(out_h, out_w), mode="bilinear", align_corners=False)


def _measurement_rows(
    target: Dict[str, object],
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    box_profiles: List[Dict],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for det_idx in range(int(boxes.shape[0])):
        entry = _profile_entry(box_profiles[det_idx]) if det_idx < len(box_profiles) else None
        label_value = int(labels[det_idx].item()) if det_idx < labels.numel() else 0
        score_value = float(scores[det_idx].item()) if det_idx < scores.numel() else 0.0
        coords = boxes[det_idx].detach().cpu().tolist()
        damage_class = None
        overlap = 0
        cutoff = 0
        empty = 0
        measurements: Dict[str, object] = {}
        if entry is not None:
            label_value = int(entry[PROFILE_LABEL_IDX])
            score_value = float(entry[PROFILE_SCORE_IDX])
            coords = list(entry[PROFILE_COORDS_IDX])
            damage_class = int(entry[PROFILE_DAMAGE_CLASS_IDX])
            overlap = int(entry[PROFILE_OVERLAP_IDX])
            cutoff = int(entry[PROFILE_CUTOFF_IDX])
            empty = int(entry[PROFILE_EMPTY_IDX])
            if len(entry) > PROFILE_MEASUREMENTS_IDX and isinstance(entry[PROFILE_MEASUREMENTS_IDX], dict):
                measurements = dict(entry[PROFILE_MEASUREMENTS_IDX])
            else:
                measurements = {
                    "tail_dna_percent": float(entry[PROFILE_TAIL_PERCENT_IDX]),
                    "tail_moment": float(entry[PROFILE_TAIL_MOMENT_IDX]),
                    "olive_moment": float(entry[PROFILE_OLIVE_MOMENT_IDX]),
                }

        row: Dict[str, object] = {
            "image_name": str(target.get("file_name", "")),
            "relative_path": str(target.get("relative_path", "")),
            "image_path": str(target.get("image_path", "")),
            "treatment": str(target.get("treatment", "")),
            "detection_index": int(det_idx),
            "good_bad": "Good" if label_value == 1 else "Bad",
            "score": finite_or_none(score_value),
            "damage_class": damage_class,
            "label": int(label_value),
            "overlap": int(overlap),
            "cutoff": int(cutoff),
            "empty": int(empty),
            "box_x1": float(coords[0]) if len(coords) > 0 and coords[0] is not None else None,
            "box_y1": float(coords[1]) if len(coords) > 1 and coords[1] is not None else None,
            "box_x2": float(coords[2]) if len(coords) > 2 and coords[2] is not None else None,
            "box_y2": float(coords[3]) if len(coords) > 3 and coords[3] is not None else None,
        }
        for col in MEASUREMENT_COLUMNS:
            row[col] = finite_or_none(measurements.get(col))
        rows.append(row)
    return rows


def _summarize_target_rows(
    target: Dict[str, object],
    image_name: str,
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    summary = summarize_image_rows(image_name, rows)
    summary["relative_path"] = str(target.get("relative_path", ""))
    summary["image_path"] = str(target.get("image_path", ""))
    summary["treatment"] = str(target.get("treatment", ""))
    return summary


def _profile_entry(profile: Dict) -> Optional[Tuple]:
    if not profile:
        return None
    entry = profile.get(next(iter(profile)))
    return entry if isinstance(entry, tuple) else None


def _write_run_config(config: PipelineConfig) -> None:
    payload = asdict(config)
    payload["input_dir"] = str(config.input_dir)
    payload["output_dir"] = str(config.output_dir)
    payload["weights"] = str(config.weights) if config.weights else None
    payload["timestamp_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("compass")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
