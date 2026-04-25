from __future__ import annotations

from typing import Dict, List, Optional

import torch


PROFILE_COORDS_IDX = 0
PROFILE_SCORE_IDX = 1
PROFILE_LABEL_IDX = 2
PROFILE_DAMAGE_CLASS_IDX = 3
PROFILE_OVERLAP_IDX = 4
PROFILE_CUTOFF_IDX = 5
PROFILE_EMPTY_IDX = 6
PROFILE_TAIL_PERCENT_IDX = 7
PROFILE_TAIL_MOMENT_IDX = 8
PROFILE_OLIVE_MOMENT_IDX = 9
PROFILE_MEASUREMENTS_IDX = 10


def iou_boxes(b1: torch.Tensor, b2: torch.Tensor) -> float:
    x_a = max(float(b1[0].item()), float(b2[0].item()))
    y_a = max(float(b1[1].item()), float(b2[1].item()))
    x_b = min(float(b1[2].item()), float(b2[2].item()))
    y_b = min(float(b1[3].item()), float(b2[3].item()))
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    area1 = max(0.0, float((b1[2] - b1[0]).item())) * max(0.0, float((b1[3] - b1[1]).item()))
    area2 = max(0.0, float((b2[2] - b2[0]).item())) * max(0.0, float((b2[3] - b2[1]).item()))
    union = area1 + area2 - inter
    return (inter / union) if union > 0 else 0.0


def assign_damage_classes(
    boxes: torch.Tensor,
    box_features: torch.Tensor,
    box_profiles: List[Dict],
    labels: torch.Tensor,
    treatment: str,
) -> List[int]:
    damage_classes: List[int] = []
    treatment_key = str(treatment).strip().lower()
    for i in range(int(boxes.shape[0])):
        profile = box_profiles[i]
        if not profile:
            continue
        key = next(iter(profile))
        entry = list(profile[key])
        if entry[PROFILE_LABEL_IDX] == 0:
            labels[i] = 0
            continue

        pct_tail_dna = float(box_features[i][0])
        if pct_tail_dna < 0.05:
            damage_class = 0
        elif pct_tail_dna < 0.12:
            damage_class = 1
        elif pct_tail_dna < 0.25:
            damage_class = 2
        elif pct_tail_dna < 0.40:
            damage_class = 3
        elif pct_tail_dna < 0.70:
            damage_class = 4
        else:
            damage_class = -1

        good = False
        if treatment_key == "baseline":
            good = damage_class in (0, 1)
        elif treatment_key in ("damage", "repair"):
            good = damage_class in (1, 2, 3)

        labels[i] = 1 if good else 0
        entry[PROFILE_DAMAGE_CLASS_IDX] = int(damage_class)
        entry[PROFILE_LABEL_IDX] = int(labels[i])
        profile[key] = tuple(entry)
        damage_classes.append(int(damage_class))
    return damage_classes


def apply_overlap_suppression(boxes: torch.Tensor, box_profiles: List[Dict], labels: torch.Tensor) -> None:
    for i in range(int(boxes.shape[0])):
        for j in range(i + 1, int(boxes.shape[0])):
            iou = iou_boxes(boxes[i], boxes[j])
            if 0.0 < iou < 0.5:
                labels[i] = 0
                labels[j] = 0
                _mark_profile_rejected(box_profiles[i], PROFILE_OVERLAP_IDX)
                _mark_profile_rejected(box_profiles[j], PROFILE_OVERLAP_IDX)


def _mark_profile_rejected(profile: Dict, reason_idx: int) -> None:
    if not profile:
        return
    key = next(iter(profile))
    entry = list(profile[key])
    entry[PROFILE_LABEL_IDX] = 0
    entry[reason_idx] = 1
    profile[key] = tuple(entry)


def filter_enclosed_boxes(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    num = int(boxes.shape[0])
    if num == 0:
        return torch.zeros((0,), dtype=torch.long)
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    keep = torch.ones((num,), dtype=torch.bool)
    for i in range(num):
        if not keep[i]:
            continue
        for j in range(num):
            if i == j or not keep[i]:
                continue
            inside = (
                boxes[i, 0] >= boxes[j, 0]
                and boxes[i, 1] >= boxes[j, 1]
                and boxes[i, 2] <= boxes[j, 2]
                and boxes[i, 3] <= boxes[j, 3]
            )
            if not inside:
                continue
            if areas[i] < areas[j] or (areas[i] == areas[j] and scores.numel() == num and scores[i] < scores[j]):
                keep[i] = False
                break
    return torch.where(keep)[0]


def profile_damage_class(profile: Dict) -> Optional[int]:
    if not profile:
        return None
    entry = profile.get(next(iter(profile)))
    if not isinstance(entry, tuple) or len(entry) <= PROFILE_DAMAGE_CLASS_IDX:
        return None
    try:
        return int(entry[PROFILE_DAMAGE_CLASS_IDX])
    except (TypeError, ValueError):
        return None
