from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch

from .viz import render_prediction_overlay


MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 1000
MIN_MANUAL_BOX_SIZE = 5.0
CLICK_DRAG_TOLERANCE = 4.0
WINDOW_NAME = "COMPASS interactive review"
PREVIEW_BOX_COLOR_BGR = (255, 128, 0)
PREVIEW_DASH_LENGTH = 8
PREVIEW_GAP_LENGTH = 6


class ReviewAborted(RuntimeError):
    pass


@dataclass
class ReviewEdits:
    deselected_indices: Set[int]
    manual_boxes: torch.Tensor
    selected_indices: Set[int] = field(default_factory=set)


def assert_interactive_review_available() -> None:
    if sys.platform.startswith("linux"):
        has_display = any(
            os.environ.get(name)
            for name in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
        )
        if not has_display:
            raise RuntimeError(
                "--interactive-review needs an available graphical display. "
                "Run COMPASS locally, use X11/Wayland forwarding, or omit --interactive-review for batch jobs."
            )


def review_detections(
    img_tensor: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    target: dict,
    box_profiles: Optional[List[dict]] = None,
) -> ReviewEdits:
    assert_interactive_review_available()
    boxes_cpu = boxes.detach().cpu().to(torch.float32) if boxes is not None else torch.zeros((0, 4))
    labels_cpu = labels.detach().cpu().to(torch.int64) if labels is not None else torch.zeros((0,), dtype=torch.int64)
    if labels_cpu.numel() < int(boxes_cpu.shape[0]):
        labels_cpu = torch.cat(
            [labels_cpu, torch.zeros((int(boxes_cpu.shape[0]) - labels_cpu.numel(),), dtype=torch.int64)]
        )
    labels_cpu = labels_cpu[: int(boxes_cpu.shape[0])]

    h = int(img_tensor.shape[-2])
    w = int(img_tensor.shape[-1])
    state = _ReviewState(boxes_cpu, labels_cpu, h, w)
    scale = min(1.0, MAX_DISPLAY_WIDTH / max(1, w), MAX_DISPLAY_HEIGHT / max(1, h))

    image_name = str(target.get("relative_path") or target.get("file_name") or "image")
    print(
        f"Reviewing {image_name}: click model boxes to flip selected/rejected state, drag to add a box, "
        "right-click a manual box to remove it, u undo, r reset, Enter/Space accept, q abort.",
        flush=True,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, state.on_mouse)
    try:
        while True:
            frame = state.render(img_tensor, box_profiles=box_profiles)
            display = _scale_frame(frame, scale)
            if state.drag_start is not None and state.drag_current is not None:
                x1, y1, x2, y2 = _normalize_box((*state.drag_start, *state.drag_current), w, h)
                pt1 = (int(round(x1 * scale)), int(round(y1 * scale)))
                pt2 = (int(round(x2 * scale)), int(round(y2 * scale)))
                _draw_dotted_rectangle(display, pt1, pt2, PREVIEW_BOX_COLOR_BGR, 2)
            state.display_scale = scale
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):
                manual = (
                    torch.tensor(state.manual_boxes, dtype=torch.float32)
                    if state.manual_boxes
                    else torch.zeros((0, 4), dtype=torch.float32)
                )
                return ReviewEdits(
                    deselected_indices=set(state.deselected_indices),
                    manual_boxes=manual,
                    selected_indices=set(state.selected_indices),
                )
            if key in (ord("q"), ord("Q"), 27):
                raise ReviewAborted(f"Interactive review aborted for {image_name}.")
            if key in (ord("u"), ord("U")):
                state.undo()
            if key in (ord("r"), ord("R")):
                state.reset()
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
        cv2.destroyWindow(WINDOW_NAME)


class _ReviewState:
    def __init__(self, boxes: torch.Tensor, labels: torch.Tensor, height: int, width: int) -> None:
        self.boxes = boxes
        self.labels = labels
        self.height = int(height)
        self.width = int(width)
        self.deselected_indices: Set[int] = set()
        self.selected_indices: Set[int] = set()
        self.manual_boxes: List[Tuple[float, float, float, float]] = []
        self.history: List[Tuple[Set[int], Set[int], List[Tuple[float, float, float, float]]]] = []
        self.drag_start: Optional[Tuple[float, float]] = None
        self.drag_current: Optional[Tuple[float, float]] = None
        self.display_scale = 1.0

    def snapshot(self) -> None:
        self.history.append((set(self.deselected_indices), set(self.selected_indices), list(self.manual_boxes)))

    def undo(self) -> None:
        if not self.history:
            return
        deselected, selected, manual = self.history.pop()
        self.deselected_indices = deselected
        self.selected_indices = selected
        self.manual_boxes = manual

    def reset(self) -> None:
        self.snapshot()
        self.deselected_indices.clear()
        self.selected_indices.clear()
        self.manual_boxes.clear()

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        image_x = float(x) / max(self.display_scale, 1e-6)
        image_y = float(y) / max(self.display_scale, 1e-6)
        image_x = max(0.0, min(float(self.width - 1), image_x))
        image_y = max(0.0, min(float(self.height - 1), image_y))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (image_x, image_y)
            self.drag_current = (image_x, image_y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (image_x, image_y)
            return
        if event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            start = self.drag_start
            current = (image_x, image_y)
            self.drag_start = None
            self.drag_current = None
            drag_distance = max(abs(current[0] - start[0]), abs(current[1] - start[1]))
            if drag_distance <= CLICK_DRAG_TOLERANCE:
                self._toggle_model_box(current[0], current[1])
                return
            box = _normalize_box((*start, *current), self.width, self.height)
            if box[2] - box[0] >= MIN_MANUAL_BOX_SIZE and box[3] - box[1] >= MIN_MANUAL_BOX_SIZE:
                self.snapshot()
                self.manual_boxes.append(box)
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self._remove_manual_box(image_x, image_y)

    def render(self, img_tensor: torch.Tensor, box_profiles: Optional[List[dict]]) -> np.ndarray:
        boxes, labels = self._display_tensors()
        rows = self._display_rows()
        profiles = _display_profiles(box_profiles, len(rows))
        overlay = render_prediction_overlay(img_tensor, boxes, labels, box_profiles=profiles, rows=rows)
        rgb = np.asarray(overlay.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _toggle_model_box(self, x: float, y: float) -> None:
        idx = _smallest_containing_box(self.boxes, x, y)
        if idx is None:
            return
        self.snapshot()
        if int(self.labels[idx].item()) == 1:
            self.selected_indices.discard(idx)
            if idx in self.deselected_indices:
                self.deselected_indices.remove(idx)
            else:
                self.deselected_indices.add(idx)
        else:
            self.deselected_indices.discard(idx)
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
            else:
                self.selected_indices.add(idx)

    def _remove_manual_box(self, x: float, y: float) -> None:
        for idx in range(len(self.manual_boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.manual_boxes[idx]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.snapshot()
                del self.manual_boxes[idx]
                return

    def _display_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        manual = (
            torch.tensor(self.manual_boxes, dtype=torch.float32)
            if self.manual_boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        boxes = torch.cat([self.boxes, manual], dim=0) if manual.numel() > 0 else self.boxes
        manual_labels = torch.ones((int(manual.shape[0]),), dtype=torch.int64)
        model_labels = self.labels.clone()
        for idx in self.deselected_indices:
            if 0 <= idx < int(model_labels.numel()):
                model_labels[idx] = 0
        for idx in self.selected_indices:
            if 0 <= idx < int(model_labels.numel()):
                model_labels[idx] = 1
        labels = torch.cat([model_labels, manual_labels], dim=0) if manual_labels.numel() > 0 else model_labels
        return boxes, labels

    def _display_rows(self) -> List[dict]:
        rows: List[dict] = []
        for idx, label in enumerate(self.labels.tolist()):
            if idx in self.selected_indices:
                review_action = "manual_selected"
                good_bad = "Good"
            elif idx in self.deselected_indices:
                review_action = "manual_deselected"
                good_bad = "Deselected"
            else:
                review_action = "none"
                good_bad = "Good" if int(label) == 1 else "Bad"
            rows.append(
                {
                    "detection_source": "model",
                    "review_action": review_action,
                    "good_bad": good_bad,
                }
            )
        for _ in self.manual_boxes:
            rows.append({"detection_source": "manual", "review_action": "manual_added", "good_bad": "Good"})
        return rows


def _display_profiles(box_profiles: Optional[List[dict]], num_rows: int) -> List[dict]:
    profiles = list(box_profiles or [])
    if len(profiles) < num_rows:
        profiles.extend({} for _ in range(num_rows - len(profiles)))
    return profiles[:num_rows]


def _normalize_box(values: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in values]
    left = max(0.0, min(x1, x2, float(width - 1)))
    top = max(0.0, min(y1, y2, float(height - 1)))
    right = max(0.0, min(max(x1, x2), float(width)))
    bottom = max(0.0, min(max(y1, y2), float(height)))
    if right <= left:
        right = min(float(width), left + 1.0)
    if bottom <= top:
        bottom = min(float(height), top + 1.0)
    return (left, top, right, bottom)


def _smallest_containing_box(boxes: torch.Tensor, x: float, y: float) -> Optional[int]:
    best_idx = None
    best_area = float("inf")
    for idx, box in enumerate(boxes.tolist()):
        x1, y1, x2, y2 = [float(v) for v in box]
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < best_area:
            best_area = area
            best_idx = int(idx)
    return best_idx


def _scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame.copy()
    width = max(1, int(round(frame.shape[1] * scale)))
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _draw_dotted_rectangle(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    left, right = sorted((int(pt1[0]), int(pt2[0])))
    top, bottom = sorted((int(pt1[1]), int(pt2[1])))
    _draw_dotted_line(image, (left, top), (right, top), color, thickness)
    _draw_dotted_line(image, (right, top), (right, bottom), color, thickness)
    _draw_dotted_line(image, (right, bottom), (left, bottom), color, thickness)
    _draw_dotted_line(image, (left, bottom), (left, top), color, thickness)


def _draw_dotted_line(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int = PREVIEW_DASH_LENGTH,
    gap_length: int = PREVIEW_GAP_LENGTH,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = float(np.hypot(dx, dy))
    if length <= 0.0:
        return

    dash = max(1, int(dash_length))
    gap = max(0, int(gap_length))
    step = max(1, dash + gap)
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        start_ratio = distance / length
        end_ratio = segment_end / length
        segment_start = (int(round(x1 + dx * start_ratio)), int(round(y1 + dy * start_ratio)))
        segment_stop = (int(round(x1 + dx * end_ratio)), int(round(y1 + dy * end_ratio)))
        cv2.line(image, segment_start, segment_stop, color, thickness, lineType=cv2.LINE_8)
        distance += step
