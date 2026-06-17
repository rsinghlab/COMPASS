from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F


GOOD_BOX_COLOR = (0, 255, 0)
BAD_BOX_COLOR = (255, 0, 0)
MANUAL_DESELECTED_BOX_COLOR = BAD_BOX_COLOR
MANUAL_SELECTED_BOX_COLOR = GOOD_BOX_COLOR
MANUAL_ADDED_BOX_COLOR = (0, 128, 255)
BOX_LINE_WIDTH = 3
LABEL_FONT_SIZE = 36
LABEL_STROKE_WIDTH = 3
DOTTED_DASH_LENGTH = 8
DOTTED_GAP_LENGTH = 6
LABEL_FONT_PATHS = (
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _load_label_font() -> ImageFont.ImageFont:
    for font_path in LABEL_FONT_PATHS:
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, size=LABEL_FONT_SIZE)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=LABEL_FONT_SIZE)
    except OSError:
        try:
            return ImageFont.load_default(size=LABEL_FONT_SIZE)
        except TypeError:
            return ImageFont.load_default()


def box_color_for_status(label: int, row: Optional[Dict[str, object]] = None) -> tuple[int, int, int]:
    if row is not None:
        review_action = str(row.get("review_action", "none")).strip().lower()
        detection_source = str(row.get("detection_source", "model")).strip().lower()
        if review_action == "manual_deselected":
            return MANUAL_DESELECTED_BOX_COLOR
        if review_action == "manual_selected":
            return MANUAL_SELECTED_BOX_COLOR
        if review_action == "manual_added" or detection_source == "manual":
            return MANUAL_ADDED_BOX_COLOR
        good_bad = row.get("good_bad")
        if good_bad == "Good":
            return GOOD_BOX_COLOR
        if good_bad in {"Bad", "Deselected"}:
            return BAD_BOX_COLOR
    return GOOD_BOX_COLOR if int(label) == 1 else BAD_BOX_COLOR


def box_style_for_status(row: Optional[Dict[str, object]] = None) -> str:
    if row is None:
        return "solid"
    review_action = str(row.get("review_action", "none")).strip().lower()
    if review_action in {"manual_deselected", "manual_selected"}:
        return "dotted"
    return "solid"


def render_prediction_overlay(
    img_tensor: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    box_profiles: Optional[List[Dict]] = None,
    rows: Optional[Sequence[Dict[str, object]]] = None,
) -> Image.Image:
    base_img = (img_tensor.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
    boxes_cpu = boxes.detach().cpu().to(torch.float32) if boxes is not None else torch.zeros((0, 4))
    labels_cpu = labels.detach().cpu().to(torch.int64) if labels is not None else torch.zeros((0,), dtype=torch.int64)
    num_boxes = int(boxes_cpu.shape[0])
    if labels_cpu.numel() < num_boxes:
        labels_cpu = torch.cat([labels_cpu, torch.zeros((num_boxes - labels_cpu.numel(),), dtype=torch.int64)])
    labels_cpu = labels_cpu[:num_boxes]

    overlay_pil = F.to_pil_image(base_img)
    boxes_px = boxes_cpu.round().to(torch.int64)
    if num_boxes > 0:
        draw = ImageDraw.Draw(overlay_pil)
        for idx, (label, box) in enumerate(zip(labels_cpu.tolist(), boxes_px.tolist())):
            row = rows[idx] if rows is not None and idx < len(rows) else None
            color = box_color_for_status(
                int(label),
                row,
            )
            if box_style_for_status(row) == "dotted":
                _draw_dotted_rectangle(draw, box, color, BOX_LINE_WIDTH)
            else:
                _draw_rectangle(draw, box, color, BOX_LINE_WIDTH)

    if num_boxes > 0:
        draw = ImageDraw.Draw(overlay_pil)
        font = _load_label_font()
        width, height = overlay_pil.size
        label_text = [str(i) for i in range(num_boxes)]
        if box_profiles is not None and len(box_profiles) == num_boxes:
            label_text = [str(i) for i, _ in enumerate(box_profiles)]
        for text, box in zip(label_text, boxes_px.tolist()):
            x1, y1, x2, y2 = [int(v) for v in box]
            text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=LABEL_STROKE_WIDTH)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            x = max(0, min(x1, width - text_w - 1))
            y = y1 - text_h - 2
            if y < 0:
                y = min(max(0, y2 + 2), height - text_h - 1)
            draw.text(
                (x, y),
                text,
                fill=(255, 255, 255),
                font=font,
                stroke_width=LABEL_STROKE_WIDTH,
                stroke_fill=(0, 0, 0),
            )
    return overlay_pil


def _draw_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    for offset in range(max(1, int(width))):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)


def _draw_dotted_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    _draw_dotted_line(draw, (left, top), (right, top), color, width)
    _draw_dotted_line(draw, (right, top), (right, bottom), color, width)
    _draw_dotted_line(draw, (right, bottom), (left, bottom), color, width)
    _draw_dotted_line(draw, (left, bottom), (left, top), color, width)


def _draw_dotted_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
    dash_length: int = DOTTED_DASH_LENGTH,
    gap_length: int = DOTTED_GAP_LENGTH,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = float(math.hypot(dx, dy))
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
        draw.line((segment_start, segment_stop), fill=color, width=max(1, int(width)))
        distance += step


def save_prediction_overlay(
    img_tensor: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    target: Dict[str, object],
    out_dir: Path,
    box_profiles: Optional[List[Dict]] = None,
    rows: Optional[Sequence[Dict[str, object]]] = None,
) -> Path:
    relative_path = str(target.get("relative_path", "")).strip()
    if relative_path:
        out_dir = out_dir / Path(relative_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = str(target.get("file_name", "image"))
    out_path = out_dir / f"{Path(file_name).stem}_overlay.png"
    overlay_pil = render_prediction_overlay(img_tensor, boxes, labels, box_profiles=box_profiles, rows=rows)
    overlay_pil.save(out_path)
    return out_path
