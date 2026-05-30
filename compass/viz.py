from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes


GOOD_BOX_COLOR = (0, 255, 0)
BAD_BOX_COLOR = (255, 0, 0)
BOX_LINE_WIDTH = 3
LABEL_FONT_SIZE = 36
LABEL_STROKE_WIDTH = 3
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


def save_prediction_overlay(
    img_tensor: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    target: Dict[str, object],
    out_dir: Path,
    box_profiles: Optional[List[Dict]] = None,
) -> Path:
    relative_path = str(target.get("relative_path", "")).strip()
    if relative_path:
        out_dir = out_dir / Path(relative_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_img = (img_tensor.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
    boxes_cpu = boxes.detach().cpu().to(torch.float32) if boxes is not None else torch.zeros((0, 4))
    labels_cpu = labels.detach().cpu().to(torch.int64) if labels is not None else torch.zeros((0,), dtype=torch.int64)
    num_boxes = int(boxes_cpu.shape[0])
    if labels_cpu.numel() < num_boxes:
        labels_cpu = torch.cat([labels_cpu, torch.zeros((num_boxes - labels_cpu.numel(),), dtype=torch.int64)])
    labels_cpu = labels_cpu[:num_boxes]

    overlay = base_img
    boxes_px = boxes_cpu.round().to(torch.int64)
    if num_boxes > 0:
        colors = [GOOD_BOX_COLOR if int(label) == 1 else BAD_BOX_COLOR for label in labels_cpu.tolist()]
        overlay = draw_bounding_boxes(overlay, boxes_px, colors=colors, width=BOX_LINE_WIDTH)

    file_name = str(target.get("file_name", "image"))
    out_path = out_dir / f"{Path(file_name).stem}_overlay.png"
    overlay_pil = F.to_pil_image(overlay)

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

    overlay_pil.save(out_path)
    return out_path
