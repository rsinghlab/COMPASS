from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_COMPASS_HOME = Path("/users/jrober48/scratch/COMPASS")


def compass_home() -> Path:
    return Path(os.environ.get("COMPASS_HOME", str(DEFAULT_COMPASS_HOME))).expanduser().resolve()


def default_weights_dir() -> Path:
    return compass_home() / "weights"


def default_output_dir() -> Path:
    return compass_home() / "outputs"


def resolve_output_dir(path_value: Optional[str], model_name: str, input_dir: Optional[Path | str] = None) -> Path:
    model_key = str(model_name).strip().lower()
    if path_value:
        base_dir = Path(path_value).expanduser().resolve()
    else:
        input_name = Path(input_dir).expanduser().name if input_dir else "default"
        output_name = input_name.removesuffix("_edited") or "default"
        base_dir = default_output_dir() / output_name
    return base_dir / model_key


def default_weight_path(model_name: str) -> Path:
    model_key = str(model_name).strip().lower()
    names = {
        "yolo": "best_yolo11x_seg.pt",
        "sam": "sam_vit_b_01ec64.pth",
        "mrcnn": "best_maskrcnn.pth",
    }
    if model_key not in names:
        raise ValueError(f"Unsupported model: {model_name}")
    return default_weights_dir() / names[model_key]
