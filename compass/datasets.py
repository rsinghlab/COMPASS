from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


VALID_IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def discover_images(input_dir: Path, valid_exts: Iterable[str] = VALID_IMAGE_EXTS) -> List[Path]:
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input image folder not found: {root}")
    Image.init()
    ext_set = {ext.lower() for ext in valid_exts}
    ext_set.update(ext.lower() for ext in Image.registered_extensions())
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in ext_set)
    if not images:
        raise RuntimeError(f"No image files found in {root}")
    return images


def infer_treatment(path: Path) -> str:
    text = " ".join(part.lower() for part in path.parts)
    if "repair" in text:
        return "repair"
    if "damage" in text or "uv" in text:
        return "damage"
    if "baseline" in text or "base" in text or "control" in text:
        return "baseline"
    return "baseline"


def resize_image(img_t: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.resize(
        img_t,
        list(size_hw),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).contiguous()


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        *,
        size_hw: Tuple[int, int],
        treatment: str = "infer",
        max_images: Optional[int] = None,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        image_paths = discover_images(self.input_dir)
        if max_images is not None:
            image_paths = image_paths[: max(0, int(max_images))]
        self.image_paths = image_paths
        self.size_hw = size_hw
        self.treatment = str(treatment).strip().lower()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        img_t = resize_image(F.to_tensor(image), self.size_hw)
        h, w = int(img_t.shape[1]), int(img_t.shape[2])
        rel_path = image_path.relative_to(self.input_dir)
        treatment = infer_treatment(rel_path) if self.treatment == "infer" else self.treatment
        target: Dict[str, object] = {
            "file_name": image_path.name,
            "relative_path": str(rel_path),
            "image_path": str(image_path),
            "treatment": treatment,
            "orig_size": (int(image.height), int(image.width)),
            "processed_size": (h, w),
        }
        return img_t, target


def collate_images(batch: Sequence[Tuple[torch.Tensor, Dict[str, object]]]):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
