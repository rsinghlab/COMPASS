from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


DEFAULT_MIN_VAL = 39.0
DEFAULT_MAX_VAL = 95.0
DEFAULT_OUTPUT_SUFFIX = ".tif"
VALID_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
VALID_BATS_CONDITIONS = {"Base", "Damage", "Repair"}
VALID_BATS_IMAGE_EXTS = {".tif", ".tiff"}


@dataclass(frozen=True)
class PreprocessRecord:
    image_path: Path
    relative_output_path: Path


def linear_scale_to_uint8(arr: np.ndarray, min_val: float = DEFAULT_MIN_VAL, max_val: float = DEFAULT_MAX_VAL) -> np.ndarray:
    denom = max(float(max_val) - float(min_val), 1e-6)
    out = (arr.astype(np.float32) - float(min_val)) * (255.0 / denom)
    return np.rint(np.clip(out, 0.0, 255.0)).astype(np.uint8)


def apply_green_lut(gray_u8: np.ndarray) -> np.ndarray:
    rgb = np.zeros((gray_u8.shape[0], gray_u8.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 1] = gray_u8
    return rgb


def preprocess_image(
    image: Image.Image,
    *,
    min_val: float = DEFAULT_MIN_VAL,
    max_val: float = DEFAULT_MAX_VAL,
    flip_horizontal: bool = False,
    green_lut: bool = True,
) -> Image.Image:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if bool(flip_horizontal):
        arr = np.fliplr(arr)
    gray_u8 = linear_scale_to_uint8(arr, min_val=min_val, max_val=max_val)
    if bool(green_lut):
        return Image.fromarray(apply_green_lut(gray_u8))
    return Image.fromarray(gray_u8)


def iter_mirror_records(input_dir: Path, *, output_suffix: str = DEFAULT_OUTPUT_SUFFIX) -> Iterable[PreprocessRecord]:
    input_root = Path(input_dir).expanduser().resolve()
    for image_path in sorted(input_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTS:
            continue
        rel_path = image_path.relative_to(input_root).with_suffix(output_suffix)
        yield PreprocessRecord(image_path=image_path, relative_output_path=rel_path)


def iter_bats_records(input_dir: Path, *, output_suffix: str = DEFAULT_OUTPUT_SUFFIX) -> Iterable[PreprocessRecord]:
    input_root = Path(input_dir).expanduser().resolve()
    for date_dir in sorted(input_root.glob("EPFU_Comets_*")):
        if not date_dir.is_dir():
            continue
        for condition_dir in sorted(date_dir.iterdir()):
            if not condition_dir.is_dir() or condition_dir.name not in VALID_BATS_CONDITIONS:
                continue
            for sample_dir in sorted(condition_dir.glob("EPFU_*")):
                if not sample_dir.is_dir():
                    continue
                picture_input_dir = sample_dir / "Picture Input"
                if not picture_input_dir.is_dir():
                    continue
                for image_path in sorted(picture_input_dir.iterdir()):
                    if not image_path.is_file() or image_path.suffix.lower() not in VALID_BATS_IMAGE_EXTS:
                        continue
                    rel_path = Path(date_dir.name, condition_dir.name, sample_dir.name, image_path.stem + output_suffix)
                    yield PreprocessRecord(image_path=image_path, relative_output_path=rel_path)


def preprocess_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    layout: str = "mirror",
    min_val: float = DEFAULT_MIN_VAL,
    max_val: float = DEFAULT_MAX_VAL,
    flip_horizontal: bool = False,
    green_lut: bool = True,
    output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
) -> int:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input image folder not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records = _records_for_layout(input_root, layout=layout, output_suffix=output_suffix)
    processed = 0
    for record in records:
        dst_path = output_root / record.relative_output_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record.image_path) as image:
            edited = preprocess_image(
                image,
                min_val=min_val,
                max_val=max_val,
                flip_horizontal=flip_horizontal,
                green_lut=green_lut,
            )
            edited.save(dst_path)
        processed += 1
    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compass-preprocess",
        description="Optionally preprocess comet assay images before running COMPASS.",
    )
    parser.add_argument("--input", required=True, help="Raw image folder.")
    parser.add_argument("--output", required=True, help="Edited image output folder.")
    parser.add_argument(
        "--layout",
        choices=("mirror", "bats"),
        default="mirror",
        help="Input folder layout. 'mirror' preserves relative paths; 'bats' reads */Picture Input/*.tif[f].",
    )
    parser.add_argument("--min-val", type=float, default=DEFAULT_MIN_VAL, help="Lower fixed contrast value.")
    parser.add_argument("--max-val", type=float, default=DEFAULT_MAX_VAL, help="Upper fixed contrast value.")
    parser.add_argument("--output-suffix", default=DEFAULT_OUTPUT_SUFFIX, help="Output image suffix.")
    parser.add_argument("--flip-horizontal", action="store_true", default=False, help="Horizontally flip images.")
    parser.add_argument("--green-lut", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-green-lut", action="store_false", dest="green_lut", help="Write grayscale output instead of green RGB.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    processed = preprocess_folder(
        Path(args.input),
        Path(args.output),
        layout=str(args.layout),
        min_val=float(args.min_val),
        max_val=float(args.max_val),
        flip_horizontal=bool(args.flip_horizontal),
        green_lut=bool(args.green_lut),
        output_suffix=str(args.output_suffix),
    )
    print(f"COMPASS preprocessing wrote {processed} image(s) to {Path(args.output).expanduser().resolve()}")


def _records_for_layout(input_root: Path, *, layout: str, output_suffix: str) -> Iterable[PreprocessRecord]:
    layout_key = str(layout).strip().lower()
    if layout_key == "mirror":
        return iter_mirror_records(input_root, output_suffix=output_suffix)
    if layout_key == "bats":
        return iter_bats_records(input_root, output_suffix=output_suffix)
    raise ValueError("Layout must be one of: mirror, bats")


if __name__ == "__main__":
    main()
