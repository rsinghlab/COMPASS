from __future__ import annotations

import argparse

import torch

from .pipeline import build_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compass",
        description="Run COMPASS comet detection, measurement, scoring, and selection on an image folder.",
    )
    parser.add_argument("--input", required=True, help="Folder of comet assay images.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output base directory. Writes under a model subfolder; defaults to $COMPASS_HOME/outputs/<input>/<model>.",
    )
    parser.add_argument("--model", choices=("yolo", "sam", "mrcnn"), default="yolo", help="Pretrained model to run.")
    parser.add_argument(
        "--treatment",
        choices=("infer", "baseline", "damage", "repair"),
        default="infer",
        help="Treatment used for good/bad selection. 'infer' reads path names when possible.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu, cuda, or cuda:0.")
    parser.add_argument("--max-images", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")

    parser.add_argument("--yolo-conf", type=float, default=0.2, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.5, help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo-max-det", type=int, default=None, help="Maximum YOLO detections per image.")
    parser.add_argument("--yolo-imgsz", default="960", help="YOLO inference size, for example 960 or 1024.")
    parser.add_argument("--yolo-agnostic-nms", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-yolo-agnostic-nms", action="store_false", dest="yolo_agnostic_nms", help=argparse.SUPPRESS)
    parser.add_argument("--retina-masks", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-retina-masks", action="store_false", dest="retina_masks", help=argparse.SUPPRESS)

    parser.add_argument("--mrcnn-conf", type=float, default=0.05, help="Mask R-CNN confidence threshold.")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Mask probability threshold for measurements.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_pipeline(build_config(args))
    print(f"COMPASS outputs written to {output_dir}")


if __name__ == "__main__":
    main()
