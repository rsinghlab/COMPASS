# COMPASS

**COMPASS** is the Comet Object Measurement Pipeline with Automated Selection and Scoring. It runs deep learning models finetuned for comet assay segmentation models on a specified image folder, measures each detected comet, applies automated good/bad selection, and writes a compact Excel workbook with matching visual overlays.

## Install

```bash
uv sync --frozen
```

## Download Weights

Set `COMPASS_HOME` to a folder with enough space, then place the pretrained weights in `$COMPASS_HOME/weights`.

```bash
export COMPASS_HOME=/path/to/compass_data
mkdir -p "$COMPASS_HOME/weights"
```

Pretrained weights:

| Model | File name | Download |
| --- | --- | --- |
| YOLO, default | `best_yolo11x_seg.pt` | Google Drive link to add before release |
| SAM, optional | `sam_vit_b_01ec64.pth` | Google Drive link to add before release |
| Mask R-CNN, optional | `best_maskrcnn.pth` | Google Drive link to add before release |

## Run

Run the default YOLO pipeline:

```bash
uv run compass --input /path/to/image_folder
```

Optional model choices:

```bash
uv run compass --input /path/to/image_folder --model sam
uv run compass --input /path/to/image_folder --model mrcnn
```

By default, outputs are written under a model-specific folder:

- YOLO: `$COMPASS_HOME/outputs/yolo`
- SAM: `$COMPASS_HOME/outputs/sam`
- Mask R-CNN: `$COMPASS_HOME/outputs/mrcnn`

If `--output /path/to/output_base` is provided, COMPASS writes to `/path/to/output_base/<model>`.

## Outputs

Each run writes:

- `measurements.xlsx`: one sheet per image and one row per detected comet.
- `detections.csv`: flat per-comet measurements for downstream analysis.
- `run_summary.csv`: one row per image with detection and selection counts.
- `overlays/`: final red/green box images, where green marks selected comets and red marks rejected detections.
- `run_config.json`: model, weights, thresholds, input, output, and timestamp.
- `run.log`: basic run progress and warnings.
