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

Pretrained weights: https://drive.google.com/drive/folders/11R1PAq6_QYcVv7e1Ka_tNiX41VYs6YKQ?usp=drive_link

| Model | File name | Download |
| --- | --- | --- |
| YOLO, default | `best_yolo11x_seg.pt` | 
| SAM, optional | `sam_vit_b_01ec64.pth` | 
| Mask R-CNN, optional | `best_maskrcnn.pth` | 

## Optional Preprocessing

COMPASS normally accepts image folders directly. For tricky datasets whose raw image distribution differs from DeepComet-style images, detection can improve if images are first passed through the optional Fiji-style preprocessing step. This applies fixed 39-95 contrast scaling and a baked green LUT by default; pass `--flip-horizontal` to horizontally flip images.

```bash
uv run compass-preprocess --input /path/to/raw_images --output /path/to/edited_images
uv run compass --input /path/to/edited_images
```

## Run

Run the default YOLO pipeline:

```bash
uv run compass --input /path/to/image_folder
```

YOLO hyperparameters, including the `--yolo-conf` confidence threshold default, can be changed in `compass/cli.py` if desired.

By default, `--treatment infer` assigns the treatment from image path names. Include one of these words in each image's folder path or file name so good/bad selection uses the intended treatment: `baseline`, `base`, or `control` for baseline images; `damage` or `uv` for damage images; and `repair` for repair images. If no keyword is found, COMPASS treats the image as baseline. You can also force one treatment for the full run with `--treatment baseline`, `--treatment damage`, or `--treatment repair`.

Optional model choices:

```bash
uv run compass --input /path/to/image_folder --model sam
uv run compass --input /path/to/image_folder --model mrcnn
```

By default, outputs are written under an input- and model-specific folder:

- YOLO: `$COMPASS_HOME/outputs/<input-folder>/yolo`
- SAM: `$COMPASS_HOME/outputs/<input-folder>/sam`
- Mask R-CNN: `$COMPASS_HOME/outputs/<input-folder>/mrcnn`

If `--output /path/to/output_base` is provided, COMPASS writes to `/path/to/output_base/<model>`.

## Outputs

Each run writes:

- `measurements.xlsx`: one combined sheet with one row per detected comet.
- `detections.csv`: flat per-comet measurements for downstream analysis.
- `run_summary.csv`: one row per image with detection and selection counts.
- `overlays/`: final red/green box images, where green marks selected comets and red marks rejected detections.
- `run_config.json`: model, weights, thresholds, input, output, and timestamp.
- `run.log`: basic run progress and warnings.
