# COMPASS

**COMPASS** is the Comet Object Measurement Pipeline with Automated Selection and Scoring. It runs deep learning models finetuned for comet assay segmentation models on a specified image folder, measures each detected comet, applies automated good/bad selection, and writes a compact Excel workbook with matching visual overlays.

## Setup

Set `COMPASS_HOME` to a writable folder with enough space for weights, outputs, the uv environment, and local caches.

```
# Store COMPASS data, downloaded weights, outputs, and local caches together.
export COMPASS_HOME=/path/to/compass_data

# Tell uv to create the project environment and caches under COMPASS_HOME.
export UV_PROJECT_ENVIRONMENT="$COMPASS_HOME/.venv"
export UV_CACHE_DIR="$COMPASS_HOME/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$COMPASS_HOME/.uv-python"
export UV_LINK_MODE=copy

# Give Matplotlib a writable config/cache directory.
export MPLCONFIGDIR="$COMPASS_HOME/.matplotlib"

mkdir -p "$COMPASS_HOME/weights" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$MPLCONFIGDIR"

# Download packages if needed and install the pinned dependencies from uv.lock.
uv sync --frozen
```

Use the same exports in any shell where you run COMPASS, or add them to your shell profile.

Pretrained weights: https://drive.google.com/drive/folders/11R1PAq6_QYcVv7e1Ka_tNiX41VYs6YKQ?usp=drive_link

| Model | File name | Download |
| --- | --- | --- |
| YOLO, default | `best_yolo11x_seg.pt` | 
| SAM, optional | `sam_vit_b_01ec64.pth` | 
| Mask R-CNN, optional | `best_maskrcnn.pth` | 

Interactive review is enabled by default and uses an OpenCV desktop window. To use it, run COMPASS in a session with access to a graphical display. A quick check is:

```
echo "$DISPLAY"
```

If this prints nothing, configure a graphical display or run COMPASS with `--no-interactive-review`. Keep the project dependency `opencv-python`; `opencv-python-headless` cannot open the review window.

## Optional Preprocessing

COMPASS normally accepts image folders directly. For tricky datasets whose raw image distribution differs from DeepComet-style images, detection can improve if images are first passed through the optional Fiji-style preprocessing step. This applies fixed 39-95 contrast scaling and a baked green LUT by default; pass `--flip-horizontal` to horizontally flip images.

```
uv run compass-preprocess --input /path/to/raw_images --output /path/to/edited_images
```

## Run

Run COMPASS locally on CPU with interactive review:

```
uv run compass --input /path/to/image_folder --output /path/to/output_base
```

Only `--input` and `--output` should be needed for the normal workflow. COMPASS defaults to the YOLO model, CPU execution, the default YOLO weights under `$COMPASS_HOME/weights`, and interactive review. The output path is a base directory; the model name is appended automatically, so `--output /path/to/output_base` writes to `/path/to/output_base/yolo` for the default model.

Interactive review opens one image at a time after automatic scoring. Click model boxes to toggle manual deselection, drag with the live dotted preview to add missed comet boxes, right-click a manual box to remove it, press `u` to undo, `r` to reset the current image, `Enter` or `Space` to accept, and `q` to abort.

To run without the review window, turn interactive review off explicitly:

```
uv run compass --input /path/to/image_folder --output /path/to/output_base --no-interactive-review
```

### Optional Arguments

Use these only when you want to override the defaults:

| Option | Default | Notes |
| --- | --- | --- |
| `--interactive-review`, `--no-interactive-review` | on | Interactive OpenCV review is the default; use `--no-interactive-review` for batch/no-display runs. |
| `--model {yolo,sam,mrcnn}` | `yolo` | Selects the pretrained model. |
| `--weights /path/to/file` | `$COMPASS_HOME/weights/<model weight file>` | Override the default weights path. |
| `--device {cpu,cuda,cuda:0}` | `cpu` | Use `cuda` only when you intentionally want GPU inference. |
| `--max-images N` | all images | Useful for dry runs. |
| `--treatment {infer,baseline,damage,repair}` | `infer` | `infer` reads treatment from image paths. |
| `--yolo-conf FLOAT` | `0.2` | YOLO confidence threshold. |
| `--yolo-iou FLOAT` | `0.5` | YOLO NMS IoU threshold. |
| `--yolo-imgsz SIZE` | `960` | YOLO inference image size. |
| `--batch-size N` | `1` | Inference batch size. |
| `--num-workers N` | `0` | DataLoader worker processes. |
| `--mrcnn-conf FLOAT` | `0.05` | Mask R-CNN confidence threshold. |
| `--mask-thresh FLOAT` | `0.5` | Mask probability threshold used for measurements. |

By default, `--treatment infer` assigns the treatment from image path names. Include one of these words in each image's folder path or file name so good/bad selection uses the intended treatment: `baseline`, `base`, or `control` for baseline images; `damage` or `uv` for damage images; and `repair` for repair images. If no keyword is found, COMPASS treats the image as baseline. You can also force one treatment for the full run with `--treatment baseline`, `--treatment damage`, or `--treatment repair`.

## Outputs

Each run writes:

- `measurements.xlsx`: one combined sheet with one row per detected comet.
- `detections.csv`: flat per-comet measurements for downstream analysis.
- `run_summary.csv`: one row per image with detection, selection, and manual-review counts.
- `overlays/`: final box images, where green marks auto-selected comets, red marks auto-rejected detections, orange marks manually deselected detections, and blue marks manually added detections.
- `run_config.json`: model, weights, thresholds, input, output, and timestamp.
- `run.log`: basic run progress and warnings.
