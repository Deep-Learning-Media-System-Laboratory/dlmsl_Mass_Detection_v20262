# dlmsl_Mass_Detection_v20262: YOLO26 with MixStyle & Cross-Domain MixStyle

This repository provides an implementation of **YOLO26** equipped with **MixStyle** and **Cross-Domain MixStyle (CDMS)** for robust breast mass detection across diverse mammography domains (e.g., CBIS-DDSM, INbreast, CMMD, ViNDr-Mammo, EMBED, RSNA, hospital datasets, and synthetic data).

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training YOLO26 MixStyle (Standard)](#training-yolo26-mixstyle-standard)
- [Training YOLO26 Cross-Domain MixStyle (CDMS)](#training-yolo26-cross-domain-mixstyle-cdms)
  - [Step 1: Generate Domain Manifest (`make_domain_manifest.py`)](#step-1-generate-domain-manifest-make_domain_manifestpy)
  - [Step 2: Verify Domain Manifest (`verify_domain_manifest.py`)](#step-2-verify-domain-manifest-verify_domain_manifestpy)
  - [Step 3: Run Cross-Domain Training (`train_yolo26_cd_mixstyle.py`)](#step-3-run-cross-domain-training-train_yolo26_cd_mixstylepy)
- [Configuration & Hyperparameter Details](#configuration--hyperparameter-details)
- [Evaluation & Post-Processing](#evaluation--post-processing)
  - [1. Single-Point Evaluation & Visualization (`evaluate.py`)](#1-single-point-evaluation--visualization-evaluatepy)
  - [2. NMS IoU & Confidence Threshold Sweep (`evaluate_nms_conf_sweep.py`)](#2-nms-iou--confidence-threshold-sweep-evaluate_nms_conf_sweeppy)

---

## Overview

Mammography imaging is subject to severe domain shifts due to varying scanner vendors, acquisition protocols, breast density variations, and institutional differences. This repository addresses domain shift using feature-level style mixing:

| Method                           | Behavior                                                                                                                      | Use Case                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Standard MixStyle**            | Mixes instance feature statistics ($\mu, \sigma$) between random sample pairs within the mini-batch using Beta interpolation. | General feature perturbation and regularization.                                       |
| **Cross-Domain MixStyle (CDMS)** | Enforces that the style donor image **must come from a different domain/dataset** than the source image.                      | Eliminating domain-specific biases and learning domain-invariant mass representations. |

Both methods:

- Are injected into the YOLO26 backbone via custom C3k2 blocks (`C3k2MixStyle` and `C3k2CrossDomainMixStyle`) at stages P2 and P4 (layers 2 and 4).
- Operate **only during training**. They are automatically bypassed during validation and inference (`model.eval()`).

---

## Directory Structure

```text
├── ultralytics/
│   ├── cfg/models/26/
│   │   ├── yolo26-mixstyle.yaml          # Backbone definition with C3k2MixStyle
│   │   └── yolo26-cd-mixstyle.yaml       # Backbone definition with C3k2CrossDomainMixStyle
│   └── nn/modules/
│       ├── mixstyle.py                   # MixStyle & C3k2MixStyle definitions
│       └── crossdomain_mixstyle.py       # CrossDomainMixStyle & C3k2CrossDomainMixStyle definitions
├── train_yolo26_mixstyle.py              # Training script for Standard MixStyle
├── train_yolo26_cd_mixstyle.py           # Training script for Cross-Domain MixStyle
├── make_domain_manifest.py               # Manifest generator script mapping images to domain labels
├── verify_domain_manifest.py             # Manifest integrity verification script
├── cd_mixstyle_trainer.py                # DomainMixStyleTrainer (injects domain IDs into batches)
├── cd_mixstyle_data.py                   # DomainYOLODataset (loads manifest & provides domain IDs)
├── evaluate.py                           # Single-point evaluation & annotation visualization
├── evaluate_nms_conf_sweep.py            # NMS IoU & confidence threshold sweep script
└── yolo26n.pt                            # Pretrained YOLO26 backbone weights
```

---

## Environment Setup

1. **Clone the repository and navigate into the workspace:**

   ```bash
   cd dlmsl_Mass_Detection_v20262
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ and PyTorch with CUDA support installed:

   ```bash
   pip install -e .
   ```

3. **Verify pretrained weights:**
   Ensure `yolo26n.pt` is present in the repository root (or provide your path to pretrained YOLO26 weights).

---

## Data Preparation

Dataset definitions follow the standard YOLO format defined in a `.yaml` file (e.g., `data.yaml` or `with-hospital.yaml`):

```yaml
path: /path/to/dataset_root # Optional dataset root
train: images/train # Can be a directory path or a .txt list of image paths
val: images/val # Validation image directory or .txt list
names:
  0: mass
```

---

## Training YOLO26 MixStyle (Standard)

Standard MixStyle mixes feature representations between randomly chosen sample pairs within the mini-batch.

### 1. Configure `train_yolo26_mixstyle.py`

Open `train_yolo26_mixstyle.py` and adjust the configuration parameters:

```python
# Architecture YAML containing C3k2MixStyle
MODEL_YAML = "ultralytics/cfg/models/26/yolo26-mixstyle.yaml"

# Pretrained weights
PRETRAINED_WEIGHTS = "yolo26n.pt"

# Dataset YAML
DATA_YAML = "path/to/your_dataset.yaml"

# Training parameters
EPOCHS = 500
IMGSZ = 1280
BATCH = 16
DEVICE = "0"  # GPU ID or "0,1"
WORKERS = 4
PROJECT = "runs/yolo26_mixstyle"
NAME = "mixstyle_experiment"
```

### 2. Launch Training

Run the training script:

```bash
python train_yolo26_mixstyle.py
```

The script will:

1. Build YOLO26 with `C3k2MixStyle` layers.
2. Load pretrained weights from `yolo26n.pt`.
3. Check and confirm `MixStyle` module presence in the backbone.
4. Execute training using mammography-tailored augmentations (e.g., `mosaic=0`, `mixup=0`, grayscale preserving `hsv_v=0.1`, `scale=0.25`).

---

## Training YOLO26 Cross-Domain MixStyle (CDMS)

Cross-Domain MixStyle requires image domain labels so that style statistics are swapped **exclusively across distinct domains** (e.g. ViNDr $\leftrightarrow$ INbreast $\leftrightarrow$ Hospital).

### Step 1: Generate Domain Manifest (`make_domain_manifest.py`)

Before training CDMS, you must generate a CSV manifest that maps each training image to its corresponding domain identifier.

#### 1. Configure `make_domain_manifest.py`

Open `make_domain_manifest.py` and configure the dataset source and output CSV filenames:

```python
DATA_YAML = "path/to/your_dataset.yaml"
OUTPUT_CSV = "domain_manifest.csv"
UNKNOWN_CSV = "domain_manifest_unknown.csv"
```

#### 2. Verify Domain Detection Rules

`make_domain_manifest.py` detects domain tags by examining filenames and path patterns in `detect_domain(image_path)`:

| Domain         | Matching Patterns                            |
| -------------- | -------------------------------------------- |
| `vindr`        | `vindr_`, `vindr-n_`                         |
| `cmmd`         | `cmmd_`, `cmmd-n_`, `d1-`, `d2-`             |
| `cbis`         | `cbis_`, `mass-training_`, `mass-testing_`   |
| `inbreast`     | `inbreast_`, 8-digit numeric stems           |
| `embed`        | `embed_`, `embed-n_`                         |
| `rsna`         | `rsna_`, `rsna-n_`                           |
| `bcs-dbt`      | `bcs_`, `bcs-dbt_`, `bcsdbt-`, `bcsdbt_`     |
| `hospital`     | `hospital_`, `done_`                         |
| `hospital-new` | `b20260814_`, `b202511_`, `b20260202_`, etc. |
| `hospital-old` | `hospital-old_`, `hospital_phuc_`, `puc_`    |
| `bcc` / `img`  | `bcc_`, `img_`                               |
| `synthetic`    | `synthetic_`                                 |

> [!TIP]
> If your dataset uses custom image naming conventions, add your prefix rules to `detect_domain()` in `make_domain_manifest.py`.

#### 3. Run the Manifest Generator

Execute:

```bash
python make_domain_manifest.py
```

The script prints a summary table:

```text
======================================================================
DOMAIN MANIFEST SUMMARY
======================================================================
cbis                     1200
cmmd                     1800
inbreast                  410
vindr                    2500
----------------------------------------------------------------------
CLASSIFIED               5910
UNKNOWN                     0
TOTAL                    5910
======================================================================
```

> [!IMPORTANT]
> **Ensure `UNKNOWN = 0`**. If any images are classified as unknown, check `domain_manifest_unknown.csv`, update the prefix detection rules in `make_domain_manifest.py`, and re-run until all images are assigned a valid domain.

---

### Step 2: Verify Domain Manifest (`verify_domain_manifest.py`)

Verify the manifest's integrity (checks for broken file paths, duplicate entries, and guarantees at least two domains):

```python
# In verify_domain_manifest.py:
MANIFEST = "domain_manifest.csv"
```

Run:

```bash
python verify_domain_manifest.py
```

Expected output:

```text
Rows: 5910
Duplicate paths: 0
Missing files: 0
Domains:
...
PASS: manifest verified.
```

---

### Step 3: Run Cross-Domain Training (`train_yolo26_cd_mixstyle.py`)

#### 1. Configure `train_yolo26_cd_mixstyle.py`

Edit the configuration settings in `train_yolo26_cd_mixstyle.py`:

```python
MODEL_YAML = "ultralytics/cfg/models/26/yolo26-cd-mixstyle.yaml"
PRETRAINED_WEIGHTS = "yolo26n.pt"
DATA_YAML = "path/to/your_dataset.yaml"
DOMAIN_MANIFEST = "domain_manifest.csv"

PROJECT = "runs/yolo26_cd_mixstyle"
NAME = "cd_mixstyle_experiment"

EPOCHS = 300
IMGSZ = 1280
BATCH = 16
DEVICE = "0"
WORKERS = 4
```

#### 2. Launch Training

Run:

```bash
python train_yolo26_cd_mixstyle.py
```

#### How it works under the hood:

1. Sets `os.environ["CDMS_DOMAIN_MANIFEST"] = DOMAIN_MANIFEST`.
2. Passes `DomainMixStyleTrainer` to `model.train(trainer=DomainMixStyleTrainer, ...)`.
3. `DomainYOLODataset` maps each training image to its integer domain label.
4. `DomainMixStyleTrainer.preprocess_batch` supplies `domain_ids` directly to the `CrossDomainMixStyle` layers.
5. In each forward pass, style donors are sampled uniformly from **different domains** present in the batch (`build_cross_domain_perm`).
6. At training start, the first batch domain distribution is logged to the terminal:
   ```text
   ======================================================================
   FIRST CROSS-DOMAIN TRAIN BATCH
   ======================================================================
   Domain IDs: [0, 2, 1, 3, 0, 1, 2, 3, ...]
   Counts: {0: 4, 1: 4, 2: 4, 3: 4}
   CDMS modules: 2
   ======================================================================
   ```

---

## Configuration & Hyperparameter Details

### Mammography Augmentation Guidelines

Mammography images require careful augmentation choices so as not to distort micro-calcifications or subtle mass margins:

- `mosaic=0.0`, `mixup=0.0`, `copy_paste=0.0`: Disabled to prevent destructive artifacts.
- `hsv_h=0.0`, `hsv_s=0.0`: Kept at 0 for grayscale mammograms.
- `hsv_v=0.1`: Gentle contrast/brightness jitter.
- `scale=0.25`: Moderate scale jitter.

### Single-Domain Fine-Tuning (Optional)

If fine-tuning an existing CDMS checkpoint on a dataset containing only one domain, set:

```bash
export CDMS_ALLOW_SINGLE_DOMAIN=1 # Linux / macOS
# or in PowerShell:
$env:CDMS_ALLOW_SINGLE_DOMAIN="1"
```

In this mode, CDMS logs a warning and leaves features untouched when no different donor domain is present.

---

## Evaluation & Post-Processing

This repository provides two specialized evaluation scripts tailored for mammography mass detection:

- **`evaluate.py`**: Detailed single-threshold evaluation with per-box CSV logging, JSON metrics summary, and annotated image visualizations.
- **`evaluate_nms_conf_sweep.py`**: Multi-threshold grid sweep over NMS IoU and confidence values to identify the optimal operating point (maximizing F1-score or AP).

---

### 1. Single-Point Evaluation & Visualization (`evaluate.py`)

`evaluate.py` evaluates a trained `.pt` or exported `.onnx` model on a designated dataset split and produces quantitative metrics as well as visual inspection images.

#### Dataset Directory Layout

The evaluation script expects the dataset directory structure:

```text
<VAL_DIR>/
├── images/
│   └── <SPLIT>/          # e.g., test/, val/
└── labels/
    └── <SPLIT>/          # e.g., test/, val/
```

_(If `SPLIT = "all"`, it looks directly inside `<VAL_DIR>/images` and `<VAL_DIR>/labels`.)_

#### Configuration Parameters

Open `evaluate.py` and adjust the configuration block at the top:

```python
# Model and dataset paths
MODEL_PATH = "runs/yolo26_cd_mixstyle/cd_mixstyle_experiment/weights/best.pt"
VAL_DIR = "/path/to/mammography_dataset"
SPLIT = "test"  # Options: "test", "val", "train", or "all"

# Detection parameters
TARGET_IMG_SIZE = 1280
CONF_THRES = 0.3  # Confidence cutoff for predictions
IOU_THRES = 0.5  # NMS IoU threshold for overlapping predictions
GT_IOU_THRES = 0.2  # IoU threshold to count prediction as TP (standard for breast masses)

# Hardware and output
DEVICE = "0"  # GPU ID or "cpu"
DRAW_ANNOTATIONS = True  # Set True to save annotated images
OUTPUT_FOLDER = "evaluation_results/single_run"
```

#### Run Evaluation

Execute:

```bash
python evaluate.py
```

#### Outputs in `OUTPUT_FOLDER`

1. **Console Summary**: Displays TP, FP, FN, Precision, Recall, F1 Score, and mAP@50.
2. **`results_summary.json`**: Complete metrics summary and per-image prediction counts.
3. **`results_per_box.csv`**: Granular per-box CSV with columns `[image, pred_x, pred_y, pred_w, pred_h, conf, best_iou, tp]`.
4. **Annotated Images (`*_annotated.jpg`)** _(if `DRAW_ANNOTATIONS = True`)_:
   - 🟩 **Green boxes**: True Positives (TP, matching GT with IoU > `GT_IOU_THRES`).
   - 🟧 **Orange boxes**: False Positives (FP).
   - 🟦 **Blue boxes**: Ground Truth (GT) annotations.

---

### 2. NMS IoU & Confidence Threshold Sweep (`evaluate_nms_conf_sweep.py`)

In clinical applications, finding the best balance between sensitivity (Recall) and false alarms (Precision) is critical. `evaluate_nms_conf_sweep.py` performs a grid search across multiple NMS IoU thresholds and confidence thresholds.

#### Efficient Sweep Design

The script runs inference once per NMS IoU threshold at `INFER_CONF_FLOOR = min(CONF_THRESHOLDS)` and filters upward across confidence levels, significantly speeding up large sweeps without redundant inference passes.

#### Configuration Parameters

Open `evaluate_nms_conf_sweep.py` and configure the sweep ranges:

```python
MODEL_PATH = "runs/yolo26_cd_mixstyle/cd_mixstyle_experiment/weights/best.pt"
VAL_DIR = "/path/to/mammography_dataset"
SPLIT = "test"

# Sweep grids
NMS_IOU_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]
CONF_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

GT_IOU_THRES = 0.2  # Fixed ground truth matching IoU
TARGET_IMG_SIZE = 1280
DEVICE = "0"
DRAW_ANNOTATIONS = False  # Recommended False during grid sweeps
OUTPUT_FOLDER = "evaluation_results/nms_conf_sweep"
```

#### Run the Sweep

Execute:

```bash
python evaluate_nms_conf_sweep.py
```

#### Outputs

1. **Matrix Table in Console**:
   ```text
   =========================================================================================================
   NMS IoU / Confidence Sweep
   =========================================================================================================
    NMS IoU |   Conf |    TP |    FP |    FN |  Precision |     Recall |         F1 |   AP@GTIoU
   ---------------------------------------------------------------------------------------------------------
       0.10 |   0.10 |   240 |   180 |    45 |     0.5714 |     0.8421 |     0.6809 |     0.7950
       0.10 |   0.20 |   230 |   110 |    55 |     0.6765 |     0.8070 |     0.7360 |     0.7712
       ...
   =======================================================
   BEST COMBINATION BY F1
   =======================================================
   NMS IoU       : 0.20
   Confidence    : 0.30
   GT Match IoU  : 0.20
   TP            : 228
   FP            : 74
   FN            : 57
   Precision     : 0.7550
   Recall        : 0.8000
   F1            : 0.7768
   AP@GTIoU      : 0.7645
   =======================================================
   ```
2. **`sweep_results.csv`**: Tabular CSV recording metrics for every combination of NMS IoU and confidence.
3. **`sweep_results.json`**: Structured JSON detailing evaluation metadata, the full sweep results array, and the best configuration by F1 score.
