#!/usr/bin/env python3
"""
fine_tune_cdms_augmented_oversample_selected.py

Fine-tune the ORIGINAL best Cross-Domain MixStyle checkpoint using the
already-generated policy-augmented oversampling dataset.

Run from:
  /home/yeozekiel/yolo/YOLO-MS/ultralytics
"""

import os
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# Change only if your generator used a different OUTPUT_ROOT.
OUTPUT_ROOT = Path("cdms_augmented_oversample_2x").resolve()

DATA_YAML = OUTPUT_ROOT / "all_augmented_oversample.yaml"
DOMAIN_MANIFEST = OUTPUT_ROOT / "domain_manifest_augmented_oversample.csv"

# Start from the ORIGINAL strongest CDMS checkpoint.
MODEL_PATH = Path(
    "runs/detect/runs/yolo26_cd_mixstyle/"
    "cd_mixstyle_layer2_layer4_no_balance/"
    "weights/best.pt"
).resolve()

# IMPORTANT: set before importing custom CDMS trainer.
os.environ["CDMS_DOMAIN_MANIFEST"] = str(DOMAIN_MANIFEST)

from ultralytics import YOLO
from cd_mixstyle_trainer import DomainMixStyleTrainer


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 50
IMGSZ = 1280
BATCH = 16
DEVICE = 0
WORKERS = 4

PROJECT = "runs/yolo26_cd_mixstyle_augmented_oversample"
NAME = "cdms_policy_augoversample_selected_domains_ft"


def main():
    for p in [MODEL_PATH, DATA_YAML, DOMAIN_MANIFEST]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file:\n{p}")

    print("=" * 80)
    print("CDMS + POLICY-AUGMENTED OVERSAMPLING FINE-TUNE")
    print("=" * 80)
    print(f"Model     : {MODEL_PATH}")
    print(f"Data YAML : {DATA_YAML}")
    print(f"Manifest  : {DOMAIN_MANIFEST}")
    print(f"Epochs    : {EPOCHS}")
    print(f"Image size: {IMGSZ}")
    print(f"Batch     : {BATCH}")
    print(f"Device    : {DEVICE}")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    model.train(
        trainer=DomainMixStyleTrainer,
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,

        project=PROJECT,
        name=NAME,
        exist_ok=False,

        # Conservative fine-tuning
        optimizer="AdamW",
        lr0=5e-5,
        lrf=0.01,
        cos_lr=True,
        weight_decay=5e-4,
        patience=50,

        # Disable online augmentation so this experiment isolates
        # the offline best_policy.json augmented oversampling.
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        fliplr=0.0,
        flipud=0.0,
        multi_scale=False,

        val=True,
        plots=True,

        # New experiment, not optimizer/scheduler resume.
        resume=False,
    )


if __name__ == "__main__":
    main()
