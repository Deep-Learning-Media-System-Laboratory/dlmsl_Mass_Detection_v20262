#!/usr/bin/env python3
import os

OUTPUT_ROOT = "cdms_augmented_oversample_2x"

CDMS_MANIFEST = os.path.abspath(os.path.join(OUTPUT_ROOT, "domain_manifest_augmented_oversample.csv"))
os.environ["CDMS_DOMAIN_MANIFEST"] = CDMS_MANIFEST

from cd_mixstyle_trainer import DomainMixStyleTrainer
from ultralytics import YOLO

# Start from ORIGINAL best CDMS, not the plain-oversampled checkpoint.
BEST_MODEL = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_layer2_layer4_no_balance/weights/best.pt"

DATA_YAML = os.path.abspath(os.path.join(OUTPUT_ROOT, "all_augmented_oversample.yaml"))

EPOCHS = 50
IMGSZ = 1280
BATCH = 16
DEVICE = 4
WORKERS = 4

PROJECT = "runs/yolo26_cd_mixstyle_augmented_oversample"
RUN_NAME = "cdms_no_balance_policy_augoversample2x_finetune"


def main():
    for p in (BEST_MODEL, DATA_YAML, CDMS_MANIFEST):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    print("Model   :", os.path.abspath(BEST_MODEL))
    print("Data    :", DATA_YAML)
    print("Manifest:", CDMS_MANIFEST)

    model = YOLO(BEST_MODEL)

    model.train(
        trainer=DomainMixStyleTrainer,
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=False,
        optimizer="AdamW",
        lr0=5e-5,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        patience=50,
        # Offline policy augmentation already supplies the intended variation.
        augment=False,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        flipud=0.0,
        multi_scale=False,
        plots=True,
        val=True,
        resume=False,
    )


if __name__ == "__main__":
    main()
