import os

from ultralytics import YOLO

from cd_mixstyle_trainer import (
    DomainMixStyleTrainer,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_YAML = (
    "ultralytics/cfg/models/26/"
    "yolo26n-cd-mixstyle.yaml"
)

PRETRAINED_WEIGHTS = (
    "yolo26n.pt"
    # "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_layer2_layer4_new_retrain-4/weights/last.pt"
)

DATA_YAML = "all-supress100.yaml"

DOMAIN_MANIFEST = (
    "allsup100_domain_manifest.csv"
)


PROJECT = (
    "runs/yolo26_cd_mixstyle"
)

NAME = (
    "cd_mixstyle_layer2_layer4_allsup100"
)


# Start with ONE epoch
EPOCHS = 300

IMGSZ = 1280
BATCH = 16
DEVICE = "3"
WORKERS = 4


def main():

    # ========================================================
    # DOMAIN MANIFEST
    # ========================================================

    os.environ[
        "CDMS_DOMAIN_MANIFEST"
    ] = str(
        DOMAIN_MANIFEST
    )


    # ========================================================
    # MODEL
    # ========================================================

    print("=" * 80)
    print(
        "Building YOLO26 + "
        "Cross-Domain MixStyle"
    )
    print("=" * 80)

    model = YOLO(
        MODEL_YAML,
        task="detect",
    )


    # ========================================================
    # PRETRAINED YOLO26
    # ========================================================

    model.load(
        PRETRAINED_WEIGHTS
    )


    # ========================================================
    # CHECK MODULES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Checking Cross-Domain MixStyle"
    )
    print("=" * 80)

    found = []

    for name, module in (
        model.model.named_modules()
    ):

        class_name = (
            module.__class__.__name__
        )

        if (
            "CrossDomainMixStyle"
            in class_name
        ):

            found.append(
                (
                    name,
                    class_name,
                )
            )

            print(
                f"FOUND: "
                f"{name:<40}"
                f"{class_name}"
            )

    if not found:

        raise RuntimeError(
            "No CrossDomainMixStyle "
            "modules found."
        )


    # ========================================================
    # TRAIN
    # ========================================================

    model.train(
        resume=True,
        trainer=DomainMixStyleTrainer,

        data=DATA_YAML,

        epochs=EPOCHS,

        imgsz=IMGSZ,

        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,

        project=PROJECT,
        name=NAME,

        exist_ok=False,

        pretrained=True,

        optimizer="auto",

        # ====================================================
        # SAME AUGMENTATION AS RANDOM MIXSTYLE
        # ====================================================

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        cutmix=0.0,
        erasing=0.0,

        flipud=0.0,
        fliplr=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.1,
        amp=True,

        scale=0.25,

        degrees=0.0,
        translate=0.0,
        shear=0.0,
        perspective=0.0,

        patience=100,

        save=True,
        save_period=-1,

        val=True,

        seed=0,
        deterministic=True,

        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()