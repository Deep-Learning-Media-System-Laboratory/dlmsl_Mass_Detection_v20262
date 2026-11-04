from pathlib import Path

from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

# Your custom YOLO26 architecture containing C3k2MixStyle
MODEL_YAML = "ultralytics/cfg/models/26/yolo26n-mixstyle.yaml"

# Official pretrained YOLO26 weights
PRETRAINED_WEIGHTS = "yolo26n.pt"

# Your mammography dataset YAML
# DATA_YAML = "with-hospital.yaml"
DATA_YAML = "thesis.yaml"

# Output
PROJECT = "runs/yolo26_mixstyle"
# NAME = "mixstyle_layer2_layer4-no_hos_legacy"
NAME = "mixstyle_layer2_layer4-thesis-fix"

# Training
EPOCHS = 500
IMGSZ = 1280
BATCH = 16
DEVICE = "3"
WORKERS = 0


def main():

    # ========================================================
    # 1. BUILD CUSTOM YOLO26 + MIXSTYLE ARCHITECTURE
    # ========================================================

    print("=" * 80)
    print("Building YOLO26 + MixStyle")
    print("=" * 80)

    model = YOLO(
        MODEL_YAML,
        task="detect",
    )


    # ========================================================
    # 2. LOAD YOLO26 PRETRAINED WEIGHTS
    # ========================================================
    #
    # Because C3k2MixStyle inherits from C3k2,
    # the original C3k2 parameters should retain compatible
    # parameter names/shapes.
    #
    # MixStyle itself has no trainable parameters.
    # ========================================================

    print("\n" + "=" * 80)
    print("Loading pretrained YOLO26 weights")
    print("=" * 80)

    model.load(PRETRAINED_WEIGHTS)


    # ========================================================
    # 3. PRINT MODEL INFORMATION
    # ========================================================

    print("\n" + "=" * 80)
    print("Model information")
    print("=" * 80)

    model.info(detailed=True)


    # ========================================================
    # 4. CONFIRM MIXSTYLE EXISTS
    # ========================================================

    print("\n" + "=" * 80)
    print("Checking MixStyle modules")
    print("=" * 80)

    mixstyle_found = []

    for name, module in model.model.named_modules():

        class_name = module.__class__.__name__

        if "MixStyle" in class_name:
            mixstyle_found.append(
                (name, class_name)
            )

            print(
                f"FOUND: {name:<40} "
                f"{class_name}"
            )

    if not mixstyle_found:
        raise RuntimeError(
            "\nNo MixStyle module found!\n"
            "Check:\n"
            "  1. mixstyle.py\n"
            "  2. modules/__init__.py\n"
            "  3. tasks.py\n"
            "  4. yolo26n_mixstyle.yaml\n"
        )

    print(
        f"\nTotal MixStyle-related modules: "
        f"{len(mixstyle_found)}"
    )


    # ========================================================
    # 5. TRAIN
    # ========================================================

    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    results = model.train(

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------
        data=DATA_YAML,

        # ----------------------------------------------------
        # Training duration
        # ----------------------------------------------------
        epochs=EPOCHS,

        # ----------------------------------------------------
        # Mammography resolution
        # ----------------------------------------------------
        imgsz=IMGSZ,

        # ----------------------------------------------------
        # Hardware
        # ----------------------------------------------------
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
        project=PROJECT,
        name=NAME,
        exist_ok=False,

        # ----------------------------------------------------
        # Pretrained model
        #
        # We already explicitly loaded yolo26n.pt above.
        # ----------------------------------------------------
        pretrained=True,

        # ----------------------------------------------------
        # Optimizer
        #
        # auto lets Ultralytics choose the appropriate
        # optimizer configuration.
        # ----------------------------------------------------
        optimizer="auto",

        # ----------------------------------------------------
        # Mammography augmentation
        #
        # Keep geometric/destructive augmentations small
        # so the MixStyle comparison stays clean.
        # ----------------------------------------------------
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        # No vertical/horizontal flip initially
        flipud=0.0,
        fliplr=0.0,

        # No hue/saturation modification for grayscale
        hsv_h=0.0,
        hsv_s=0.0,

        # Small brightness/intensity augmentation
        hsv_v=0.1,

        # Existing moderate scale augmentation
        scale=0.25,

        # ----------------------------------------------------
        # Other geometry
        # ----------------------------------------------------
        degrees=0.0,
        translate=0.0,
        shear=0.0,
        perspective=0.0,
        
        cutmix=0.0,
        erasing=0.0,
        amp=True,

        # ----------------------------------------------------
        # Training control
        # ----------------------------------------------------
        patience=100,
        save=True,
        save_period=-1,

        # ----------------------------------------------------
        # Validation
        #
        # MixStyle automatically switches OFF here because
        # validation uses model.eval().
        # ----------------------------------------------------
        val=True,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------
        seed=0,
        deterministic=True,

        # ----------------------------------------------------
        # Logging / visualization
        # ----------------------------------------------------
        plots=True,
        verbose=True,
    )

    print("\nTraining finished.")

    return results


if __name__ == "__main__":
    main()