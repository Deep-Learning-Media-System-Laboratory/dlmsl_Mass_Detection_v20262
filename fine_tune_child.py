"""
fine_tune_child.py

Callable YOLO fine-tuning function for SA-AutoAug policy search.

This file is adapted from your original fine-tune.py script:
- model_path, data_yaml, run_name, epochs, project are now arguments.
- It returns the best and last model paths for the child model.
"""

import os
from typing import Dict

from ultralytics import YOLO


def fine_tune_child_model(
    model_path: str,
    data_yaml: str,
    run_name: str,
    project: str = "runs/sa_autoaug_search",
    epochs: int = 5,
    imgsz: int = 1280,
    batch: int = 8,
    device=0,
    workers: int = 0,
    lr0: float = 0.00005,
    lrf: float = 0.01,
    patience: int = 5,
    plots: bool = False,
    val: bool = True,
    exist_ok: bool = True,
) -> Dict:
    """
    Fine-tune one child model from the same plain/baseline model.

    For search, keep epochs small, e.g. 3-5.
    For final training after the best policy is found, use longer epochs.
    """
    print(f"[TRAIN] Base model: {model_path}")
    print(f"[TRAIN] Data YAML : {data_yaml}")
    print(f"[TRAIN] Run name  : {run_name}")
    print(f"[TRAIN] Epochs    : {epochs}")

    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        pretrained=True,
        amp=True,
        patience=patience,
        device=device,
        workers=workers,

        # Optimizer
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        cos_lr=True,
        weight_decay=0.0005,

        # Disable Ultralytics built-in augmentations.
        # The policy-specific dataset already contains the augmentation.
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

        # Detection settings
        max_det=300,

        # Logging
        plots=plots,
        val=val,
        project=os.path.abspath(project),
        name=run_name,
        exist_ok=exist_ok,
    )

    run_dir = os.path.join(project, run_name)
    best_model_path = os.path.join(run_dir, "weights", "best.pt")
    last_model_path = os.path.join(run_dir, "weights", "last.pt")

    if not os.path.exists(best_model_path):
        print(f"[WARN] best.pt not found at {best_model_path}. Falling back to last.pt.")
        best_model_path = last_model_path

    return {
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "run_dir": run_dir,
        "results": results,
    }


if __name__ == "__main__":
    # Example standalone usage, equivalent to your original fine-tune.py.
    fine_tune_child_model(
        model_path="best-with-embed-cmmd-normal.pt",
        data_yaml="hard_exp_c_aug_3.yaml",
        run_name="exp_c_hard_finetune_with_aug_2_",
        project="runs/detect",
        epochs=30,
        imgsz=1280,
        batch=8,
        device=0,
        workers=0,
        patience=20,
        plots=True,
        val=True,
    )
