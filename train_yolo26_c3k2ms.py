from pathlib import Path

from ultralytics import YOLO

# MODEL_YAML = "ultralytics/cfg/models/26/yolo26n-c3k2ms.yaml"
MODEL_YAML = "ultralytics/cfg/models/26/yolo26n-c3k2ms-deeplite.yaml"
PRETRAINED_WEIGHTS = "yolo26n.pt"
DATA_YAML = "all.yaml"

DEVICE = "1"
PROJECT = "runs/detect"
RUN_NAME = "with-embed-cmmd-normal_yolo26n-deeplite-gate01"


def main() -> None:
    """Build, initialize, and train YOLO26-C3k2MS."""
    # Check only files that must physically exist under this exact name.
    data_yaml = Path(DATA_YAML)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML was not found:\n{data_yaml.resolve()}")

    # Do not call Path.exists() on the scale-specific model alias.
    # Ultralytics resolves yolo26n-c3k2ms.yaml to yolo26-c3k2ms.yaml.
    # print(f"Building custom model from: {MODEL_YAML}")
    model = YOLO(MODEL_YAML)

    # print(f"Loading compatible weights from: {PRETRAINED_WEIGHTS}")
    # model.load(PRETRAINED_WEIGHTS)

    # model.info(detailed=True)

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        patience=60,
        imgsz=1280,
        batch=16,
        device=DEVICE,
        workers=8,
        amp=True,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=False,
        seed=0,
        deterministic=True,
        optimizer="auto",
        lr0=0.003,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.1,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        val=True,
        plots=True,
        max_det=300,
        verbose=True,
        save=True,
        save_period=-1,
    )

    print("Training completed.")
    print(f"Results directory: {results.save_dir}")


if __name__ == "__main__":
    main()
