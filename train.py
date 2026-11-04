from ultralytics import YOLO


model = YOLO(
    "ultralytics/cfg/models/26/yolo26n-msneck.yaml"
)

model.load("yolo26n.pt")

results = model.train(
    data="all.yaml",
    epochs=500,
    imgsz=1280,
    name="yolo26n-msneck",
    device=5,
    batch=16,
    amp=True,
    patience=100,
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
    scale=0.25,
    shear=0.0,
    perspective=0.0,
    fliplr=0.0,
    flipud=0.0,
    max_det=300,
    plots=True,
    val=True,
)