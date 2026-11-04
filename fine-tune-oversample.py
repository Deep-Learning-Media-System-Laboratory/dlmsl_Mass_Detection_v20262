import os

# ---------------------------------------------------------
# Cross-Domain MixStyle domain manifest
# ---------------------------------------------------------
os.environ["CDMS_DOMAIN_MANIFEST"] = (
    "/home/yeozekiel/yolo/YOLO-MS/ultralytics/"
    # "copypaste_q1_training_2/domain_manifest_copypaste.csv"
    # "gan100_domain_manifest.csv"
    "allsup100_domain_manifest.csv"
    # "sim75_domain_manifest.csv"
)

# os.environ["CDMS_ALLOW_SINGLE_DOMAIN"]="1"


from ultralytics import YOLO

# Replace this import with the filename where
# you saved DomainMixStyleTrainer.

from cd_mixstyle_trainer import DomainMixStyleTrainer


BEST_MODEL = (
    "runs/detect/runs/yolo26_cd_mixstyle/"
    "cd_mixstyle_layer2_layer4_no_balance/"
    # "cd_mixstyle_no_balance_hospital_selected_finetune_v2/"
    "weights/best.pt"
)


model = YOLO(BEST_MODEL)


model.train(
    trainer=DomainMixStyleTrainer,

    data="all-supress100.yaml",
    # data="/home/yeozekiel/yolo/feature_selected_40_v2/feature_selected.yaml",

    epochs=200,
    imgsz=1280,
    batch=16,

    device=4,
    workers=4,

    # -------------------------
    # Keep your existing setup
    # -------------------------

    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    erasing=0.0,

    hsv_v=0.1,
    scale=0.25,

    fliplr=0.0,
    flipud=0.0,

    # IMPORTANT
    # This is a new fine-tune,
    # not resume.
    resume=False,

    project="runs/yolo26_cd_mixstyle",

    name=(
        "cd_mixstyle_no_balance_"
        # "hospital_suppress100_finetune"
        "all_supress100_finetune"
        # "cp10000_finetune"
    ),
)