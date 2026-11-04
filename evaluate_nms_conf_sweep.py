import os
import csv
import json
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
# from inference_all import preprocess_image

# ==== CONFIGURATIONS ====
CATEGORY_NAME = "mass"
TARGET_IMG_SIZE = 1280
# Sweep settings
NMS_IOU_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]
CONF_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

# Run inference at the lowest confidence once per NMS IoU, then filter upward.
INFER_CONF_FLOOR = min(CONF_THRESHOLDS)

# Keep GT matching fixed while testing NMS IoU / confidence.
GT_IOU_THRES = 0.2
DO_PREPROCESSING = False
DRAW_ANNOTATIONS = False  # Recommended False during a 5x5 sweep
END2END=False
NMS = True
DEVICE="0"

# MODEL_PATH = "../../best-with-embed-cmmd-normal-2.pt"
# MODEL_PATH = "massdet_last_20251118.onnx"
# MODEL_PATH = "runs/detect/all-pseudo-finetune-v12/weights/best.pt"
# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/no-cmmd-embed-medical-v1/weights/best.pt"
# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/all-v2/weights/best.pt"
# MODEL_PATH = "/home/yeozekiel/yolo/runs/detect/all-pseudo-finetune-v2"
# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/with-embed-cmmd-normal3/weights/best.pt"
# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/with-embed-cmmd-normal_bloss10_dfl23/weights/best.pt"
# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/patch512_finetune_from_exp1_best/weights/best.pt"
# MODEL_PATH = "best-fine-tuned-2.onnx"
# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/focal_finetune/weights/best.pt"
# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/focal_finetune_g1.0_a0.25_/weights/best.pt"
# MODEL_PATH = "best-with-embed-cmmd-normal.onnx"
# MODEL_PATH = "/home/yeozekiel/yolo/YOLO-MS/ultralytics/runs/detect/runs/detect/with-embed-cmmd-normal_yolo26n-deeplite-gate01-4/weights/best.pt"

# MODEL_PATH = "/home/yeozekiel/yolo/YOLO-MS/ultralytics/runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_layer2_layer4-2/weights/best.pt" #Balanced CD
# MODEL_PATH = "/home/yeozekiel/yolo/YOLO-MS/ultralytics/runs/detect/runs/yolo26_mixstyle/mixstyle_layer2_layer4-2/weights/best.pt" #Random
# MODEL_PATH = "runs/detect/runs/yolo26_mixstyle/mixstyle_layer2_layer4-no_hos_legacy-3/weights/best.pt" # Random Refined No hos
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_layer2_layer4_no_balance/weights/best.pt" # CD
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle_oversample/cd_mixstyle_no_balance_oversample4x_finetune-3/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle_augmented_oversample/cdms_no_balance_policy_augoversample2x_finetune/weights/best.pt"

# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_copypaste_finetune-3/weights/best.pt"

# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_finetune-3/weights/best.pt"

# MODEL_PATH = "/data/nas07_new/PersonalData/yeozekiel/yolo26_new/ultralytics/runs/detect/final_finetune_runs/best_policy_full_finetune/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_layer2_layer4_new-2/weights/best.pt" # Retraining with Benchmark
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_cp10000_finetune-2/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_finetune-4/weights/best.pt" 
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_selected_finetune/weights/best.pt" 

# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_layer2_layer4_new_retrain-4/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_selected_finetune_v2/weights/best.pt"

# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_layer2_layer4_new_retrain/weights/best.pt"

# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_new_finetune_v2/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_selected_finetune_v2-2/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_gan75_finetune/weights/best.pt"

# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_new_finetune_v2/weights/best.onnx"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_gan75_finetune-2/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_gan50_finetune-2/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_gan100_finetune-2/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_suppress100_finetune-2/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_hospital_simulate75_finetune/weights/best.pt"
MODEL_PATH = "runs/detect/runs/yolo26_cd_mixstyle/cd_mixstyle_no_balance_all_supress100_finetune/weights/best.pt"


# YOLO dataset directory — expects:
#   <VAL_DIR>/images/<image>.jpg
  # <VAL_DIR>/labels/<image>.txt
# VAL_DIR = r"../../dataset_without_embed_cmmd_refined"
# VAL_DIR = r"../../hospital_yolo_dataset_v3"
# VAL_DIR = r"../../hospital_yolo_dataset_v5"
VAL_DIR = "/home/yeozekiel/yolo/hospital_yolo_dataset_v5_suppress100"
# VAL_DIR = "/home/yeozekiel/yolo/hospital_yolo_dataset_v5_simulate75"
# VAL_DIR = "../../hospital_suppress_50_test/lesion_cycle_ganA/test_50"
# VAL_DIR = "../../harmonized_2"
# VAL_DIR = "../../hos_harmonized"
# VAL_DIR = "../../dataset_phuc_legacy_preprocessed_no_clahe_cleaned"
# VAL_DIR = "../../dataset_with_hospital_cleaned"
# VAL_DIR = "dataset_preprocessed_v4"
# VAL_DIR = r"/data/nas07_new/PersonalData/yeozekiel/yolo26_new/dataset_preprocessed"
# VAL_DIR = r"dataset_patched"
# VAL_DIR = r"yolo26_new/sbbdem_preprocessing"
# VAL_DIR = r"/data/nas07_new/PersonalData/yeozekiel/dataset_cbis_ddsm"
# SPLIT = "newbenchmark_v2.2"  # Options: "train", "val", "test" , newbenchmark_v2.2"
SPLIT = "test"

OUTPUT_PARENT = "evaluation_results"
OUTPUT_FOLDER = f"{OUTPUT_PARENT}/all-result_suppress100" #_selected_retrain_old

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ==== HELPER FUNCTIONS ====
def calculate_iou(box1, box2):
    """Both boxes in [x, y, w, h] absolute pixel format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / float(union_area)


def load_yolo_labels(label_path, img_w, img_h):
    """
    Read a YOLO label .txt file and convert normalised [cx, cy, w, h]
    to absolute pixel [x_min, y_min, w, h].

    Returns a list of bounding boxes, one per annotation line.
    Lines with no annotation (empty files = negative images) return [].
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes  # image has no label file → treat as negative

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # class_id cx cy w h  (all normalised)
            _cls, cx, cy, nw, nh = parts[:5]
            cx, cy, nw, nh = float(cx), float(cy), float(nw), float(nh)

            # Convert to absolute [x_min, y_min, w, h]
            abs_w = nw * img_w
            abs_h = nh * img_h
            abs_x = cx * img_w - abs_w / 2.0
            abs_y = cy * img_h - abs_h / 2.0
            bboxes.append([abs_x, abs_y, abs_w, abs_h])

    return bboxes

# Preprocessing helpers
def get_holes(gray, thresh):
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw_inv = cv2.bitwise_not(im_bw)
    contour, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)
    nt = cv2.bitwise_not(im_bw)
    im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)
    return im_bw_inv

def remove_background(image, thresh, scale_factor=.25, kernel_range=range(1, 15), border=None):
    border = border or kernel_range[-1]
    holes = get_holes(image, thresh)
    small = cv2.resize(holes, None, fx=scale_factor, fy=scale_factor)
    bordered = cv2.copyMakeBorder(small, border, border, border, border, cv2.BORDER_CONSTANT)
    for i in kernel_range:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * i + 1, 2 * i + 1))
        bordered = cv2.morphologyEx(bordered, cv2.MORPH_CLOSE, kernel)
    unbordered = bordered[border: -border, border: -border]
    mask = cv2.resize(unbordered, (image.shape[1], image.shape[0]))
    fg = cv2.bitwise_and(image, image, mask=mask)
    return fg, mask

def checkLRFlip(mask):
    nrows, ncols = mask.shape
    x_center = ncols // 2
    col_sum = mask.sum(axis=0)
    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])
    return left_sum < right_sum

def makeLRFlip(img):
    return np.fliplr(img)

def globalBinarise(img, thresh, maxval):
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval
    return binarised_img

def editMask(mask, ksize=(23, 23), operation="open"):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)
    return edited_mask

def sortContoursByArea(contours, reverse=True):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
    return sorted_contours, bounding_boxes

def xLargestBlobs(mask, top_x=None, reverse=True):
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    n_contours = len(contours)
    if n_contours > 0:
        if n_contours < top_x or top_x is None:
            top_x = n_contours
        sorted_contours, _ = sortContoursByArea(contours=contours, reverse=reverse)
        X_largest_contours = sorted_contours[0:top_x]
        to_draw_on = np.zeros(mask.shape, np.uint8)
        X_largest_blobs = cv2.drawContours(image=to_draw_on, contours=X_largest_contours, contourIdx=-1, color=1, thickness=-1)
    else:
        X_largest_blobs = mask
    return n_contours, X_largest_blobs

def applyMask(img, mask):
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img

def clahe(img, clip=2.0, tile=(8, 8)):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_uint8 = img.astype("uint8")
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)
    return clahe_img

def flip_bboxes(bboxes, img_w):
    if bboxes is not None:
        if isinstance(bboxes, list):
            for i in range(len(bboxes)):
                # bboxes shape: [x, y, w, h]
                old_x = bboxes[i][0]
                w = bboxes[i][2]
                bboxes[i][0] = img_w - (old_x + w)
        else:
            # Assuming numpy array
            old_x = bboxes[:, 0].copy()
            w = bboxes[:, 2].copy()
            bboxes[:, 0] = img_w - (old_x + w)
    return bboxes

def pad(img, pad_size):
    nrows, ncols = img.shape
    target_shape = (nrows + pad_size[0], ncols + pad_size[1])
    padded_img = np.zeros(shape=target_shape, dtype=img.dtype)
    padded_img[:nrows, :ncols] = img
    return padded_img

def crop_pad(img, bboxes):
    idxs = np.where(img > 128)
    if len(idxs[1]) == 0 or len(idxs[0]) == 0:
        return img, bboxes, [0,0,0,0,0,0] # fallback

    min_x = min(idxs[1])
    max_x = max(idxs[1])
    min_y = min(idxs[0])
    max_y = max(idxs[0])
    img = img[min_y:max_y, min_x:max_x]

    if bboxes is not None:
        if isinstance(bboxes, list):
            for b in bboxes:
                b[0] -= min_x
                b[1] -= min_y
        else:
            # Assuming numpy array
            bboxes[:, 0] -= min_x
            bboxes[:, 1] -= min_y

    pad_size = (min(0, img.shape[0] - img.shape[1]) * -1, min(0, img.shape[1] - img.shape[0]) * -1)
    img = pad(img, pad_size)
    img_h, img_w = img.shape

    if bboxes is not None:
        if isinstance(bboxes, list):
            for b in bboxes:
                x1 = np.clip(b[0], 0, img_w - 1)
                y1 = np.clip(b[1], 0, img_h - 1)
                x2 = np.clip(b[0] + b[2], 0, img_w - 1)
                y2 = np.clip(b[1] + b[3], 0, img_h - 1)
                b[0] = x1
                b[1] = y1
                b[2] = float(max(0, x2 - x1))
                b[3] = float(max(0, y2 - y1))
        else:
            x1 = np.clip(bboxes[:, 0], 0, img_w - 1)
            y1 = np.clip(bboxes[:, 1], 0, img_h - 1)
            x2 = np.clip(bboxes[:, 0] + bboxes[:, 2], 0, img_w - 1)
            y2 = np.clip(bboxes[:, 1] + bboxes[:, 3], 0, img_h - 1)
            bboxes[:, 0] = x1
            bboxes[:, 1] = y1
            bboxes[:, 2] = np.maximum(0, x2 - x1)
            bboxes[:, 3] = np.maximum(0, y2 - y1)

    crop_pad_paras = [int(min_x), int(max_x), int(min_y), int(max_y), int(pad_size[0]), int(pad_size[1])]
    return img, bboxes, crop_pad_paras

def cropBorders(img, bboxes, l=0.01, r=0.01, u=0.04, d=0.04):
    try:
        nrows, ncols = img.shape
        l_crop = int(ncols * l)
        r_crop = int(ncols * (1 - r))
        u_crop = int(nrows * u)
        d_crop = int(nrows * (1 - d))

        cropped_img = img[u_crop:d_crop, l_crop:r_crop]
        if bboxes is not None:
            if isinstance(bboxes, list):
                 for b in bboxes:
                     b[0] -= l_crop
                     b[1] -= u_crop
                     x1 = np.clip(b[0], 0, r_crop - l_crop - 1)
                     y1 = np.clip(b[1], 0, d_crop - u_crop - 1)
                     x2 = np.clip(b[0] + b[2], 0, r_crop - l_crop - 1)
                     y2 = np.clip(b[1] + b[3], 0, d_crop - u_crop - 1)
                     b[0] = x1
                     b[1] = y1
                     b[2] = float(max(0, x2 - x1))
                     b[3] = float(max(0, y2 - y1))
            else:
                 bboxes[:, 0] -= l_crop
                 bboxes[:, 1] -= u_crop
                 x1 = np.clip(bboxes[:, 0], 0, r_crop - l_crop - 1)
                 y1 = np.clip(bboxes[:, 1], 0, d_crop - u_crop - 1)
                 x2 = np.clip(bboxes[:, 0] + bboxes[:, 2], 0, r_crop - l_crop - 1)
                 y2 = np.clip(bboxes[:, 1] + bboxes[:, 3], 0, d_crop - u_crop - 1)
                 bboxes[:, 0] = x1
                 bboxes[:, 1] = y1
                 bboxes[:, 2] = np.maximum(0, x2 - x1)
                 bboxes[:, 3] = np.maximum(0, y2 - y1)

        img = cropped_img
        crop_paras = [int(l_crop), int(r_crop), int(u_crop), int(d_crop)]
    except Exception as e:
        print((f"Unable to get cropBorders!\n{e}"))
        crop_paras = [0,0,0,0]
    return img, bboxes, crop_paras

def preprocess_image(img, bboxes_gt=None, do_masking=True):
    if len(img.shape) > 2:
        img = img[:, :, 0]
    orig_shape = img.shape
    # # keep track of scaling factors internally if needed but bounding boxes transform
    # if "DDSM" in BASE_DIR:  # Not dynamically handling base dir properly, skipped
    #     img, bboxes_gt, _ = cropBorders(img, bboxes_gt)
    
    nb_img, mask = remove_background(img, 0)
    img *= np.where(mask < 10, 1, 0).astype('uint8')
    
    norm_img = img.astype('float32') / 255.
    binarised_img = globalBinarise(img=norm_img, thresh=0.05, maxval=1.0)
    edited_mask = editMask(mask=binarised_img, ksize=(23, 23), operation="open")
    _, xlargest_mask = xLargestBlobs(mask=edited_mask, top_x=1, reverse=True)
    masked_img = applyMask(img=norm_img, mask=xlargest_mask)
    
    lr_flip = checkLRFlip(mask=masked_img)
    if lr_flip:
        img = makeLRFlip(img=masked_img)
        if bboxes_gt is not None:
             bboxes_gt = flip_bboxes(bboxes_gt, img.shape[1])
    
    img = clahe(img=img, clip=2.0, tile=(8, 8))
    img, bboxes_gt, _ = crop_pad(img, bboxes_gt)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    height, width = img.shape[:2]
    resize_factor_x = TARGET_IMG_SIZE / float(width)
    resize_factor_y = TARGET_IMG_SIZE / float(height)
    
    if width > TARGET_IMG_SIZE:
        img = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        
    # Scale boxes
    if bboxes_gt is not None:
        for b in bboxes_gt:
            b[0] *= resize_factor_x
            b[2] *= resize_factor_x  # Note: W 
            b[1] *= resize_factor_y
            b[3] *= resize_factor_y  # Note: H

    return img, bboxes_gt


def match_predictions(pred_bboxes, p_bboxes_gt):
    """
    Match predictions to GT using the existing GT_IOU_THRES logic.

    pred_bboxes: [[x, y, w, h, conf], ...]
    p_bboxes_gt: [[x, y, w, h], ...]

    Returns:
        tp, fp, fn, pred_info_list, all_preds_for_ap
    """
    pred_info_list = []
    all_preds_for_ap = []

    if p_bboxes_gt:
        matched_gt = set()
        pred_bboxes_sorted = sorted(pred_bboxes, key=lambda x: x[4], reverse=True)

        for pred in pred_bboxes_sorted:
            px, py, pw, ph, conf = pred
            best_iou = 0.0
            best_gt_idx = -1

            for i, gt_box in enumerate(p_bboxes_gt):
                if i in matched_gt:
                    continue

                iou = calculate_iou([px, py, pw, ph], gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou > GT_IOU_THRES:
                matched_gt.add(best_gt_idx)
                all_preds_for_ap.append({"conf": conf, "tp": 1})
                pred_info_list.append(
                    (px, py, pw, ph, conf, best_iou, 1)
                )
            else:
                all_preds_for_ap.append({"conf": conf, "tp": 0})
                pred_info_list.append(
                    (px, py, pw, ph, conf, best_iou, 0)
                )

        tp = len(matched_gt)
        fp = len(pred_bboxes_sorted) - tp
        fn = len(p_bboxes_gt) - tp

    else:
        # Negative image: every prediction is FP.
        tp = 0
        fp = len(pred_bboxes)
        fn = 0

        for pred in pred_bboxes:
            px, py, pw, ph, conf = pred
            all_preds_for_ap.append({"conf": conf, "tp": 0})
            pred_info_list.append(
                (px, py, pw, ph, conf, 0.0, 0)
            )

    return tp, fp, fn, pred_info_list, all_preds_for_ap


def calculate_ap_from_predictions(all_preds, gt_count):
    """
    Keeps the same PR integration style as the original evaluator.

    IMPORTANT:
    Because GT matching uses GT_IOU_THRES=0.2 by default, this is AP@0.2,
    not AP@0.5. The original script named it mAP50 even though matching
    was performed at IoU > 0.2.
    """
    if len(all_preds) == 0 or gt_count <= 0:
        return 0.0

    preds = sorted(all_preds, key=lambda x: x["conf"], reverse=True)

    tps_arr = np.array([p["tp"] for p in preds])
    fps_arr = 1 - tps_arr

    cum_tps = np.cumsum(tps_arr)
    cum_fps = np.cumsum(fps_arr)

    recalls_arr = cum_tps / gt_count
    precisions_arr = cum_tps / np.maximum(
        cum_tps + cum_fps,
        np.finfo(float).eps
    )

    recalls_arr = np.concatenate(([0.0], recalls_arr, [1.0]))
    precisions_arr = np.concatenate(([0.0], precisions_arr, [0.0]))

    for i in range(len(precisions_arr) - 2, -1, -1):
        precisions_arr[i] = max(
            precisions_arr[i],
            precisions_arr[i + 1]
        )

    indices = np.where(
        recalls_arr[1:] != recalls_arr[:-1]
    )[0]

    ap = np.sum(
        (recalls_arr[indices + 1] - recalls_arr[indices])
        * precisions_arr[indices + 1]
    )

    return float(ap)


def process_validation():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(task="detect", model=MODEL_PATH)

    if SPLIT != "all":
        images_dir = os.path.join(VAL_DIR, "images", SPLIT)
        labels_dir = os.path.join(VAL_DIR, "labels", SPLIT)
    else:
        images_dir = os.path.join(VAL_DIR, "images")
        labels_dir = os.path.join(VAL_DIR, "labels")

    supported_exts = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".tif", ".tiff"
    }

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in supported_exts
    ])

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images in '{VAL_DIR}'.")
    print(f"NMS IoU sweep : {NMS_IOU_THRESHOLDS}")
    print(f"Confidence sweep: {CONF_THRESHOLDS}")
    print(f"GT matching IoU stays fixed at: {GT_IOU_THRES}")
    print(f"Inference confidence floor: {INFER_CONF_FLOOR}")

    # One accumulator per (NMS IoU, confidence) pair.
    combo_data = {}

    for nms_iou in NMS_IOU_THRESHOLDS:
        for conf_thres in CONF_THRESHOLDS:
            combo_data[(nms_iou, conf_thres)] = {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "all_preds": [],
            }

    gt_count = 0
    evaluated_images = 0

    # ------------------------------------------------------------
    # Image outer loop:
    # preprocess each image once, then run each NMS setting.
    # ------------------------------------------------------------
    for file_name in tqdm(image_files, desc="Sweeping NMS IoU / confidence"):
        img_path = os.path.join(images_dir, file_name)
        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")

        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        bboxes_gt = load_yolo_labels(
            label_path,
            img_w,
            img_h
        )

        bboxes_gt_copy = [
            [x, y, w, h]
            for x, y, w, h in bboxes_gt
        ]

        # Preprocess ONCE for this image.
        if DO_PREPROCESSING:
            try:
                prep_img, p_bboxes_gt = preprocess_image(
                    img,
                    bboxes_gt_copy
                )
            except Exception as e:
                print(f"Error preprocessing {file_name}: {e}")
                continue
        else:
            prep_img = (
                img
                if len(img.shape) == 3
                else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            )
            p_bboxes_gt = bboxes_gt_copy

        evaluated_images += 1
        gt_count += len(p_bboxes_gt)

        # --------------------------------------------------------
        # Run the model once for EACH NMS IoU at conf=0.1.
        # --------------------------------------------------------
        for nms_iou in NMS_IOU_THRESHOLDS:

            results = model(
                prep_img,
                conf=INFER_CONF_FLOOR,
                iou=nms_iou,
                end2end=END2END,
                nms=NMS,
                imgsz=TARGET_IMG_SIZE,
                verbose=False,
                device=DEVICE,
            )

            result = results[0]

            all_pred_bboxes = []

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for box, conf in zip(xyxy, confs):
                    x = float(box[0])
                    y = float(box[1])
                    w = float(box[2] - box[0])
                    h = float(box[3] - box[1])

                    all_pred_bboxes.append(
                        [x, y, w, h, float(conf)]
                    )

            # ----------------------------------------------------
            # No additional inference is needed for higher conf.
            # Filter detections retained from conf=0.1.
            # ----------------------------------------------------
            for conf_thres in CONF_THRESHOLDS:

                pred_bboxes = [
                    pred
                    for pred in all_pred_bboxes
                    if pred[4] >= conf_thres
                ]

                (
                    tp,
                    fp,
                    fn,
                    _pred_info_list,
                    all_preds_for_ap,
                ) = match_predictions(
                    pred_bboxes,
                    p_bboxes_gt
                )

                data = combo_data[
                    (nms_iou, conf_thres)
                ]

                data["TP"] += tp
                data["FP"] += fp
                data["FN"] += fn
                data["all_preds"].extend(
                    all_preds_for_ap
                )

    # ------------------------------------------------------------
    # Build one summary row per combination.
    # ------------------------------------------------------------
    sweep_rows = []

    for nms_iou in NMS_IOU_THRESHOLDS:
        for conf_thres in CONF_THRESHOLDS:

            data = combo_data[
                (nms_iou, conf_thres)
            ]

            tp = data["TP"]
            fp = data["FP"]
            fn = data["FN"]

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            ap_at_gt_iou = calculate_ap_from_predictions(
                data["all_preds"],
                gt_count
            )

            sweep_rows.append({
                "nms_iou": round(nms_iou, 2),
                "conf": round(conf_thres, 2),
                "gt_match_iou": GT_IOU_THRES,
                "total_images": evaluated_images,
                "total_gt_boxes": gt_count,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "ap_at_gt_iou": round(ap_at_gt_iou, 6),
            })

    # ------------------------------------------------------------
    # Find highest F1.
    # ------------------------------------------------------------
    best = max(
        sweep_rows,
        key=lambda row: row["f1"]
    )

    # ------------------------------------------------------------
    # Save CSV.
    # ------------------------------------------------------------
    csv_path = os.path.join(
        OUTPUT_FOLDER,
        "nms_conf_sweep.csv"
    )

    csv_fields = [
        "nms_iou",
        "conf",
        "gt_match_iou",
        "total_images",
        "total_gt_boxes",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "ap_at_gt_iou",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields
        )
        writer.writeheader()
        writer.writerows(sweep_rows)

    # ------------------------------------------------------------
    # Save JSON.
    # ------------------------------------------------------------
    json_path = os.path.join(
        OUTPUT_FOLDER,
        "nms_conf_sweep.json"
    )

    json_output = {
        "dataset": VAL_DIR,
        "split": SPLIT,
        "model": MODEL_PATH,
        "end2end": END2END,
        "gt_match_iou": GT_IOU_THRES,
        "inference_conf_floor": INFER_CONF_FLOOR,
        "nms_iou_thresholds": NMS_IOU_THRESHOLDS,
        "confidence_thresholds": CONF_THRESHOLDS,
        "total_images": evaluated_images,
        "total_gt_boxes": gt_count,
        "best_by_f1": best,
        "results": sweep_rows,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            json_output,
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # Print matrix-like summary.
    # ------------------------------------------------------------
    print("\n" + "=" * 105)
    print("NMS IoU / Confidence Sweep")
    print("=" * 105)

    print(
        f"{'NMS IoU':>8} | "
        f"{'Conf':>6} | "
        f"{'TP':>5} | "
        f"{'FP':>5} | "
        f"{'FN':>5} | "
        f"{'Precision':>10} | "
        f"{'Recall':>10} | "
        f"{'F1':>10} | "
        f"{'AP@GTIoU':>10}"
    )

    print("-" * 105)

    for row in sweep_rows:
        print(
            f"{row['nms_iou']:>8.2f} | "
            f"{row['conf']:>6.2f} | "
            f"{row['TP']:>5d} | "
            f"{row['FP']:>5d} | "
            f"{row['FN']:>5d} | "
            f"{row['precision']:>10.4f} | "
            f"{row['recall']:>10.4f} | "
            f"{row['f1']:>10.4f} | "
            f"{row['ap_at_gt_iou']:>10.4f}"
        )

    print("\n" + "=" * 55)
    print("BEST COMBINATION BY F1")
    print("=" * 55)
    print(f"NMS IoU       : {best['nms_iou']:.2f}")
    print(f"Confidence    : {best['conf']:.2f}")
    print(f"GT Match IoU  : {best['gt_match_iou']:.2f}")
    print(f"TP            : {best['TP']}")
    print(f"FP            : {best['FP']}")
    print(f"FN            : {best['FN']}")
    print(f"Precision     : {best['precision']:.4f}")
    print(f"Recall        : {best['recall']:.4f}")
    print(f"F1            : {best['f1']:.4f}")
    print(f"AP@GTIoU      : {best['ap_at_gt_iou']:.4f}")
    print("=" * 55)

    print(f"\nCSV saved  -> {csv_path}")
    print(f"JSON saved -> {json_path}")

    print(
        "\nNOTE: GT_IOU_THRES is fixed during this experiment. "
        "Only NMS IoU and confidence are swept."
    )


if __name__ == "__main__":
    process_validation()