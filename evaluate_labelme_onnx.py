import csv
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

# =========================================================
# CONFIGURATION & PATH RESOLUTION
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(SCRIPT_DIR, p)


MODEL_PATH = resolve_path("massdet_last_20251118.onnx")
DATA_ROOT = resolve_path("標註影像資料")
OUTPUT_FOLDER = resolve_path("eval_results_massdet_onnx")

# Evaluation Thresholds (Matching project standard in evaluate.py)
TARGET_IMG_SIZE = 1280
CONF_THRES = 0.5
IOU_THRES = 0.4
GT_IOU_THRES = 0.2
DRAW_ANNOTATIONS = True
END2END = True

# Removed file patterns as specified by project standard and user requirement
REMOVED_PATTERNS = ["MG_1_1_", "MG_745_1_1", "MG_0000"]

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Preprocessing Option: Without CLAHE
APPLY_CLAHE = False


# =========================================================
# PREPROCESSING FUNCTIONS (No CLAHE when APPLY_CLAHE=False)
# =========================================================
def get_holes(gray, thresh):
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw_inv = cv2.bitwise_not(im_bw)
    contours, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)
    nt = cv2.bitwise_not(im_bw)
    im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)
    return im_bw_inv


def remove_background(image, thresh, scale_factor=0.25, kernel_range=range(1, 15), border=None):
    border = border or list(kernel_range)[-1]
    holes = get_holes(image, thresh)
    small = cv2.resize(holes, None, fx=scale_factor, fy=scale_factor)
    bordered = cv2.copyMakeBorder(small, border, border, border, border, cv2.BORDER_CONSTANT)

    for i in kernel_range:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * i + 1, 2 * i + 1))
        bordered = cv2.morphologyEx(bordered, cv2.MORPH_CLOSE, kernel)

    unbordered = bordered[border:-border, border:-border]
    mask = cv2.resize(unbordered, (image.shape[1], image.shape[0]))
    fg = cv2.bitwise_and(image, image, mask=mask)
    return fg, mask


def check_lr_flip(mask):
    _, ncols = mask.shape
    x_center = ncols // 2
    col_sum = mask.sum(axis=0)
    left_sum = np.sum(col_sum[:x_center])
    right_sum = np.sum(col_sum[x_center:])
    return left_sum < right_sum


def make_lr_flip(img):
    return np.fliplr(img)


def global_binarise(img, thresh, maxval):
    out = np.zeros(img.shape, np.uint8)
    out[img >= thresh] = maxval
    return out


def edit_mask(mask, ksize=(23, 23), operation="open"):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    if operation == "open":
        edited = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        edited = mask.copy()
    edited = cv2.morphologyEx(edited, cv2.MORPH_DILATE, kernel)
    return edited


def sort_contours_by_area(contours, reverse=True):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
    return sorted_contours, bounding_boxes


def x_largest_blobs(mask, top_x=1, reverse=True):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    n_contours = len(contours)

    if n_contours > 0:
        if top_x is None or top_x > n_contours:
            top_x = n_contours
        sorted_contours, _ = sort_contours_by_area(contours, reverse=reverse)
        selected = sorted_contours[:top_x]
        canvas = np.zeros(mask.shape, np.uint8)
        blobs = cv2.drawContours(canvas, selected, -1, color=1, thickness=-1)
    else:
        blobs = mask

    return n_contours, blobs


def apply_mask(img, mask):
    out = img.copy()
    out[mask == 0] = 0
    return out


def flip_bboxes(bboxes, img_w):
    for b in bboxes:
        old_x = b[1]
        bw = b[3]
        b[1] = img_w - (old_x + bw)
    return bboxes


def pad_to_square(img):
    h, w = img.shape
    target = max(h, w)
    padded = np.zeros((target, target), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


def crop_pad(img, bboxes):
    idxs = np.where(img > 10)
    if len(idxs[0]) == 0 or len(idxs[1]) == 0:
        return pad_to_square(img), bboxes

    min_x = int(np.min(idxs[1]))
    max_x = int(np.max(idxs[1]))
    min_y = int(np.min(idxs[0]))
    max_y = int(np.max(idxs[0]))

    cropped = img[min_y : max_y + 1, min_x : max_x + 1]

    for b in bboxes:
        b[1] -= min_x
        b[2] -= min_y

    square = pad_to_square(cropped)
    sq_h, sq_w = square.shape

    cleaned = []
    for cls_id, x, y, w, h in bboxes:
        x1 = np.clip(x, 0, sq_w - 1)
        y1 = np.clip(y, 0, sq_h - 1)
        x2 = np.clip(x + w, 0, sq_w - 1)
        y2 = np.clip(y + h, 0, sq_h - 1)
        nw = float(max(0, x2 - x1))
        nh = float(max(0, y2 - y1))
        if nw > 0 and nh > 0:
            cleaned.append([cls_id, float(x1), float(y1), nw, nh])

    return square, cleaned


def preprocess_image_and_boxes(img, bboxes, target_img_size=1280, is_hospital=True, apply_clahe=False):
    """Standard preprocessing without CLAHE for hospital mammography images. - Background removal and mask cleaning -
    Left-right breast orientation alignment - Tight cropping and 1:1 square padding - Bbox translation and scaling
    to target_img_size (1280).
    """
    if len(img.shape) == 3:
        img = img[:, :, 0]
    bboxes = [b.copy() for b in bboxes]

    _, mask = remove_background(img, 0)
    img = img * np.where(mask < 10, 1, 0).astype("uint8")

    thresh_val = 0.05
    ksize_val = (23, 23)

    norm_img = img.astype("float32") / 255.0
    binarised = global_binarise(norm_img, thresh=thresh_val, maxval=1)
    edited_mask = edit_mask(binarised, ksize=ksize_val, operation="open")
    _, largest_mask = x_largest_blobs(edited_mask, top_x=1, reverse=True)

    img = apply_mask(img, largest_mask)
    masked_img = apply_mask(norm_img, largest_mask)

    if check_lr_flip(masked_img):
        img = make_lr_flip(img)
        bboxes = flip_bboxes(bboxes, img.shape[1])

    if apply_clahe:
        clip_limit = 2.0
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype("uint8")
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        img = clahe_obj.apply(img)

    img, bboxes = crop_pad(img, bboxes)

    h, w = img.shape
    scale_x = target_img_size / float(w)
    scale_y = target_img_size / float(h)

    if w > target_img_size:
        img = cv2.resize(img, (target_img_size, target_img_size), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (target_img_size, target_img_size), interpolation=cv2.INTER_LANCZOS4)

    for b in bboxes:
        b[1] *= scale_x
        b[2] *= scale_y
        b[3] *= scale_x
        b[4] *= scale_y

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    H, W = img.shape[:2]
    cleaned = []
    for cls_id, x, y, w, h in bboxes:
        x1 = np.clip(x, 0, W - 1)
        y1 = np.clip(y, 0, H - 1)
        x2 = np.clip(x + w, 0, W - 1)
        y2 = np.clip(y + h, 0, H - 1)
        nw = float(max(0, x2 - x1))
        nh = float(max(0, y2 - y1))
        if nw > 0 and nh > 0:
            cleaned.append([cls_id, float(x1), float(y1), nw, nh])

    return img, cleaned


# =========================================================
# EVALUATION METRICS HELPERS
# =========================================================
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between box1 and box2. Both boxes are in [x, y, w, h] absolute pixel
    format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0.0, xi2 - xi1)
    inter_height = max(0.0, yi2 - yi1)
    inter_area = inter_width * inter_height

    union_area = w1 * h1 + w2 * h2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def load_labelme_gt(json_path):
    """Read a Labelme .json file and extract bounding boxes for class 'mass'. Converts polygon/rectangle points into [0,
    x, y, w, h] absolute pixel format.
    """
    bboxes = []
    if not os.path.exists(json_path):
        return bboxes

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip().lower()
            if label == "mass":
                pts = shape.get("points", [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                w = max_x - min_x
                h = max_y - min_y
                if w > 0 and h > 0:
                    bboxes.append([0, float(min_x), float(min_y), float(w), float(h)])
    except Exception as e:
        print(f"[WARNING] Failed to parse JSON {json_path}: {e}")

    return bboxes


def collect_dataset_by_subfolder(root_dir, removed_patterns=REMOVED_PATTERNS):
    """Recursively collect image and Labelme JSON paths, grouped by top subfolder under root_dir."""
    subfolders_data = {}

    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root).lower() == "labelme_annotations":
            continue

        labelme_dir = os.path.join(root, "labelme_annotations")
        rel_root = os.path.relpath(root, root_dir)
        top_folder = rel_root.split(os.sep)[0]

        if top_folder == ".":
            continue

        if top_folder not in subfolders_data:
            subfolders_data[top_folder] = {"items": [], "removed_files": []}

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                img_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]

                if any(pattern in f for pattern in removed_patterns):
                    subfolders_data[top_folder]["removed_files"].append((f, img_path))
                    continue

                json_path = os.path.join(labelme_dir, stem + ".json")
                subfolders_data[top_folder]["items"].append(
                    {
                        "file_name": f,
                        "img_path": img_path,
                        "json_path": json_path,
                        "rel_dir": rel_root,
                        "subfolder": top_folder,
                    }
                )

    return subfolders_data


def compute_metrics_from_preds(all_preds, gt_count, tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    map50 = 0.0
    if len(all_preds) > 0 and gt_count > 0:
        all_preds_sorted = sorted(all_preds, key=lambda x: x["conf"], reverse=True)
        tps_arr = np.array([p["tp"] for p in all_preds_sorted])
        fps_arr = 1 - tps_arr
        cum_tps = np.cumsum(tps_arr)
        cum_fps = np.cumsum(fps_arr)
        recalls_arr = cum_tps / gt_count
        precisions_arr = cum_tps / np.maximum(cum_tps + cum_fps, np.finfo(float).eps)

        recalls_arr = np.concatenate(([0.0], recalls_arr, [1.0]))
        precisions_arr = np.concatenate(([0.0], precisions_arr, [0.0]))

        for i in range(len(precisions_arr) - 2, -1, -1):
            precisions_arr[i] = max(precisions_arr[i], precisions_arr[i + 1])

        indices = np.where(recalls_arr[1:] != recalls_arr[:-1])[0]
        map50 = np.sum((recalls_arr[indices + 1] - recalls_arr[indices]) * precisions_arr[indices + 1])

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "mAP50": round(float(map50), 4),
    }


# =========================================================
# MAIN EVALUATION ROUTINE
# =========================================================
def evaluate_model(max_images=None, output_folder=OUTPUT_FOLDER):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("=== Starting Per-Folder Model Evaluation (Preprocessing without CLAHE) ===")
    print(f"Model Path        : {MODEL_PATH}")
    print(f"Data Root         : {DATA_ROOT}")
    print(f"Removed Patterns  : {REMOVED_PATTERNS}")
    print(f"Apply CLAHE       : {APPLY_CLAHE}")
    print(f"Confidence Thres  : {CONF_THRES}")
    print(f"GT IoU Threshold  : {GT_IOU_THRES}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    print("Loading ONNX model...")
    model = YOLO(MODEL_PATH, task="detect")

    subfolders_data = collect_dataset_by_subfolder(DATA_ROOT, REMOVED_PATTERNS)
    sorted_subfolder_names = sorted(subfolders_data.keys())

    print(f"\nFound {len(sorted_subfolder_names)} subfolders inside '{DATA_ROOT}':")
    for sf in sorted_subfolder_names:
        v_cnt = len(subfolders_data[sf]["items"])
        r_cnt = len(subfolders_data[sf]["removed_files"])
        print(f"  - {sf:<30}: {v_cnt} valid images, {r_cnt} removed images")

    folder_summaries = {}
    overall_all_preds = []
    overall_gt_count = 0
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    overall_eval_images = 0
    overall_removed_images = 0

    all_csv_rows = []
    all_img_records = []

    for sf in sorted_subfolder_names:
        sf_items = subfolders_data[sf]["items"]
        sf_removed = subfolders_data[sf]["removed_files"]
        overall_removed_images += len(sf_removed)

        if max_images is not None:
            sf_items = sf_items[:max_images]

        if not sf_items:
            continue

        print(f"\nEvaluating folder: [{sf}] ({len(sf_items)} images)...")
        sf_metrics = {"TP": 0, "FP": 0, "FN": 0}
        sf_gt_count = 0
        sf_preds = []

        vis_folder = os.path.join(output_folder, "visualizations", sf)
        if DRAW_ANNOTATIONS and not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        for item in tqdm(sf_items, desc=f"Evaluating {sf}"):
            img_path = item["img_path"]
            json_path = item["json_path"]
            file_name = item["file_name"]
            stem = os.path.splitext(file_name)[0]

            img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                print(f"[WARNING] Could not read image: {img_path}")
                continue

            # Load ground truth bounding boxes in absolute pixels [0, x, y, w, h]
            gt_boxes_raw = load_labelme_gt(json_path)

            # Preprocess image and GT boxes without CLAHE
            prep_img, prep_boxes_cls = preprocess_image_and_boxes(
                img=img_raw,
                bboxes=gt_boxes_raw,
                target_img_size=TARGET_IMG_SIZE,
                is_hospital=True,
                apply_clahe=APPLY_CLAHE,
            )

            p_bboxes_gt = [[b[1], b[2], b[3], b[4]] for b in prep_boxes_cls]
            sf_gt_count += len(p_bboxes_gt)

            # Run ONNX inference on preprocessed 1280x1280 image
            results = model(
                prep_img, imgsz=TARGET_IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, end2end=END2END, verbose=False
            )
            result = results[0]

            pred_bboxes = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for box, conf in zip(xyxy, confs):
                    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                    pred_bboxes.append([float(x), float(y), float(w), float(h), float(conf)])

            # Match predictions to GT boxes
            pred_info_list = []
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
                        sf_metrics["TP"] += 1
                        matched_gt.add(best_gt_idx)
                        sf_preds.append({"conf": conf, "tp": 1})
                        overall_all_preds.append({"conf": conf, "tp": 1})
                        pred_info_list.append((px, py, pw, ph, conf, best_iou, 1))
                    else:
                        sf_metrics["FP"] += 1
                        sf_preds.append({"conf": conf, "tp": 0})
                        overall_all_preds.append({"conf": conf, "tp": 0})
                        pred_info_list.append((px, py, pw, ph, conf, best_iou, 0))

                sf_metrics["FN"] += len(p_bboxes_gt) - len(matched_gt)
            else:
                sf_metrics["FP"] += len(pred_bboxes)
                for pred in pred_bboxes:
                    sf_preds.append({"conf": pred[4], "tp": 0})
                    overall_all_preds.append({"conf": pred[4], "tp": 0})
                    pred_info_list.append((pred[0], pred[1], pred[2], pred[3], pred[4], 0.0, 0))

            img_tp = sum(p[6] for p in pred_info_list)
            img_fp = sum(1 - p[6] for p in pred_info_list)
            img_fn = max(0, len(p_bboxes_gt) - img_tp)

            for p in pred_info_list:
                all_csv_rows.append(
                    {
                        "subfolder": sf,
                        "image": file_name,
                        "folder": item["rel_dir"],
                        "pred_x": round(p[0], 2),
                        "pred_y": round(p[1], 2),
                        "pred_w": round(p[2], 2),
                        "pred_h": round(p[3], 2),
                        "conf": round(p[4], 4),
                        "best_iou": round(p[5], 4),
                        "tp": p[6],
                    }
                )

            all_img_records.append(
                {
                    "subfolder": sf,
                    "image": file_name,
                    "folder": item["rel_dir"],
                    "gt_boxes": len(p_bboxes_gt),
                    "tp": img_tp,
                    "fp": img_fp,
                    "fn": img_fn,
                    "predictions": [
                        {
                            "x": round(p[0], 2),
                            "y": round(p[1], 2),
                            "w": round(p[2], 2),
                            "h": round(p[3], 2),
                            "conf": round(p[4], 4),
                            "iou": round(p[5], 4),
                            "tp": p[6],
                        }
                        for p in pred_info_list
                    ],
                }
            )

            # Draw Visualization
            if DRAW_ANNOTATIONS:
                annotated_img = prep_img.copy()
                for p in pred_info_list:
                    px, py, pw, ph, conf, best_iou, tp_flag = p
                    color = (0, 255, 0) if tp_flag else (0, 165, 255)
                    cv2.rectangle(annotated_img, (int(px), int(py)), (int(px + pw), int(py + ph)), color, 2)
                    label = f"Mass: {conf:.2f}"
                    cv2.putText(
                        annotated_img,
                        label,
                        (int(px), max(int(py) - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                for gt_box in p_bboxes_gt:
                    gx, gy, gw, gh = gt_box
                    cv2.rectangle(annotated_img, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), (255, 0, 0), 2)
                    cv2.putText(
                        annotated_img,
                        "GT Mass",
                        (int(gx), int(gy) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                out_img_path = os.path.join(vis_folder, stem + "_eval.jpg")
                cv2.imwrite(out_img_path, annotated_img)

        # Folder-level metrics computation
        sf_tp, sf_fp, sf_fn = sf_metrics["TP"], sf_metrics["FP"], sf_metrics["FN"]
        sf_res = compute_metrics_from_preds(sf_preds, sf_gt_count, sf_tp, sf_fp, sf_fn)

        folder_summaries[sf] = {
            "total_images": len(sf_items),
            "removed_images": len(sf_removed),
            "gt_boxes": sf_gt_count,
            "TP": sf_tp,
            "FP": sf_fp,
            "FN": sf_fn,
            "precision": sf_res["precision"],
            "recall": sf_res["recall"],
            "f1": sf_res["f1"],
            "mAP50": sf_res["mAP50"],
        }

        overall_tp += sf_tp
        overall_fp += sf_fp
        overall_fn += sf_fn
        overall_gt_count += sf_gt_count
        overall_eval_images += len(sf_items)

    # Compute overall combined metrics across all subfolders
    overall_res = compute_metrics_from_preds(overall_all_preds, overall_gt_count, overall_tp, overall_fp, overall_fn)

    full_summary = {
        "model": MODEL_PATH,
        "data_root": DATA_ROOT,
        "removed_patterns": REMOVED_PATTERNS,
        "apply_clahe": APPLY_CLAHE,
        "overall_metrics": {
            "total_images_evaluated": overall_eval_images,
            "total_images_removed": overall_removed_images,
            "total_gt_boxes": overall_gt_count,
            "TP": overall_tp,
            "FP": overall_fp,
            "FN": overall_fn,
            "precision": overall_res["precision"],
            "recall": overall_res["recall"],
            "f1": overall_res["f1"],
            "mAP50": overall_res["mAP50"],
        },
        "folder_metrics": folder_summaries,
        "images": all_img_records,
    }

    # Save JSON summary
    json_path = os.path.join(output_folder, "results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON summary -> {json_path}")

    # Save Per-Folder Summary CSV
    folder_csv_path = os.path.join(output_folder, "folder_summary.csv")
    folder_csv_fields = [
        "subfolder",
        "total_images",
        "removed_images",
        "gt_boxes",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "mAP50",
    ]
    with open(folder_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=folder_csv_fields)
        writer.writeheader()
        for sf, metrics in folder_summaries.items():
            row = {"subfolder": sf}
            row.update(metrics)
            writer.writerow(row)
        # Add overall summary row
        overall_row = {
            "subfolder": "OVERALL",
            "total_images": overall_eval_images,
            "removed_images": overall_removed_images,
            "gt_boxes": overall_gt_count,
            "TP": overall_tp,
            "FP": overall_fp,
            "FN": overall_fn,
            "precision": overall_res["precision"],
            "recall": overall_res["recall"],
            "f1": overall_res["f1"],
            "mAP50": overall_res["mAP50"],
        }
        writer.writerow(overall_row)
    print(f"Saved Folder Summary CSV -> {folder_csv_path}")

    # Save Box-level CSV
    box_csv_path = os.path.join(output_folder, "results_per_box.csv")
    box_csv_fields = ["subfolder", "image", "folder", "pred_x", "pred_y", "pred_w", "pred_h", "conf", "best_iou", "tp"]
    with open(box_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=box_csv_fields)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"Saved Per-Box CSV        -> {box_csv_path}")

    # Print Console Summary Table
    print("\n" + "=" * 115)
    print(f"{'SUBFOLDER EVALUATION SUMMARY (No CLAHE Preprocessing)':^115}")
    print("=" * 115)
    header = f"{'Subfolder':<28} | {'Images':<6} | {'Removed':<7} | {'GT':<5} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | {'mAP50':<6}"
    print(header)
    print("-" * 115)

    for sf, m in folder_summaries.items():
        print(
            f"{sf:<28} | {m['total_images']:<6} | {m['removed_images']:<7} | {m['gt_boxes']:<5} | {m['TP']:<4} | {m['FP']:<4} | {m['FN']:<4} | {m['precision']:<9.4f} | {m['recall']:<7.4f} | {m['f1']:<6.4f} | {m['mAP50']:<6.4f}"
        )

    print("-" * 115)
    m = overall_res
    print(
        f"{'OVERALL (ALL FOLDERS)':<28} | {overall_eval_images:<6} | {overall_removed_images:<7} | {overall_gt_count:<5} | {m['TP']:<4} | {m['FP']:<4} | {m['FN']:<4} | {m['precision']:<9.4f} | {m['recall']:<7.4f} | {m['f1']:<6.4f} | {m['mAP50']:<6.4f}"
    )
    print("=" * 115)
    print(f"Results saved to directory: {output_folder}")

    return full_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate massdet_last_20251118.onnx on each folder in 標註影像資料 (Preprocessing without CLAHE)"
    )
    parser.add_argument(
        "--max_images", type=int, default=None, help="Optional max images limit per folder for quick test run"
    )
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER, help="Output directory")
    args = parser.parse_args()

    evaluate_model(max_images=args.max_images, output_folder=args.output_folder)
