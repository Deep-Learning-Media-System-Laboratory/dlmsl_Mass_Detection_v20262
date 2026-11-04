"""
evaluate_scale.py.

Callable YOLO evaluation for SA-AutoAug search.

This file is adapted from your original evaluate (2).py script:
- MODEL_PATH, VAL_DIR, SPLIT, OUTPUT_FOLDER are now function arguments.
- It returns a summary dictionary instead of only printing/saving.
- It adds scale-wise metrics for small / medium / large lesions.
- It avoids calling the custom AP value "mAP50" when the GT IoU threshold is 0.2.
"""

from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

try:
    from inference_all import preprocess_image
except Exception:
    preprocess_image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Both boxes are [x, y, w, h] in absolute pixel format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> list[list[float]]:
    """Read YOLO label .txt and convert normalized [cx, cy, w, h] to absolute pixel [x_min, y_min, w, h].
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                _cls, cx, cy, nw, nh = parts[:5]
                cx, cy, nw, nh = float(cx), float(cy), float(nw), float(nh)
            except ValueError:
                continue

            abs_w = nw * img_w
            abs_h = nh * img_h
            abs_x = cx * img_w - abs_w / 2.0
            abs_y = cy * img_h - abs_h / 2.0

            if abs_w > 0 and abs_h > 0:
                bboxes.append([abs_x, abs_y, abs_w, abs_h])

    return bboxes


def get_scale_category_from_box(box: list[float]) -> str:
    _, _, w, h = box
    area = w * h
    if area < 9764:
        return "small"
    if area < 21864:
        return "medium"
    return "large"


def compute_ap_from_records(pred_records: list[dict], gt_count: int) -> float:
    """Compute AP from records containing {"conf": float, "tp": 0/1}. This uses the same precision-recall integration
    style as your original code.
    """
    if not pred_records or gt_count <= 0:
        return 0.0

    pred_records = sorted(pred_records, key=lambda x: x["conf"], reverse=True)
    tps_arr = np.array([p["tp"] for p in pred_records], dtype=np.float32)
    fps_arr = 1.0 - tps_arr

    cum_tps = np.cumsum(tps_arr)
    cum_fps = np.cumsum(fps_arr)

    recalls = cum_tps / max(gt_count, np.finfo(float).eps)
    precisions = cum_tps / np.maximum(cum_tps + cum_fps, np.finfo(float).eps)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def fbeta_score(precision: float, recall: float, beta: float = 2.0, eps: float = 1e-8) -> float:
    """Compute F-beta score.

    beta=1.0 gives F1. beta=2.0 gives F2, which weights recall more strongly.
    """
    precision = float(precision)
    recall = float(recall)

    if precision <= 0.0 and recall <= 0.0:
        return 0.0

    beta2 = beta**2
    return float((1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps))


def evaluate_model(
    model_path: str,
    val_dir: str,
    split: str = "val",
    output_folder: str | None = None,
    imgsz: int = 1280,
    conf_thres: float = 0.3,
    pred_iou_thres: float = 0.4,
    gt_iou_thres: float = 0.2,
    do_preprocessing: bool = False,
    draw_annotations: bool = False,
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict:
    """Evaluate YOLO model and return overall + scale-wise metrics.

    Directory expected: val_dir/images/<split> val_dir/labels/<split>
    """
    if output_folder is None:
        output_folder = f"{split}_result"

    if save_outputs:
        ensure_dir(output_folder)

    if do_preprocessing and preprocess_image is None:
        raise RuntimeError("do_preprocessing=True, but inference_all.preprocess_image could not be imported.")

    if verbose:
        print(f"[EVAL] Loading model: {model_path}")

    model = YOLO(task="detect", model=model_path)

    images_dir = os.path.join(val_dir, "images", split)
    labels_dir = os.path.join(val_dir, "labels", split)

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Image split folder not found: {images_dir}")

    image_files = sorted([f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    if verbose:
        print(f"[EVAL] Found {len(image_files)} images in {images_dir}")

    all_metrics = {"TP": 0, "FP": 0, "FN": 0}
    gt_count = 0
    all_pred_records = []

    scale_metrics = {
        "small": {"TP": 0, "FP": 0, "FN": 0, "gt_count": 0, "pred_records": []},
        "medium": {"TP": 0, "FP": 0, "FN": 0, "gt_count": 0, "pred_records": []},
        "large": {"TP": 0, "FP": 0, "FN": 0, "gt_count": 0, "pred_records": []},
    }

    csv_rows = []
    img_records = []

    iterator = tqdm(image_files, desc="Evaluating Images") if verbose else image_files

    for file_name in iterator:
        img_path = os.path.join(images_dir, file_name)
        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Failed to load image: {img_path}")
            continue

        if img.ndim == 2:
            img_for_model = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_for_model = img.copy()

        img_h, img_w = img_for_model.shape[:2]

        bboxes_gt = load_yolo_labels(label_path, img_w, img_h)
        bboxes_gt_copy = [[x, y, w, h] for x, y, w, h in bboxes_gt]

        if do_preprocessing:
            try:
                prep_img, p_bboxes_gt = preprocess_image(img_for_model, bboxes_gt_copy)
            except Exception as e:
                print(f"[WARN] Error preprocessing {file_name}: {e}")
                continue
        else:
            prep_img = img_for_model
            p_bboxes_gt = bboxes_gt_copy

        gt_scales = [get_scale_category_from_box(gt) for gt in p_bboxes_gt]
        for s in gt_scales:
            scale_metrics[s]["gt_count"] += 1

        gt_count += len(p_bboxes_gt)

        results = model(
            prep_img,
            conf=conf_thres,
            iou=pred_iou_thres,
            imgsz=imgsz,
            verbose=False,
        )
        result = results[0]

        pred_bboxes = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                x = float(box[0])
                y = float(box[1])
                w = float(box[2] - box[0])
                h = float(box[3] - box[1])
                if w > 0 and h > 0:
                    pred_bboxes.append([x, y, w, h, float(conf)])

        pred_info_list = []
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

            if best_iou > gt_iou_thres and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_scale = gt_scales[best_gt_idx]

                all_metrics["TP"] += 1
                scale_metrics[matched_scale]["TP"] += 1

                all_pred_records.append({"conf": conf, "tp": 1})
                scale_metrics[matched_scale]["pred_records"].append({"conf": conf, "tp": 1})

                pred_info_list.append((px, py, pw, ph, conf, best_iou, 1, matched_scale))
            else:
                pred_scale = get_scale_category_from_box([px, py, pw, ph])

                all_metrics["FP"] += 1
                scale_metrics[pred_scale]["FP"] += 1

                all_pred_records.append({"conf": conf, "tp": 0})
                scale_metrics[pred_scale]["pred_records"].append({"conf": conf, "tp": 0})

                pred_info_list.append((px, py, pw, ph, conf, best_iou, 0, pred_scale))

        for gt_idx, _ in enumerate(p_bboxes_gt):
            if gt_idx not in matched_gt:
                fn_scale = gt_scales[gt_idx]
                all_metrics["FN"] += 1
                scale_metrics[fn_scale]["FN"] += 1

        img_tp = sum(p[6] for p in pred_info_list)
        img_fp = sum(1 - p[6] for p in pred_info_list)
        img_fn = max(0, len(p_bboxes_gt) - img_tp)

        for p in pred_info_list:
            csv_rows.append(
                {
                    "image": file_name,
                    "pred_x": round(p[0], 2),
                    "pred_y": round(p[1], 2),
                    "pred_w": round(p[2], 2),
                    "pred_h": round(p[3], 2),
                    "conf": round(p[4], 4),
                    "best_iou": round(p[5], 4),
                    "tp": p[6],
                    "scale": p[7],
                }
            )

        img_records.append(
            {
                "image": file_name,
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
                        "scale": p[7],
                    }
                    for p in pred_info_list
                ],
            }
        )

        if draw_annotations and save_outputs:
            for p in pred_info_list:
                px, py, pw, ph, conf, best_iou, tp_flag, scale = p
                color = (0, 255, 0) if tp_flag else (0, 165, 255)
                cv2.rectangle(prep_img, (int(px), int(py)), (int(px + pw), int(py + ph)), color, 2)
                label = f"{scale} Conf:{conf:.2f} IoU:{best_iou:.2f}"
                cv2.putText(
                    prep_img,
                    label,
                    (int(px), max(int(py) - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            for gt_box in p_bboxes_gt:
                gx, gy, gw, gh = gt_box
                scale = get_scale_category_from_box(gt_box)
                cv2.rectangle(prep_img, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), (255, 0, 0), 2)
                cv2.putText(
                    prep_img,
                    f"GT-{scale}",
                    (int(gx), int(gy) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            out_path = os.path.join(output_folder, stem + "_annotated.jpg")
            cv2.imwrite(out_path, prep_img)

    tp = all_metrics["TP"]
    fp = all_metrics["FP"]
    fn = all_metrics["FN"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = fbeta_score(precision, recall, beta=1.0)
    f2 = fbeta_score(precision, recall, beta=2.0)
    custom_ap = compute_ap_from_records(all_pred_records, gt_count)

    scale_summary = {}
    scale_ap = {}
    scale_f1 = {}
    scale_f2 = {}
    scale_loss_proxy = {}
    scale_f2_loss_proxy = {}

    for scale, m in scale_metrics.items():
        s_tp, s_fp, s_fn = m["TP"], m["FP"], m["FN"]
        s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = fbeta_score(s_precision, s_recall, beta=1.0)
        s_f2 = fbeta_score(s_precision, s_recall, beta=2.0)
        s_ap = compute_ap_from_records(m["pred_records"], m["gt_count"])

        scale_summary[scale] = {
            "TP": s_tp,
            "FP": s_fp,
            "FN": s_fn,
            "gt_count": m["gt_count"],
            "precision": round(s_precision, 6),
            "recall": round(s_recall, 6),
            "f1": round(s_f1, 6),
            "f2": round(s_f2, 6),
            f"AP@IoU{gt_iou_thres}": round(s_ap, 6),
        }
        scale_ap[scale] = float(s_ap)
        scale_f1[scale] = float(s_f1)
        scale_f2[scale] = float(s_f2)
        scale_loss_proxy[scale] = float(1.0 - s_ap)
        scale_f2_loss_proxy[scale] = float(1.0 - s_f2)

    summary = {
        "dataset": val_dir,
        "model": model_path,
        "split": split,
        "total_images": len(image_files),
        "total_gt_boxes": gt_count,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "f2": round(f2, 6),
        f"AP@IoU{gt_iou_thres}": round(custom_ap, 6),
        "scale_metrics": scale_summary,
        "scale_ap": scale_ap,
        "scale_f1": scale_f1,
        "scale_f2": scale_f2,
        "scale_loss_proxy": scale_loss_proxy,
        "scale_f2_loss_proxy": scale_f2_loss_proxy,
        "images": img_records,
    }

    if save_outputs:
        json_path = os.path.join(output_folder, "results_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"[EVAL] JSON saved -> {json_path}")

        csv_path = os.path.join(output_folder, "results_per_box.csv")
        csv_fields = ["image", "pred_x", "pred_y", "pred_w", "pred_h", "conf", "best_iou", "tp", "scale"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        if verbose:
            print(f"[EVAL] CSV saved -> {csv_path}")

    if verbose:
        print("\n======== Model Evaluation Metrics ========")
        print(f"Dataset       : {val_dir}")
        print(f"Split         : {split}")
        print(f"Model Path    : {model_path}")
        print(f"GT IoU Thres  : {gt_iou_thres}")
        print(f"Total Images  : {len(image_files)}")
        print(f"Total GT Boxes: {gt_count}")
        print(f"TP / FP / FN  : {tp} / {fp} / {fn}")
        print(f"Precision     : {precision:.4f}")
        print(f"Recall        : {recall:.4f}")
        print(f"F1            : {f1:.4f}")
        print(f"F2            : {f2:.4f}")
        print(f"AP@IoU{gt_iou_thres}: {custom_ap:.4f}")
        print("Scale AP      :", {k: round(v, 4) for k, v in scale_ap.items()})
        print("Scale F2      :", {k: round(v, 4) for k, v in scale_f2.items()})
        print("==========================================")

    return summary


if __name__ == "__main__":
    evaluate_model(
        model_path="runs/detect/exp_c_hard_finetune_with_aug_2_-2/weights/best.pt",
        val_dir="hard_exp_c",
        split="newbenchmark_v2.2",
        output_folder="newbenchmark_v2.2_result_aug",
        imgsz=1280,
        conf_thres=0.3,
        pred_iou_thres=0.4,
        gt_iou_thres=0.2,
        do_preprocessing=False,
        draw_annotations=True,
        save_outputs=True,
        verbose=True,
    )
