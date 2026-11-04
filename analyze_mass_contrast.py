#!/usr/bin/env python3
"""
analyze_mass_contrast.py

Calculate a local lesion-to-background intensity/density proxy for mammography
ground-truth mass boxes stored in COCO JSON.

Optional:
    Use results_per_box.csv to reproduce TP/FN assignment by one-to-one IoU
    matching, allowing direct analysis of whether false negatives are associated
    with low local contrast.

IMPORTANT
---------
These measurements are IMAGE-DERIVED intensity/contrast proxies.
They are NOT radiologist-assigned BI-RADS mass density categories.

Outputs
-------
1. lesion_contrast_per_gt.csv
2. false_negative_lesion_contrast.csv
3. contrast_summary.csv
4. fn_rate_by_conspicuity_quartile.csv
5. fn_rate_by_size_and_conspicuity.csv
6. analysis_metadata.json
7. plots (unless --no-plots)

Expected prediction CSV columns:
    image, pred_x, pred_y, pred_w, pred_h, conf, best_iou, tp

Prediction coordinates are assumed to be pixel XYWH with x,y = top-left,
matching the user's current results_per_box.csv format.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


KNOWN_PREFIXES = (
    "bcsdbt_",
    "hospital_",
    "inbreast_",
    "vindr_",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze local lesion-to-background mammographic contrast."
    )
    p.add_argument(
        "--images",
        required=True,
        help="Directory containing benchmark images. Searched recursively.",
    )
    p.add_argument(
        "--coco",
        required=True,
        help="COCO ground-truth JSON, e.g. newbenchmark_v2.2.json",
    )
    p.add_argument(
        "--pred",
        default=None,
        help="Optional results_per_box.csv for TP/FN assignment.",
    )
    p.add_argument(
        "--out",
        default="mass_contrast_analysis",
        help="Output directory (default: mass_contrast_analysis)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.2,
        help="IoU threshold used for GT/prediction matching (default: 0.2)",
    )
    p.add_argument(
        "--outer-scale",
        type=float,
        default=2.0,
        help="Outer background-ring box scale relative to GT box (default: 2.0)",
    )
    p.add_argument(
        "--inner-scale",
        type=float,
        default=1.15,
        help="Inner excluded box scale; creates a gap around lesion (default: 1.15)",
    )
    p.add_argument(
        "--min-bg-pixels",
        type=int,
        default=100,
        help="Minimum usable background pixels before fallback (default: 100)",
    )
    p.add_argument(
        "--similar-threshold",
        type=float,
        default=0.05,
        help=(
            "Heuristic normalized median difference threshold for labeling "
            "brighter/similar/darker than local background (default: 0.05). "
            "This is NOT a BI-RADS threshold."
        ),
    )
    p.add_argument(
        "--no-breast-mask",
        action="store_true",
        help="Disable automatic breast-tissue mask for lesion/ring sampling.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not create PNG plots.",
    )
    return p.parse_args()


def normalize_name(name):
    """Normalize result/image filenames so prefixed and unprefixed names match."""
    base = os.path.basename(str(name))
    low = base.lower()
    for prefix in KNOWN_PREFIXES:
        if low.startswith(prefix):
            return base[len(prefix):]
    return base


def infer_source(filename):
    """Infer source from the benchmark filename pattern."""
    base = normalize_name(filename)
    stem = Path(base).stem

    if stem.startswith("DBT-"):
        return "BCS-DBT"
    if stem.startswith("DONE_"):
        return "Hospital"
    if base.lower().startswith("rsna_"):
        return "RSNA"
    if re.fullmatch(r"\d+", stem):
        return "INbreast"
    if re.fullmatch(r"[0-9a-fA-F]{32}", stem):
        return "ViNDr"
    return "Unknown"


def build_image_index(image_root):
    """
    Build normalized-name -> paths index.
    This handles benchmark files that have dataset prefixes even if the COCO JSON
    stores the unprefixed original filename.
    """
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    index = defaultdict(list)

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_ext:
            index[normalize_name(p.name)].append(p)

    return index


def resolve_image_path(index, coco_filename):
    key = normalize_name(coco_filename)
    candidates = index.get(key, [])

    if not candidates:
        return None

    # Prefer exact basename match if it exists.
    for p in candidates:
        if p.name == coco_filename:
            return p

    # Otherwise prefixed/unprefixed normalized match.
    return candidates[0]


def read_grayscale(path):
    """Read image preserving dynamic range, then convert to grayscale float32."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img.astype(np.float32)


def robust_normalize(img, low_q=0.5, high_q=99.5):
    """
    Normalize each mammogram to [0,1] using robust image percentiles.
    This improves comparability across PNG bit depths / preprocessing ranges.
    """
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")

    lo = float(np.percentile(finite, low_q))
    hi = float(np.percentile(finite, high_q))

    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32), lo, hi

    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, lo, hi


def largest_component(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if n <= 1:
        return mask_u8.astype(bool)

    # Ignore background component 0.
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest_label


def make_breast_mask(norm_img):
    """
    Approximate breast-tissue mask using Otsu threshold + largest component.

    This mask is only used to reduce contamination from black/non-breast
    background when the local ring reaches outside the breast.
    """
    u8 = np.clip(norm_img * 255.0, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(
        u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Mild morphological cleanup.
    k = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    mask = largest_component(binary > 0)

    # If Otsu produced an implausibly tiny mask, do not use it.
    frac = float(mask.mean())
    if frac < 0.05:
        return np.ones_like(mask, dtype=bool), "fallback_all_pixels"

    return mask, "otsu_largest_component"


def clamp_bbox_xywh(bbox, width, height):
    x, y, w, h = map(float, bbox)

    x1 = max(0, min(width, int(math.floor(x))))
    y1 = max(0, min(height, int(math.floor(y))))
    x2 = max(0, min(width, int(math.ceil(x + w))))
    y2 = max(0, min(height, int(math.ceil(y + h))))

    return x1, y1, x2, y2


def scaled_box_xyxy(bbox, scale, width, height):
    x, y, w, h = map(float, bbox)
    cx = x + w / 2.0
    cy = y + h / 2.0

    sw = w * scale
    sh = h * scale

    x1 = max(0, min(width, int(math.floor(cx - sw / 2.0))))
    y1 = max(0, min(height, int(math.floor(cy - sh / 2.0))))
    x2 = max(0, min(width, int(math.ceil(cx + sw / 2.0))))
    y2 = max(0, min(height, int(math.ceil(cy + sh / 2.0))))

    return x1, y1, x2, y2


def rectangular_mask(shape, xyxy):
    h, w = shape
    x1, y1, x2, y2 = xyxy
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def make_lesion_and_ring_masks(shape, bbox, inner_scale, outer_scale):
    h, w = shape

    lesion_box = clamp_bbox_xywh(bbox, w, h)
    inner_box = scaled_box_xyxy(bbox, inner_scale, w, h)
    outer_box = scaled_box_xyxy(bbox, outer_scale, w, h)

    lesion = rectangular_mask(shape, lesion_box)
    inner = rectangular_mask(shape, inner_box)
    outer = rectangular_mask(shape, outer_box)

    ring = outer & (~inner)
    return lesion, ring


def robust_stats(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "mad": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": med,
        "std": float(np.std(arr)),
        "mad": mad,
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def iou_xywh(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0

    return inter / union


def load_predictions(pred_csv):
    df = pd.read_csv(pred_csv)

    required = {
        "image",
        "pred_x",
        "pred_y",
        "pred_w",
        "pred_h",
        "conf",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing required columns: {sorted(missing)}"
        )

    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[normalize_name(row["image"])].append(row.to_dict())

    return groups


def assign_detection_status(images, anns_by_image, pred_groups, iou_thr):
    """
    Reproduce one-to-one GT matching.

    Predictions are processed from highest confidence to lowest. Each prediction
    can match only one previously unmatched GT, and each GT can match only once.
    """
    assignment = {}

    for image in images:
        image_id = image["id"]
        gts = anns_by_image.get(image_id, [])
        if not gts:
            continue

        preds = pred_groups.get(normalize_name(image["file_name"]), [])
        preds = sorted(preds, key=lambda p: float(p["conf"]), reverse=True)

        unmatched = set(range(len(gts)))

        # Initialize every GT as FN.
        for j, gt in enumerate(gts):
            assignment[gt["id"]] = {
                "detection_status": "FN",
                "matched_pred_conf": np.nan,
                "matched_iou": np.nan,
            }

        for pred in preds:
            if not unmatched:
                break

            pb = (
                float(pred["pred_x"]),
                float(pred["pred_y"]),
                float(pred["pred_w"]),
                float(pred["pred_h"]),
            )

            best_gt_idx = None
            best_iou = 0.0

            for j in unmatched:
                score = iou_xywh(pb, gts[j]["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_gt_idx = j

            if best_gt_idx is not None and best_iou > iou_thr:
                gt = gts[best_gt_idx]
                assignment[gt["id"]] = {
                    "detection_status": "TP",
                    "matched_pred_conf": float(pred["conf"]),
                    "matched_iou": float(best_iou),
                }
                unmatched.remove(best_gt_idx)

    return assignment


def quantile_label(series, low_name="Q1_lowest", high_name="Q4_highest"):
    """
    Stable quartile classification from explicit quantiles.
    Returns labels and thresholds.
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(["Unknown"] * len(s), index=s.index), [np.nan] * 3

    q1, q2, q3 = valid.quantile([0.25, 0.50, 0.75]).tolist()

    labels = []
    for v in s:
        if pd.isna(v):
            labels.append("Unknown")
        elif v <= q1:
            labels.append(low_name)
        elif v <= q2:
            labels.append("Q2")
        elif v <= q3:
            labels.append("Q3")
        else:
            labels.append(high_name)

    return pd.Series(labels, index=s.index), [float(q1), float(q2), float(q3)]


def calculate_metrics(
    norm_img,
    tissue_mask,
    bbox,
    inner_scale,
    outer_scale,
    min_bg_pixels,
    similar_threshold,
):
    lesion_mask, ring_mask = make_lesion_and_ring_masks(
        norm_img.shape, bbox, inner_scale, outer_scale
    )

    # Prefer tissue-only sampling.
    lesion_use = lesion_mask & tissue_mask
    ring_use = ring_mask & tissue_mask

    # If tissue masking removed too much, fall back locally.
    if lesion_use.sum() < 20:
        lesion_use = lesion_mask

    bg_fallback = False
    if ring_use.sum() < min_bg_pixels:
        ring_use = ring_mask
        bg_fallback = True

    lesion_vals = norm_img[lesion_use]
    bg_vals = norm_img[ring_use]

    ls = robust_stats(lesion_vals)
    bs = robust_stats(bg_vals)

    eps = 1e-6

    if ls["n"] == 0 or bs["n"] == 0:
        return {
            "lesion_pixels": ls["n"],
            "background_pixels": bs["n"],
            "background_mask_fallback": bg_fallback,
            "lesion_mean_norm": np.nan,
            "lesion_median_norm": np.nan,
            "background_mean_norm": np.nan,
            "background_median_norm": np.nan,
            "background_std_norm": np.nan,
            "background_mad_norm": np.nan,
            "median_difference": np.nan,
            "density_ratio_median": np.nan,
            "weber_contrast_median": np.nan,
            "abs_weber_contrast_median": np.nan,
            "michelson_contrast_median": np.nan,
            "classic_cnr": np.nan,
            "robust_cnr": np.nan,
            "abs_robust_cnr": np.nan,
            "lesion_median_percentile_in_background": np.nan,
            "density_direction_heuristic": "Unknown",
        }

    median_diff = ls["median"] - bs["median"]

    density_ratio = ls["median"] / (bs["median"] + eps)
    weber = median_diff / (abs(bs["median"]) + eps)
    michelson = median_diff / (ls["median"] + bs["median"] + eps)

    classic_cnr = (ls["mean"] - bs["mean"]) / (bs["std"] + eps)
    robust_sigma_bg = 1.4826 * bs["mad"]
    robust_cnr = median_diff / (robust_sigma_bg + eps)

    percentile = float(np.mean(bg_vals <= ls["median"]))

    # Heuristic only. Uses normalized intensity difference, not BI-RADS criteria.
    if median_diff > similar_threshold:
        direction = "brighter_than_local_background"
    elif median_diff < -similar_threshold:
        direction = "darker_than_local_background"
    else:
        direction = "similar_to_local_background"

    return {
        "lesion_pixels": ls["n"],
        "background_pixels": bs["n"],
        "background_mask_fallback": bg_fallback,
        "lesion_mean_norm": ls["mean"],
        "lesion_median_norm": ls["median"],
        "background_mean_norm": bs["mean"],
        "background_median_norm": bs["median"],
        "background_std_norm": bs["std"],
        "background_mad_norm": bs["mad"],
        "median_difference": median_diff,
        "density_ratio_median": density_ratio,
        "weber_contrast_median": weber,
        "abs_weber_contrast_median": abs(weber),
        "michelson_contrast_median": michelson,
        "classic_cnr": classic_cnr,
        "robust_cnr": robust_cnr,
        "abs_robust_cnr": abs(robust_cnr),
        "lesion_median_percentile_in_background": percentile,
        "density_direction_heuristic": direction,
    }


def create_summary(df):
    rows = []

    for status, g in df.groupby("detection_status", dropna=False):
        rows.append(
            {
                "group_type": "detection_status",
                "group": status,
                "n_gt": len(g),
                "median_bbox_area_px2": g["bbox_area_px2"].median(),
                "median_bbox_area_fraction": g["bbox_area_fraction"].median(),
                "median_density_ratio": g["density_ratio_median"].median(),
                "median_weber_contrast": g["weber_contrast_median"].median(),
                "median_abs_weber_contrast": g[
                    "abs_weber_contrast_median"
                ].median(),
                "median_robust_cnr": g["robust_cnr"].median(),
                "median_abs_robust_cnr": g["abs_robust_cnr"].median(),
            }
        )

    for source, g in df.groupby("source", dropna=False):
        n = len(g)
        fn = int((g["detection_status"] == "FN").sum())
        rows.append(
            {
                "group_type": "source",
                "group": source,
                "n_gt": n,
                "fn": fn,
                "fn_rate": fn / n if n else np.nan,
                "median_bbox_area_px2": g["bbox_area_px2"].median(),
                "median_bbox_area_fraction": g["bbox_area_fraction"].median(),
                "median_density_ratio": g["density_ratio_median"].median(),
                "median_weber_contrast": g["weber_contrast_median"].median(),
                "median_abs_weber_contrast": g[
                    "abs_weber_contrast_median"
                ].median(),
                "median_robust_cnr": g["robust_cnr"].median(),
                "median_abs_robust_cnr": g["abs_robust_cnr"].median(),
            }
        )

    return pd.DataFrame(rows)


def fn_rate_table(df, group_col):
    rows = []
    for group, g in df.groupby(group_col, dropna=False):
        n = len(g)
        fn = int((g["detection_status"] == "FN").sum())
        tp = int((g["detection_status"] == "TP").sum())
        rows.append(
            {
                group_col: group,
                "total_gt": n,
                "tp": tp,
                "fn": fn,
                "fn_rate": fn / n if n else np.nan,
                "median_bbox_area_px2": g["bbox_area_px2"].median(),
                "median_abs_robust_cnr": g["abs_robust_cnr"].median(),
                "median_abs_weber_contrast": g[
                    "abs_weber_contrast_median"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def size_contrast_table(df):
    rows = []
    for (size_q, contrast_q), g in df.groupby(
        ["size_quartile", "conspicuity_quartile"], dropna=False
    ):
        n = len(g)
        fn = int((g["detection_status"] == "FN").sum())
        rows.append(
            {
                "size_quartile": size_q,
                "conspicuity_quartile": contrast_q,
                "total_gt": n,
                "fn": fn,
                "fn_rate": fn / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_plots(df, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Matplotlib unavailable; plots skipped: {e}")
        return

    plot_df = df[np.isfinite(df["abs_robust_cnr"])].copy()
    if plot_df.empty:
        print("[WARN] No finite contrast values; plots skipped.")
        return

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for status in ["TP", "FN"]:
        vals = plot_df.loc[
            plot_df["detection_status"] == status, "abs_robust_cnr"
        ].values
        if len(vals):
            ax.hist(vals, bins=25, alpha=0.5, label=status)
    ax.set_xlabel("Absolute robust CNR")
    ax.set_ylabel("Ground-truth masses")
    ax.set_title("Local lesion conspicuity: TP vs FN")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tp_fn_abs_robust_cnr_hist.png", dpi=180)
    plt.close(fig)

    # Scatter: size vs contrast
    fig, ax = plt.subplots(figsize=(8, 5))
    for status in ["TP", "FN"]:
        g = plot_df[plot_df["detection_status"] == status]
        if len(g):
            ax.scatter(
                g["bbox_area_px2"],
                g["abs_robust_cnr"],
                s=22,
                alpha=0.65,
                label=status,
            )
    ax.set_xscale("log")
    ax.set_xlabel("GT bbox area (px², log scale)")
    ax.set_ylabel("Absolute robust CNR")
    ax.set_title("Mass size vs local lesion conspicuity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mass_size_vs_conspicuity.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    if args.outer_scale <= args.inner_scale:
        raise ValueError("--outer-scale must be larger than --inner-scale")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COCO annotations...")
    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    image_by_id = {im["id"]: im for im in images}
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    print(f"Images in COCO: {len(images)}")
    print(f"GT masses in COCO: {len(annotations)}")

    detection_assignment = {}
    if args.pred:
        print("Loading predictions and assigning TP/FN...")
        pred_groups = load_predictions(args.pred)
        detection_assignment = assign_detection_status(
            images,
            anns_by_image,
            pred_groups,
            args.iou,
        )

        tp_count = sum(
            x["detection_status"] == "TP"
            for x in detection_assignment.values()
        )
        fn_count = sum(
            x["detection_status"] == "FN"
            for x in detection_assignment.values()
        )
        print(f"Reconstructed GT status @ IoU>{args.iou}: TP={tp_count}, FN={fn_count}")

    print("Indexing benchmark images...")
    image_index = build_image_index(args.images)
    print(f"Indexed normalized image names: {len(image_index)}")

    rows = []
    missing_images = []
    processed_images = 0

    for image in images:
        gt_list = anns_by_image.get(image["id"], [])
        if not gt_list:
            continue

        image_path = resolve_image_path(image_index, image["file_name"])
        if image_path is None:
            missing_images.append(image["file_name"])
            continue

        gray = read_grayscale(image_path)
        norm, norm_lo, norm_hi = robust_normalize(gray)

        if args.no_breast_mask:
            tissue_mask = np.ones_like(norm, dtype=bool)
            mask_method = "disabled"
        else:
            tissue_mask, mask_method = make_breast_mask(norm)

        actual_h, actual_w = norm.shape
        coco_w = int(image.get("width", actual_w))
        coco_h = int(image.get("height", actual_h))

        # The COCO boxes should correspond to the image pixel dimensions.
        # If not, rescale the annotation box into the actual loaded image size.
        sx = actual_w / coco_w if coco_w else 1.0
        sy = actual_h / coco_h if coco_h else 1.0

        for ann in gt_list:
            x, y, w, h = map(float, ann["bbox"])
            scaled_bbox = [x * sx, y * sy, w * sx, h * sy]

            metrics = calculate_metrics(
                norm,
                tissue_mask,
                scaled_bbox,
                args.inner_scale,
                args.outer_scale,
                args.min_bg_pixels,
                args.similar_threshold,
            )

            det = detection_assignment.get(
                ann["id"],
                {
                    "detection_status": "Unknown",
                    "matched_pred_conf": np.nan,
                    "matched_iou": np.nan,
                },
            )

            bbox_area = float(w * h)
            image_area = float(coco_w * coco_h)

            row = {
                "annotation_id": ann["id"],
                "image_id": image["id"],
                "image": image["file_name"],
                "normalized_image_key": normalize_name(image["file_name"]),
                "source": infer_source(image["file_name"]),
                "image_path": str(image_path),
                "image_width": coco_w,
                "image_height": coco_h,
                "gt_x": x,
                "gt_y": y,
                "gt_w": w,
                "gt_h": h,
                "bbox_area_px2": bbox_area,
                "bbox_area_fraction": bbox_area / image_area if image_area else np.nan,
                "detection_status": det["detection_status"],
                "matched_pred_conf": det["matched_pred_conf"],
                "matched_iou": det["matched_iou"],
                "normalization_p0_5": norm_lo,
                "normalization_p99_5": norm_hi,
                "breast_mask_method": mask_method,
            }
            row.update(metrics)
            rows.append(row)

        processed_images += 1
        if processed_images % 100 == 0:
            print(f"Processed {processed_images} positive images...")

    if not rows:
        raise RuntimeError(
            "No annotated lesions were processed. Check --images and filename matching."
        )

    df = pd.DataFrame(rows)

    # Quartiles based on ALL valid GT masses.
    df["size_quartile"], size_thresholds = quantile_label(
        df["bbox_area_px2"],
        low_name="Q1_smallest",
        high_name="Q4_largest",
    )
    df["conspicuity_quartile"], contrast_thresholds = quantile_label(
        df["abs_robust_cnr"],
        low_name="Q1_lowest_contrast",
        high_name="Q4_highest_contrast",
    )

    per_gt_path = out_dir / "lesion_contrast_per_gt.csv"
    df.to_csv(per_gt_path, index=False)

    fn_df = df[df["detection_status"] == "FN"].copy()
    fn_path = out_dir / "false_negative_lesion_contrast.csv"
    fn_df.to_csv(fn_path, index=False)

    summary = create_summary(df)
    summary_path = out_dir / "contrast_summary.csv"
    summary.to_csv(summary_path, index=False)

    if args.pred:
        qtable = fn_rate_table(df, "conspicuity_quartile")
        qtable_path = out_dir / "fn_rate_by_conspicuity_quartile.csv"
        qtable.to_csv(qtable_path, index=False)

        size_contrast = size_contrast_table(df)
        size_contrast_path = out_dir / "fn_rate_by_size_and_conspicuity.csv"
        size_contrast.to_csv(size_contrast_path, index=False)

    metadata = {
        "note": (
            "Image-derived local lesion/background intensity metrics. "
            "These are NOT BI-RADS mass density labels."
        ),
        "coco": str(args.coco),
        "images": str(args.images),
        "pred": str(args.pred) if args.pred else None,
        "iou_threshold": args.iou if args.pred else None,
        "background_ring_inner_scale": args.inner_scale,
        "background_ring_outer_scale": args.outer_scale,
        "robust_normalization_percentiles": [0.5, 99.5],
        "similar_intensity_heuristic_threshold": args.similar_threshold,
        "n_processed_gt": int(len(df)),
        "n_missing_positive_images": int(len(missing_images)),
        "size_quartile_thresholds_px2": size_thresholds,
        "abs_robust_cnr_quartile_thresholds": contrast_thresholds,
        "tp": int((df["detection_status"] == "TP").sum()),
        "fn": int((df["detection_status"] == "FN").sum()),
    }

    with open(out_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if missing_images:
        missing_path = out_dir / "missing_positive_images.txt"
        missing_path.write_text("\n".join(missing_images), encoding="utf-8")
        print(
            f"[WARN] {len(missing_images)} positive images were not found. "
            f"See {missing_path}"
        )

    if not args.no_plots:
        make_plots(df, out_dir)

    print("\n" + "=" * 72)
    print("MASS CONTRAST ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Processed GT masses : {len(df)}")
    print(f"TP                  : {(df['detection_status'] == 'TP').sum()}")
    print(f"FN                  : {(df['detection_status'] == 'FN').sum()}")
    print(f"Output directory    : {out_dir.resolve()}")
    print()
    print("Primary metric:")
    print(
        "  abs_robust_cnr = |median(lesion) - median(background)| / "
        "(1.4826 * MAD(background))"
    )
    print()
    print("Interpretation:")
    print("  smaller abs_robust_cnr -> lesion is less locally conspicuous")
    print("  larger  abs_robust_cnr -> lesion is more locally conspicuous")
    print()
    print("Remember: these are image-derived contrast proxies, NOT BI-RADS density.")
    print("=" * 72)


if __name__ == "__main__":
    main()