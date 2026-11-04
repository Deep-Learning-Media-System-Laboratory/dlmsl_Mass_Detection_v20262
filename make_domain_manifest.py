from __future__ import annotations

import csv
import re
from pathlib import Path

import yaml

# ============================================================
# CONFIG
# ============================================================

DATA_YAML = "all-simulate75.yaml"

OUTPUT_CSV = "allsim75_domain_manifest.csv"
UNKNOWN_CSV = "allsim75_domain_manifest_unknown.csv"


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# DOMAIN RULES
# ============================================================

INBREAST_NUMERIC_RE = re.compile(r"^\d{8}$")

SYNTHETIC_BBOX_RE = re.compile(
    r"^.+_[LR]_(CC|MLO)_bbox-\d+_BI-RADS-\d+_.*Mass.*$",
    re.IGNORECASE,
)


def detect_domain(image_path: str | Path) -> str | None:

    path = Path(image_path)

    name = path.name.lower()
    stem = path.stem.lower()

    # --------------------------------------------------------
    # ViNDr
    # --------------------------------------------------------

    if name.startswith(("vindr_", "vindr-n_")):
        return "vindr"

    # --------------------------------------------------------
    # CMMD
    # --------------------------------------------------------

    if name.startswith(("cmmd_", "cmmd-n_")):
        return "cmmd"

    # Optional legacy CMMD names
    if name.startswith(("d1-", "d2-")):
        return "cmmd"

    # --------------------------------------------------------
    # CBIS-DDSM
    # --------------------------------------------------------

    if name.startswith("cbis_"):
        return "cbis"

    if name.startswith(
        (
            "mass-training_",
            "mass-testing_",
        )
    ):
        return "cbis"

    # --------------------------------------------------------
    # INbreast
    # --------------------------------------------------------

    if name.startswith("inbreast_"):
        return "inbreast"

    # Legacy plain 8-digit INbreast filename
    if INBREAST_NUMERIC_RE.fullmatch(stem):
        return "inbreast"

    # --------------------------------------------------------
    # EMBED
    # --------------------------------------------------------

    if name.startswith(("embed_", "embed-n_")):
        return "embed"

    # --------------------------------------------------------
    # RSNA
    # --------------------------------------------------------

    if name.startswith(("rsna_", "rsna-n_")):
        return "rsna"

    # --------------------------------------------------------
    # BCS-DBT
    # --------------------------------------------------------

    if name.startswith(("bcs_", "bcs-dbt_", "bcsdbt-", "bcsdbt_")):
        return "bcs-dbt"

    # --------------------------------------------------------
    # Hospital
    # --------------------------------------------------------

    if name.startswith(
        (
            "hospital_",
            # "hospital-",
            "done_",
        )
    ):
        return "hospital"

    if name.startswith(
        (
            "b20260814_",
            "b202511_",
            "b20260202_",
            "b20260402_",
            "b20260814-1_",
            "b20260814-2_",
            "b20260204_",
            "b20260814-",
        )
    ):
        return "hospital-new"

    # --------------------------------------------------------
    # Hospital PUC
    #
    # EDIT THESE PREFIXES IF YOUR REAL FILES USE
    # ANOTHER NAMING RULE.
    # --------------------------------------------------------

    if name.startswith(
        (
            "hospital-old_",
            "hospital_phuc_",
            "puc_",
        )
    ):
        return "hospital-old"

    if name.startswith(("bcc_",)):
        return "bcc"

    if name.startswith(("img_",)):
        return "img"

    # --------------------------------------------------------
    # Synthetic
    # --------------------------------------------------------
    if name.startswith(("synthetic_",)):
        return "synthetic"

    # if SYNTHETIC_BBOX_RE.fullmatch(
    #     path.stem
    # ):
    #     return "synthetic"

    # Add any additional known synthetic naming rule here.

    return None


# ============================================================
# RESOLVE TRAIN IMAGES FROM YOLO YAML
# ============================================================


def collect_images_from_path(
    item: str,
    dataset_root: Path,
) -> list[Path]:

    p = Path(item)

    if not p.is_absolute():
        p = dataset_root / p

    p = p.resolve()

    # Directory
    if p.is_dir():
        return sorted(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTENSIONS)

    # Text file containing image paths
    if p.is_file() and p.suffix.lower() == ".txt":
        images = []

        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line:
                continue

            q = Path(line)

            if not q.is_absolute():
                q = p.parent / q

            images.append(q.resolve())

        return images

    raise FileNotFoundError(f"Cannot resolve training source: {p}")


def get_training_images(
    data_yaml: str | Path,
) -> list[Path]:

    yaml_path = Path(data_yaml).resolve()

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # YOLO YAML:
    #
    # path: /dataset/root
    # train: images/train
    #
    # or train can be a list.

    root = config.get(
        "path",
        yaml_path.parent,
    )

    root = Path(root)

    if not root.is_absolute():
        root = yaml_path.parent / root

    root = root.resolve()

    train = config["train"]

    if isinstance(train, str):
        train = [train]

    images = []

    for item in train:
        images.extend(
            collect_images_from_path(
                item,
                root,
            )
        )

    # Remove duplicates
    return sorted(set(images))


# ============================================================
# MAIN
# ============================================================


def main():

    images = get_training_images(DATA_YAML)

    print(f"Training images found: {len(images)}")

    rows = []
    unknown = []

    counts = {}

    for image_path in images:
        domain = detect_domain(image_path)

        if domain is None:
            unknown.append(
                {
                    "image_path": str(image_path),
                    "filename": (image_path.name),
                }
            )

            continue

        counts[domain] = counts.get(domain, 0) + 1

        rows.append(
            {
                "image_path": str(image_path),
                "filename": (image_path.name),
                "domain": domain,
            }
        )

    # --------------------------------------------------------
    # Write valid manifest
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "filename",
                "domain",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Write unknown files
    # --------------------------------------------------------

    with open(
        UNKNOWN_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "filename",
            ],
        )

        writer.writeheader()
        writer.writerows(unknown)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOMAIN MANIFEST SUMMARY")
    print("=" * 70)

    for domain in sorted(counts):
        print(f"{domain:<20} {counts[domain]:>8}")

    print("-" * 70)

    print(f"{'CLASSIFIED':<20} {len(rows):>8}")

    print(f"{'UNKNOWN':<20} {len(unknown):>8}")

    print(f"{'TOTAL':<20} {len(images):>8}")

    print("=" * 70)

    print(f"\nManifest: {OUTPUT_CSV}")

    print(f"Unknown : {UNKNOWN_CSV}")

    if unknown:
        print()
        print("WARNING: Some images were not classified.")

        print("Do NOT start Cross-Domain MixStyle training until UNKNOWN = 0.")


if __name__ == "__main__":
    main()
