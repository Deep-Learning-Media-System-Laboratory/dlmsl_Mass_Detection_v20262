#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Reuse the exact augmentation code used by your SA-AutoAug search.
import sa_autoaug_search_integrated as sai
import yaml

SEED = 42
BASE_DATA_YAML = "with-hospital.yaml"  # ORIGINAL non-oversampled YAML
ORIGINAL_DOMAIN_MANIFEST = "domain_manifest.csv"
BEST_POLICY_PATH = "../../sa_autoaug_yolo_search_f2_refined/best_policy.json"
OUTPUT_ROOT = "cdms_augmented_oversample_2x"

# Dataset/domain names used by your project.
# These are informative for selection; domain assignment still comes from
# ORIGINAL_DOMAIN_MANIFEST so we do not guess domains from filenames.
# DOMAIN_PREFIXES = {
#     "hospital-old_": "hospital-old",
#     "bcc_": "bcc",
#     "img_": "img",
#     "bcsdbt_": "bcsdbt",
#     "hospital": "hospital",
# }

# Choose which domains receive augmented oversampling.
# Examples:
#   SELECTED_DOMAINS = ["bcsdbt"]
#   SELECTED_DOMAINS = ["bcsdbt", "hospital"]
SELECTED_DOMAINS = ["hospital-old", "bcc", "img", "bcs-dbt", "hospital"]
#   SELECTED_DOMAINS = None   # augment all domains
#
# Domains NOT selected remain in the training set unchanged.
# SELECTED_DOMAINS = ["bcsdbt", "hospital"]

# Domain target = min(largest positive domain, ceil(current * MAX_FACTOR)).
MAX_FACTOR = 4.0
POSITIVE_ONLY = True
MAX_AUG_ATTEMPTS_PER_REQUEST = 8
RESET_OUTPUT = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def norm_path(path):
    return str(Path(path).expanduser().resolve())


def short_hash(text, n=10):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def safe_domain_name(domain):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(domain).strip())
    return s or "domain"


def load_best_policy(path):
    with open(path) as f:
        obj = json.load(f)
    if "best_policy" in obj:
        return obj["best_policy"]
    if "policy" in obj:
        return obj["policy"]
    return obj


def label_path_from_image(image_path):
    p = Path(image_path).expanduser().resolve()
    parts = list(p.parts)
    idxs = [i for i, part in enumerate(parts) if part == "images"]
    if not idxs:
        raise RuntimeError(f"Cannot infer label path; '/images/' not found in:\n{p}")
    parts[idxs[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def list_images_in_dir(directory):
    directory = Path(directory)
    return sorted(str(p.resolve()) for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def resolve_candidate(raw, bases):
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    for base in bases:
        q = (Path(base) / p).resolve()
        if q.exists():
            return q
    return (Path(bases[0]) / p).resolve()


def expand_train_entry(entry, dataset_root, yaml_dir):
    p = resolve_candidate(entry, [dataset_root, yaml_dir])

    if p.is_dir():
        return list_images_in_dir(p)

    if p.is_file() and p.suffix.lower() == ".txt":
        out = []
        with open(p) as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                q = resolve_candidate(raw, [p.parent, dataset_root, yaml_dir])
                if not q.exists():
                    raise FileNotFoundError(f"Train image not found:\n{raw}\nResolved:\n{q}")
                if q.suffix.lower() in IMAGE_EXTS:
                    out.append(str(q.resolve()))
        return out

    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [str(p.resolve())]

    raise FileNotFoundError(f"Could not resolve train entry: {entry}\nResolved: {p}")


def load_train_images_from_yaml(data_yaml):
    data_yaml = Path(data_yaml).expanduser().resolve()
    yaml_dir = data_yaml.parent

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    root = Path(str(cfg.get("path", "."))).expanduser()
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()
    else:
        root = root.resolve()

    train = cfg.get("train")
    if train is None:
        raise RuntimeError(f"No train entry in {data_yaml}")

    entries = train if isinstance(train, list) else [train]
    images = []
    for e in entries:
        images.extend(expand_train_entry(e, root, yaml_dir))

    dup = len(images) - len(set(images))
    if dup:
        raise RuntimeError(
            f"BASE_DATA_YAML already has {dup} duplicate entries. Use the ORIGINAL non-oversampled all.yaml."
        )

    return images, cfg


def load_domain_manifest(path):
    exact = {}
    basename_domains = defaultdict(set)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "domain"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Manifest must contain image_path,domain. Found: {reader.fieldnames}")

        for row in reader:
            raw = row["image_path"].strip()
            domain = row["domain"].strip()
            if not raw:
                continue
            n = norm_path(raw)
            if n in exact and exact[n] != domain:
                raise RuntimeError(f"Conflicting domain for {n}")
            exact[n] = domain
            basename_domains[Path(raw).name].add(domain)

    return exact, basename_domains


def match_domains(train_images, exact, basename_domains):
    rows, unmatched, ambiguous = [], [], []

    for p in train_images:
        n = norm_path(p)
        if n in exact:
            d = exact[n]
        else:
            ds = basename_domains.get(Path(p).name, set())
            if len(ds) == 1:
                d = next(iter(ds))
            elif len(ds) == 0:
                unmatched.append(p)
                continue
            else:
                ambiguous.append((p, sorted(ds)))
                continue
        rows.append((n, d))

    if unmatched or ambiguous:
        print(f"Unmatched={len(unmatched)}, ambiguous={len(ambiguous)}")
        for p in unmatched[:10]:
            print("UNMATCHED:", p)
        for p, ds in ambiguous[:10]:
            print("AMBIGUOUS:", p, ds)
        raise RuntimeError("Not every training image could be mapped to one CDMS domain.")

    return rows


def positive_sources(rows):
    all_by_domain = defaultdict(list)
    pos_by_domain = defaultdict(list)
    total_boxes = 0

    for image_path, domain in rows:
        all_by_domain[domain].append(image_path)
        bboxes = sai.read_yolo_labels(str(label_path_from_image(image_path)))
        if bboxes:
            pos_by_domain[domain].append(image_path)
            total_boxes += len(bboxes)

    return all_by_domain, pos_by_domain, total_boxes


def compute_targets(pos_by_domain):
    counts = {d: len(v) for d, v in pos_by_domain.items() if v}
    if not counts:
        raise RuntimeError("No lesion-positive images found.")

    # Use the global largest positive domain as the cap/reference, but only
    # selected domains are actually increased. Unselected domains keep
    # target == current count.
    largest = max(counts.values())

    if SELECTED_DOMAINS is None:
        selected = set(counts)
    else:
        selected = set(SELECTED_DOMAINS)
        unknown = selected - set(counts)
        if unknown:
            raise RuntimeError(
                "SELECTED_DOMAINS contains domain(s) not present among "
                f"positive training images: {sorted(unknown)}\n"
                f"Available positive domains: {sorted(counts)}"
            )

    targets = {}
    for d, n in counts.items():
        if d in selected:
            targets[d] = min(largest, math.ceil(n * MAX_FACTOR))
        else:
            targets[d] = n

    return counts, targets, largest, selected


def make_augmented_copy(
    source_image_path,
    domain,
    output_img_dir,
    output_lbl_dir,
    aug_index,
    policy,
):
    source_image_path = norm_path(source_image_path)
    img = sai.read_image(source_image_path)
    if img is None:
        return None

    bboxes = sai.read_yolo_labels(str(label_path_from_image(source_image_path)))
    if POSITIVE_ONLY and not bboxes:
        return None

    for _ in range(MAX_AUG_ATTEMPTS_PER_REQUEST):
        aug_img, aug_bboxes = sai.apply_policy_to_image(img.copy(), bboxes.copy(), policy)
        aug_bboxes = [b for b in aug_bboxes if sai.valid_yolo_box(b)]

        if bboxes and not aug_bboxes:
            continue
        if np.array_equal(aug_img, img):
            continue

        stem = Path(source_image_path).stem
        aug_name = f"aug_{safe_domain_name(domain)}_{aug_index:07d}_{stem}_{short_hash(source_image_path)}.png"

        out_img = output_img_dir / aug_name
        out_lbl = output_lbl_dir / f"{Path(aug_name).stem}.txt"

        sai.write_image(str(out_img), aug_img)
        sai.write_yolo_labels(str(out_lbl), aug_bboxes)
        return str(out_img.resolve())

    return None


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "domain"])
        w.writeheader()
        for image_path, domain in rows:
            w.writerow({"image_path": norm_path(image_path), "domain": domain})


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    base_yaml = Path(BASE_DATA_YAML).resolve()
    original_manifest = Path(ORIGINAL_DOMAIN_MANIFEST).resolve()
    policy_path = Path(BEST_POLICY_PATH).resolve()
    out_root = Path(OUTPUT_ROOT).resolve()

    for p in (base_yaml, original_manifest, policy_path):
        if not p.exists():
            raise FileNotFoundError(p)

    if out_root.exists():
        if RESET_OUTPUT:
            shutil.rmtree(out_root)
        else:
            raise RuntimeError(f"{out_root} already exists. Set RESET_OUTPUT=True to regenerate.")

    out_img_dir = out_root / "images" / "train_aug"
    out_lbl_dir = out_root / "labels" / "train_aug"
    out_img_dir.mkdir(parents=True)
    out_lbl_dir.mkdir(parents=True)

    train_images, base_cfg = load_train_images_from_yaml(base_yaml)
    exact, basename_domains = load_domain_manifest(original_manifest)
    rows = match_domains(train_images, exact, basename_domains)

    all_by_domain, pos_by_domain, total_boxes = positive_sources(rows)
    pos_counts, targets, largest, selected_domains = compute_targets(pos_by_domain)

    print("\n========== ORIGINAL DISTRIBUTION ==========")
    for d in sorted(all_by_domain):
        print(f"{d:24s} all={len(all_by_domain[d]):6d} positive={len(pos_by_domain[d]):6d}")
    print("Total positive boxes:", total_boxes)

    print("\n========== AUGMENTED OVERSAMPLING PLAN ==========")
    print("Largest positive domain:", largest)
    print("MAX_FACTOR:", MAX_FACTOR)
    print("SELECTED_DOMAINS:", sorted(selected_domains))

    planned = 0
    for d in sorted(pos_counts):
        n = pos_counts[d]
        need = targets[d] - n
        planned += need
        status = "SELECTED" if d in selected_domains else "unchanged"
        print(f"{d:24s} positive={n:6d} target={targets[d]:6d} new_aug={need:6d} [{status}]")
    print("Planned augmented images:", planned)

    policy = load_best_policy(policy_path)

    final_train_paths = [p for p, _ in rows]
    final_manifest_rows = list(rows)
    generated = Counter()
    failures = Counter()
    global_idx = 0

    for domain in sorted(pos_counts):
        sources = list(pos_by_domain[domain])
        need = targets[domain] - pos_counts[domain]
        if need <= 0:
            continue

        rng = random.Random(SEED + int(short_hash(domain, 6), 16))
        rng.shuffle(sources)

        source_idx = 0
        attempts = 0
        max_total_attempts = max(need * MAX_AUG_ATTEMPTS_PER_REQUEST * 3, 100)

        print(f"\n[{domain}] need {need}")

        while generated[domain] < need:
            if attempts >= max_total_attempts:
                raise RuntimeError(f"Could not generate enough valid samples for {domain}: {generated[domain]}/{need}")

            source = sources[source_idx % len(sources)]
            source_idx += 1
            attempts += 1
            global_idx += 1

            new_path = make_augmented_copy(source, domain, out_img_dir, out_lbl_dir, global_idx, policy)

            if new_path is None:
                failures[domain] += 1
                continue

            final_train_paths.append(new_path)
            final_manifest_rows.append((new_path, domain))
            generated[domain] += 1

            if generated[domain] % 100 == 0:
                print(f"  {generated[domain]}/{need}")

    train_txt = out_root / "train_augmented_oversample.txt"
    new_manifest = out_root / "domain_manifest_augmented_oversample.csv"
    new_yaml = out_root / "all_augmented_oversample.yaml"
    summary_path = out_root / "generation_summary.json"

    with open(train_txt, "w") as f:
        for p in final_train_paths:
            f.write(norm_path(p) + "\n")

    write_manifest(new_manifest, final_manifest_rows)

    out_cfg = dict(base_cfg)
    out_cfg["train"] = str(train_txt.resolve())
    with open(new_yaml, "w") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, allow_unicode=True)

    final_positive = Counter()
    for image_path, domain in final_manifest_rows:
        lp = label_path_from_image(image_path)
        if lp.exists() and sai.read_yolo_labels(str(lp)):
            final_positive[domain] += 1

    summary = {
        "seed": SEED,
        "max_factor": MAX_FACTOR,
        "selected_domains": sorted(selected_domains),
        "positive_only": POSITIVE_ONLY,
        "base_data_yaml": str(base_yaml),
        "original_domain_manifest": str(original_manifest),
        "best_policy_path": str(policy_path),
        "original_train_images": len(rows),
        "generated_augmented_images": sum(generated.values()),
        "final_train_entries": len(final_train_paths),
        "original_all_domain_counts": {d: len(all_by_domain[d]) for d in sorted(all_by_domain)},
        "original_positive_domain_counts": {d: len(pos_by_domain[d]) for d in sorted(pos_by_domain)},
        "target_positive_domain_counts": targets,
        "generated_by_domain": dict(generated),
        "generation_failures_by_domain": dict(failures),
        "final_positive_domain_counts": dict(final_positive),
        "train_txt": str(train_txt.resolve()),
        "new_domain_manifest": str(new_manifest.resolve()),
        "new_data_yaml": str(new_yaml.resolve()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== COMPLETE ==========")
    print("Original train :", len(rows))
    print("New augmented :", sum(generated.values()))
    print("Final entries  :", len(final_train_paths))
    print("\nFinal positive counts:")
    for d in sorted(final_positive):
        print(f"{d:24s}: {final_positive[d]:6d}")

    print("\nTrain list :", train_txt)
    print("Manifest   :", new_manifest)
    print("Data YAML  :", new_yaml)
    print("Summary    :", summary_path)
    print("\nCDMS manifest:")
    print(f'export CDMS_DOMAIN_MANIFEST="{new_manifest.resolve()}"')


if __name__ == "__main__":
    main()
