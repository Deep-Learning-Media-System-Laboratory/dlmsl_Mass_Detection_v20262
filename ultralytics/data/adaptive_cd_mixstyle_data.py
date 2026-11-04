from __future__ import annotations

import csv
import os
from pathlib import Path

import torch

from ultralytics.data.dataset import YOLODataset


class AdaptiveDomainYOLODataset(YOLODataset):
    """YOLODataset with an additional domain ID.

    This dataset is independent from the existing Cross-Domain MixStyle dataset implementation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        manifest_path = os.environ.get("ADMS_DOMAIN_MANIFEST")

        if not manifest_path:
            raise RuntimeError("ADMS_DOMAIN_MANIFEST is not set.")

        manifest_path = Path(manifest_path).expanduser().resolve()

        if not manifest_path.exists():
            raise FileNotFoundError(f"Domain manifest not found: {manifest_path}")

        self.manifest_path = manifest_path
        path_to_domain = {}

        with manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)

            required = {
                "image_path",
                "domain",
            }

            fieldnames = set(reader.fieldnames or [])

            if not required.issubset(fieldnames):
                raise ValueError(
                    f"Manifest must contain columns: 'image_path' and 'domain'. Found: {reader.fieldnames}"
                )

            for row in reader:
                image_path = Path(row["image_path"]).expanduser().resolve()

                domain = row["domain"].strip()

                if not domain:
                    raise ValueError(f"Empty domain for: {image_path}")

                path_to_domain[str(image_path)] = domain

        self.path_to_domain = path_to_domain

        domains = sorted(set(path_to_domain.values()))

        self.domain_to_id = {domain: idx for idx, domain in enumerate(domains)}

        self.id_to_domain = {idx: domain for domain, idx in self.domain_to_id.items()}

        print()
        print("=" * 70)

        print("ADAPTIVE CD MIXSTYLE DOMAIN MAPPING")

        print("=" * 70)

        for domain, domain_id in self.domain_to_id.items():
            print(f"{domain_id:3d} -> {domain}")

        print("=" * 70)

        missing = []

        for image_file in self.im_files:
            resolved = str(Path(image_file).expanduser().resolve())

            if resolved not in self.path_to_domain:
                missing.append(image_file)

        if missing:
            preview = "\n".join(str(x) for x in missing[:10])

            raise RuntimeError(
                "Some training images do not have "
                "domain information in the manifest.\n"
                f"Missing: {len(missing)} images\n\n"
                f"First examples:\n{preview}"
            )

    def get_image_and_label(
        self,
        index,
    ):

        sample = super().get_image_and_label(index)

        image_path = str(Path(self.im_files[index]).expanduser().resolve())

        domain_name = self.path_to_domain[image_path]

        domain_id = self.domain_to_id[domain_name]

        sample["domain_id"] = torch.tensor(
            domain_id,
            dtype=torch.long,
        )

        # sample[
        #     "domain_name"
        # ] = domain_name

        return sample
