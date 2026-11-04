import csv
import os
from pathlib import Path

import torch

from ultralytics.data.dataset import YOLODataset


class DomainYOLODataset(YOLODataset):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        manifest_path = os.environ.get(
            "CDMS_DOMAIN_MANIFEST"
        )

        if not manifest_path:
            raise RuntimeError(
                "CDMS_DOMAIN_MANIFEST is not set."
            )

        manifest_path = Path(
            manifest_path
        ).resolve()

        if not manifest_path.exists():
            raise FileNotFoundError(
                manifest_path
            )

        # ====================================================
        # READ MANIFEST
        # ====================================================

        path_to_domain = {}

        with manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(
                f
            )

            required = {
                "image_path",
                "domain",
            }

            if not required.issubset(
                reader.fieldnames or []
            ):
                raise RuntimeError(
                    "Manifest requires:\n"
                    "image_path, domain"
                )

            for row in reader:

                image_path = str(
                    Path(
                        row["image_path"]
                    ).resolve()
                )

                domain = (
                    row["domain"]
                    .strip()
                    .lower()
                )

                if image_path in path_to_domain:

                    old = (
                        path_to_domain[
                            image_path
                        ]
                    )

                    if old != domain:

                        raise RuntimeError(
                            "Conflicting domain:\n"
                            f"{image_path}\n"
                            f"{old} vs {domain}"
                        )

                path_to_domain[
                    image_path
                ] = domain

        # ====================================================
        # DOMAIN MAPPING
        # ====================================================

        domains = sorted(
            set(
                path_to_domain.values()
            )
        )

        # if len(domains) < 2:
        #     raise RuntimeError(
        #         "Need at least two domains."
        #     )

        # ====================================================
        # SINGLE-DOMAIN HANDLING
        # ====================================================
        
        allow_single_domain = (
            os.environ
            .get(
                "CDMS_ALLOW_SINGLE_DOMAIN",
                "0",
            )
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "y",
            }
        )
        
        if len(domains) < 2:
        
            if not allow_single_domain:
                raise RuntimeError(
                    "Need at least two domains.\n"
                    "For single-domain fine-tuning, set:\n"
                    "CDMS_ALLOW_SINGLE_DOMAIN=1"
                )
        
            print()
            print("=" * 70)
            print(
                "WARNING: SINGLE-DOMAIN "
                "CDMS FINE-TUNING"
            )
            print(
                f"Domain: {domains[0]}"
            )
            print(
                "Cross-Domain MixStyle will "
                "not be applied because no "
                "different donor domain exists."
            )
            print("=" * 70)

        self.domain_to_id = {
            name: i
            for i, name
            in enumerate(domains)
        }

        self.id_to_domain = {
            i: name
            for name, i
            in self.domain_to_id.items()
        }

        # ====================================================
        # MAP EVERY TRAINING IMAGE
        # ====================================================

        self.domain_ids = []

        missing = []

        for im_file in self.im_files:

            image_path = str(
                Path(
                    im_file
                ).resolve()
            )

            if (
                image_path
                not in path_to_domain
            ):

                missing.append(
                    image_path
                )

                continue

            domain_name = (
                path_to_domain[
                    image_path
                ]
            )

            self.domain_ids.append(
                self.domain_to_id[
                    domain_name
                ]
            )

        if missing:

            raise RuntimeError(
                f"{len(missing)} YOLO images "
                "missing from manifest.\n"
                + "\n".join(
                    missing[:20]
                )
            )

        if len(
            self.domain_ids
        ) != len(
            self.im_files
        ):

            raise RuntimeError(
                "domain_ids/image count mismatch."
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        counts = {}

        for d in self.domain_ids:

            name = self.id_to_domain[
                d
            ]

            counts[name] = (
                counts.get(
                    name,
                    0,
                )
                + 1
            )

        print()
        print("=" * 70)
        print("DOMAIN DATASET")
        print("=" * 70)

        for name, idx in (
            self.domain_to_id.items()
        ):

            print(
                f"{idx:>2} "
                f"{name:<20} "
                f"{counts.get(name, 0):>8}"
            )

        print("=" * 70)


    def __getitem__(
        self,
        index,
    ):

        sample = super().__getitem__(
            index
        )

        sample["domain_id"] = torch.tensor(
            self.domain_ids[
                index
            ],
            dtype=torch.long,
        )

        return sample


    @staticmethod
    def collate_fn(
        batch,
    ):

        out = YOLODataset.collate_fn(
            batch
        )

        domain_ids = out[
            "domain_id"
        ]

        if isinstance(
            domain_ids,
            (tuple, list),
        ):

            domain_ids = torch.stack(
                list(domain_ids),
                dim=0,
            )

        out["domain_id"] = (
            domain_ids
            .long()
            .view(-1)
        )

        return out