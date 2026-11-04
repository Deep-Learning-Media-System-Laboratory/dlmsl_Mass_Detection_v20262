from copy import copy

from ultralytics.models.yolo.detect import (
    DetectionTrainer,
)

from ultralytics.utils import colorstr

from ultralytics.utils.torch_utils import (
    unwrap_model,
)

from ultralytics.nn.modules.crossdomain_mixstyle import (
    CrossDomainMixStyle,
)

from cd_mixstyle_data import (
    DomainYOLODataset,
)


class DomainMixStyleTrainer(
    DetectionTrainer
):

    def build_dataset(
        self,
        img_path,
        mode="train",
        batch=None,
    ):

        # Validation remains normal YOLO.
        if mode != "train":

            return super().build_dataset(
                img_path,
                mode,
                batch,
            )

        gs = max(
            int(
                unwrap_model(
                    self.model
                ).stride.max()
            ),
            32,
        )

        return DomainYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=copy(
                self.args
            ),
            rect=self.args.rect,
            cache=(
                self.args.cache
                or None
            ),
            single_cls=(
                self.args.single_cls
                or False
            ),
            stride=gs,
            pad=0.0,
            prefix=colorstr(
                "train: "
            ),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )


    def preprocess_batch(
        self,
        batch,
    ):

        # Normal Ultralytics preprocessing.
        batch = super().preprocess_batch(
            batch
        )

        if "domain_id" not in batch:

            raise RuntimeError(
                "domain_id missing from batch."
            )

        domain_ids = (
            batch[
                "domain_id"
            ]
            .long()
            .view(-1)
        )

        if domain_ids.numel() != (
            batch["img"].shape[0]
        ):

            raise RuntimeError(
                "domain_id batch size mismatch."
            )

        model = unwrap_model(
            self.model
        )

        found = 0

        for module in model.modules():

            if isinstance(
                module,
                CrossDomainMixStyle,
            ):

                module.set_strict_runtime(
                    True
                )

                module.set_domain_ids(
                    domain_ids
                )

                found += 1

        if found == 0:

            raise RuntimeError(
                "No CrossDomainMixStyle "
                "module found."
            )

        # ----------------------------------------------
        # First batch debug
        # ----------------------------------------------

        if not hasattr(
            self,
            "_printed_cdms_batch",
        ):

            self._printed_cdms_batch = True

            ids = (
                domain_ids
                .detach()
                .cpu()
                .tolist()
            )

            counts = {}

            for d in ids:

                counts[d] = (
                    counts.get(
                        d,
                        0,
                    )
                    + 1
                )

            print()
            print("=" * 70)
            print(
                "FIRST CROSS-DOMAIN TRAIN BATCH"
            )
            print("=" * 70)

            print(
                "Domain IDs:",
                ids,
            )

            print(
                "Counts:",
                counts,
            )

            print(
                "CDMS modules:",
                found,
            )

            print("=" * 70)

        return batch