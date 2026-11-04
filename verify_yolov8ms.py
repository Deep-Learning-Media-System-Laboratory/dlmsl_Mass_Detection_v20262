"""Verify the Ultralytics YOLOv8-MS port before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import MSBlock, YOLOMSBackbone, YOLOMSDetect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/v8/yolov8n-ms.yaml",
        help="Scale-specific model name; Ultralytics resolves it to yolov8-ms.yaml.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = YOLO(args.model)
    network = model.model

    backbones = [m for m in network.modules() if isinstance(m, YOLOMSBackbone)]
    msblocks = [m for m in network.modules() if isinstance(m, MSBlock)]
    heads = [m for m in network.modules() if isinstance(m, YOLOMSDetect)]

    assert len(backbones) == 1, f"Expected one YOLOMSBackbone, found {len(backbones)}"
    assert len(heads) == 1, f"Expected one YOLOMSDetect, found {len(heads)}"
    assert len(msblocks) == 8, (
        "Expected four backbone MSBlocks and four neck MSBlocks, "
        f"found {len(msblocks)}"
    )

    backbone = backbones[0]
    query_ids = {
        id(backbone.stage2.msblock),
        id(backbone.stage3.msblock),
        id(backbone.stage4.msblock),
    }
    assert len(query_ids) == 3
    assert backbone.query.shape == (3, 16), f"Unexpected shared query shape: {backbone.query.shape}"

    # Capture the three tensors entering the detection head.
    captured: dict[str, list[tuple[int, ...]]] = {}

    def pre_hook(module, inputs):
        captured["head_inputs"] = [tuple(x.shape) for x in inputs[0]]

    handle = heads[0].register_forward_pre_hook(pre_hook)

    device = torch.device(args.device)
    network.to(device).eval()
    image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)

    with torch.no_grad():
        _ = network(image)

    handle.remove()
    shapes = captured["head_inputs"]
    expected_spatial = [args.imgsz // 8, args.imgsz // 16, args.imgsz // 32]
    actual_spatial = [shape[-1] for shape in shapes]
    assert actual_spatial == expected_spatial, (
        f"Expected P3/P4/P5 spatial sizes {expected_spatial}, received {actual_spatial}"
    )

    print("YOLOv8-MS verification passed.")
    print(f"Shared query shape: {tuple(backbone.query.shape)}")
    print(f"MSBlock count: {len(msblocks)} (4 backbone + 4 neck)")
    print(f"Detection inputs: {shapes}")
    network.info()


if __name__ == "__main__":
    main()
