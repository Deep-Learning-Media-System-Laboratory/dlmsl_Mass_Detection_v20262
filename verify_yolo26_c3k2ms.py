from __future__ import annotations

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import C3k2MS


MODEL = "ultralytics/cfg/models/26/yolo26n-c3k2ms.yaml"


def main() -> None:
    yolo = YOLO(MODEL)
    network = yolo.model.eval()

    expected_layers = (13, 16, 19, 22)

    for layer_index in expected_layers:
        layer = network.model[layer_index]

        print(
            f"Layer {layer_index}: "
            f"{layer.__class__.__name__}"
        )

        if not isinstance(layer, C3k2MS):
            raise RuntimeError(
                f"Expected C3k2MS at layer {layer_index}, "
                f"found {layer.__class__.__name__}"
            )

    x = torch.zeros(1, 3, 640, 640)

    with torch.no_grad():
        output = network(x)

    network.info()

    print("Output type:", type(output))
    print("YOLO26-C3k2MS verification passed.")

    # results = yolo.train(
    #     data="all.yaml",
    #     epochs=1,
    #     imgsz=640,
    #     batch=2,
    #     device="1",
    #     workers=2,
    #     project="runs/debug",
    #     name="yolo26n-c3k2ms-smoke-test",
    #     amp=True,
    # )


if __name__ == "__main__":
    main()