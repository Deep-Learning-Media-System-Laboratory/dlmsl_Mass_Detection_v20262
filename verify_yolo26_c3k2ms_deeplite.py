from __future__ import annotations

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, C3k2MS


MODEL = "ultralytics/cfg/models/26/yolo26n-c3k2ms-deeplite.yaml"


def print_gates(model) -> None:
    print("\nLearnable residual gates:")

    found = False

    for name, parameter in model.named_parameters():
        if "residual_scale" in name:
            found = True
            print(
                f"{name}: "
                f"{parameter.detach().cpu().item():.6f}"
            )

    if not found:
        raise RuntimeError(
            "No residual_scale parameters were found."
        )


def main() -> None:
    yolo = YOLO(MODEL)
    network = yolo.model.eval()

    expected = {
        13: C3k2,
        16: C3k2,
        19: C3k2MS,
        22: C3k2MS,
    }

    for index, expected_type in expected.items():
        layer = network.model[index]

        print(
            f"Layer {index}: "
            f"{layer.__class__.__name__}"
        )

        if not isinstance(layer, expected_type):
            raise RuntimeError(
                f"Layer {index} should be "
                f"{expected_type.__name__}, but found "
                f"{layer.__class__.__name__}"
            )

    print_gates(network)

    image = torch.zeros(
        1,
        3,
        640,
        640,
    )

    with torch.no_grad():
        output = network(image)

    print("\nForward output type:", type(output))
    network.info()

    print(
        "\nYOLO26-C3k2MS-DeepLite verification passed."
    )


if __name__ == "__main__":
    main()