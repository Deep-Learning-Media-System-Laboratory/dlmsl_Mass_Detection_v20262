import torch

from ultralytics.nn.tasks import DetectionModel


CONFIG = "ultralytics/cfg/models/26/yolo26n-msneck.yaml"


def main() -> None:
    model = DetectionModel(
        cfg=CONFIG,
        ch=3,
        nc=1,
        verbose=True,
    ).eval()

    expected_layers = (13, 16, 19, 22)

    for index in expected_layers:
        layer_name = model.model[index].__class__.__name__
        print(f"Layer {index}: {layer_name}")

        if layer_name != "MSBlock":
            raise RuntimeError(
                f"Expected MSBlock at layer {index}, found {layer_name}"
            )

    x = torch.zeros(1, 3, 640, 640)

    with torch.no_grad():
        output = model(x)

    model.info()

    print(f"Forward output type: {type(output)}")
    print("YOLO26-MSNeck construction and forward pass succeeded.")


if __name__ == "__main__":
    main()