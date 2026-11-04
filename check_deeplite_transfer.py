from ultralytics import YOLO

MODEL_YAML = "ultralytics/cfg/models/26/yolo26n-c3k2ms-deeplite.yaml"


def main() -> None:
    model = YOLO(MODEL_YAML)

    print("\nLoading standard YOLO26n weights...")
    model.load("yolo26n.pt")

    model.info(detailed=True)

    print("\nGate values after weight transfer:")

    for name, parameter in model.model.named_parameters():
        if "residual_scale" in name:
            print(
                name,
                parameter.detach().cpu().item(),
            )


if __name__ == "__main__":
    main()
