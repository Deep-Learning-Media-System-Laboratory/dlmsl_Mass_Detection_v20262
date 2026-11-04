from ultralytics import YOLO


WEIGHTS = (
    "/home/yeozekiel/yolo/YOLO-MS/ultralytics/runs/detect/runs/detect/with-embed-cmmd-normal_yolo26n-deeplite-gate01-4/weights/best.pt"
)


def main() -> None:
    model = YOLO(WEIGHTS)

    gates = []

    for name, parameter in model.model.named_parameters():
        if "residual_scale" in name:
            value = parameter.detach().cpu().item()
            gates.append((name, value))

    if not gates:
        raise RuntimeError(
            "No residual gate parameters found."
        )

    print("\nLearned C3k2MS residual gates:")

    for name, value in gates:
        print(f"{name}: {value:.6f}")

    mean_gate = sum(value for _, value in gates) / len(gates)

    print(f"\nMean gate: {mean_gate:.6f}")


if __name__ == "__main__":
    main()