from ultralytics import YOLO


CONFIG = "ultralytics/cfg/models/26/yolo26n-msneck.yaml"


def main() -> None:
    model = YOLO(CONFIG)

    # Downloads yolo26n.pt automatically when it is not already available.
    model.load("yolo26n.pt")

    model.info()

    print("Compatible YOLO26n weights transferred successfully.")


if __name__ == "__main__":
    main()