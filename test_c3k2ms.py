import torch

from ultralytics.nn.modules import C3k2MS


def main() -> None:
    block = C3k2MS(
        c1=128,
        c2=128,
        n=1,
        kernel_size=5,
        e=0.5,
        attn=False,
        shortcut=True,
        down_ratio=0.5,
        expand_ratio=2.0,
        layers_num=1,
    ).eval()

    x = torch.randn(1, 128, 80, 80)

    with torch.no_grad():
        y = block(x)

    print("Input: ", x.shape)
    print("Output:", y.shape)

    assert y.shape == x.shape
    print("C3k2MS standalone test passed.")


if __name__ == "__main__":
    main()