import torch

from ultralytics.nn.modules.mixstyle import MixStyle

x = torch.randn(
    8,
    128,
    80,
    80,
)

ms = MixStyle(
    p=1.0,
    alpha=0.1,
)

# Training mode
ms.train()

y = ms(x)

print("Input :", x.shape)
print("Output:", y.shape)
print("Changed:", not torch.equal(x, y))

ms.eval()

z = ms(x)

print(torch.equal(x, z))
