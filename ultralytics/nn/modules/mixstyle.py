from __future__ import annotations

import torch
from torch import nn

from .block import C3k2


class MixStyle(nn.Module):
    """MixStyle Zhou et al., ICLR 2021.

    Training:
        randomly mixes feature statistics between samples.

    Validation/inference:
        disabled automatically.
    """

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.p = p
        self.alpha = alpha
        self.eps = eps

        self.beta = torch.distributions.Beta(
            alpha,
            alpha,
        )

    def forward(self, x):

        # Disable during validation/inference
        if not self.training:
            return x

        # Cannot mix if batch contains one image
        if x.size(0) < 2:
            return x

        # Apply with probability p
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        B = x.size(0)

        # Instance-wise feature statistics
        mu = x.mean(
            dim=(2, 3),
            keepdim=True,
        )

        var = x.var(
            dim=(2, 3),
            keepdim=True,
            correction=0,
        )

        sigma = torch.sqrt(var + self.eps)

        # Do not propagate gradients through style statistics
        mu = mu.detach()
        sigma = sigma.detach()

        # Normalize content
        x_norm = (x - mu) / sigma

        # Random style donor
        perm = torch.randperm(
            B,
            device=x.device,
        )

        mu2 = mu[perm]
        sigma2 = sigma[perm]

        # Beta interpolation
        lam = self.beta.sample((B, 1, 1, 1)).to(
            device=x.device,
            dtype=x.dtype,
        )

        mu_mix = lam * mu + (1.0 - lam) * mu2

        sigma_mix = lam * sigma + (1.0 - lam) * sigma2

        return x_norm * sigma_mix + mu_mix


class C3k2MixStyle(C3k2):
    """YOLO26 C3k2 + MixStyle.

    The original C3k2 parameter structure is preserved, which helps pretrained weight transfer.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        attn=False,
        g=1,
        shortcut=True,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1,
    ):

        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            attn=attn,
            g=g,
            shortcut=shortcut,
        )

        self.mixstyle = MixStyle(
            p=mixstyle_p,
            alpha=mixstyle_alpha,
        )

    def forward(self, x):

        x = super().forward(x)

        x = self.mixstyle(x)

        return x
