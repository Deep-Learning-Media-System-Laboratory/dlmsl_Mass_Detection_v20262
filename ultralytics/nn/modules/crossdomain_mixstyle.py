from __future__ import annotations

import torch
from torch import nn

from .block import C3k2


class CrossDomainMixStyle(nn.Module):
    """Balanced Cross-Domain MixStyle.

    Difference from Random MixStyle:
    --------------------------------
    Random MixStyle:
        donor = random image in batch

    Cross-Domain MixStyle:
        1. donor MUST come from another domain
        2. donor DOMAIN is chosen uniformly
        3. donor IMAGE is chosen randomly from that domain

    Training only.
    """

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.1,
        eps: float = 1e-6,
        balanced: bool = True,
    ):
        super().__init__()

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be between 0 and 1, got {p}")

        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.balanced = bool(balanced)

        # Debug: confirm which donor mode is active
        mode_name = "balanced-domain" if self.balanced else "image-uniform"

        print(f"[CrossDomainMixStyle] mode={mode_name}, p={self.p}, alpha={self.alpha}")

        self.beta = torch.distributions.Beta(
            alpha,
            alpha,
        )

        # Set by trainer before each real training forward.
        self._domain_ids = None

        # False during model construction/dummy forward.
        # Trainer enables it during actual training.
        self.strict_runtime = False

        # ----------------------------------------------------
        # Debug / audit counters
        # ----------------------------------------------------
        self.total_calls = 0
        self.total_applied = 0
        self.total_probability_skips = 0
        self.total_single_domain_skips = 0
        self.total_pairs = 0
        self.total_same_domain_pairs = 0

        self.last_source_domains = None
        self.last_donor_domains = None

    def set_domain_ids(
        self,
        domain_ids: torch.Tensor,
    ):
        self._domain_ids = domain_ids

    def set_strict_runtime(
        self,
        enabled: bool = True,
    ):
        self.strict_runtime = enabled

    @torch.no_grad()
    def build_cross_domain_perm(
        self,
        domain_ids: torch.Tensor,
    ):
        """Build cross-domain donor indices.

        balanced=True:
            1. choose another DOMAIN uniformly
            2. choose one IMAGE uniformly from that domain

        balanced=False:
            choose any IMAGE from another domain uniformly

        In both modes:
            donor domain MUST differ from source domain.
        """
        domain_ids = domain_ids.view(-1)

        B = domain_ids.numel()
        device = domain_ids.device

        unique_domains = torch.unique(domain_ids)

        # Need >= 2 domains
        if unique_domains.numel() < 2:
            return None

        perm = torch.empty(
            B,
            dtype=torch.long,
            device=device,
        )

        for i in range(B):
            source_domain = domain_ids[i]

            # ==================================================
            # MODE 1:
            # Balanced donor-domain sampling
            # ==================================================
            if self.balanced:
                # Candidate domains = all except source domain
                donor_domains = unique_domains[unique_domains != source_domain]

                # Choose donor DOMAIN uniformly
                domain_choice = torch.randint(
                    low=0,
                    high=donor_domains.numel(),
                    size=(),
                    device=device,
                )

                donor_domain = donor_domains[domain_choice]

                # Images belonging to selected donor domain
                candidate_images = torch.where(domain_ids == donor_domain)[0]

            # ==================================================
            # MODE 2:
            # Image-uniform cross-domain sampling
            # ==================================================
            else:
                # Every image from every OTHER domain
                # becomes a candidate.
                candidate_images = torch.where(domain_ids != source_domain)[0]

            # --------------------------------------------------
            # Choose donor IMAGE uniformly
            # --------------------------------------------------
            image_choice = torch.randint(
                low=0,
                high=candidate_images.numel(),
                size=(),
                device=device,
            )

            perm[i] = candidate_images[image_choice]

        # ======================================================
        # HARD SAFETY CHECK
        # ======================================================

        source_domains = domain_ids
        donor_domains = domain_ids[perm]

        same = int((source_domains == donor_domains).sum().item())

        if same != 0:
            raise RuntimeError(f"CrossDomainMixStyle produced {same} SAME-DOMAIN donor pairs.")

        return perm

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Validation / testing / inference
        # ----------------------------------------------------

        if not self.training:
            return x

        self.total_calls += 1

        # ----------------------------------------------------
        # Get current batch domain IDs
        # ----------------------------------------------------

        domain_ids = self._domain_ids

        # Clear reference so previous batch cannot be reused
        self._domain_ids = None

        if domain_ids is None:
            if self.strict_runtime:
                raise RuntimeError("CrossDomainMixStyle did not receive domain_ids for this batch.")

            # Dummy forward during model construction
            return x

        if x.shape[0] < 2:
            return x

        domain_ids = domain_ids.to(
            device=x.device,
            dtype=torch.long,
        ).view(-1)

        if domain_ids.numel() != x.shape[0]:
            raise RuntimeError(
                "Domain ID count does not match "
                "feature batch size:\n"
                f"domain_ids = {domain_ids.numel()}\n"
                f"features   = {x.shape[0]}"
            )

        # ----------------------------------------------------
        # Apply with probability p
        # ----------------------------------------------------

        if (
            torch.rand(
                (),
                device=x.device,
            ).item()
            > self.p
        ):
            self.total_probability_skips += 1

            return x

        # ----------------------------------------------------
        # Cross-domain donor selection
        # ----------------------------------------------------

        perm = self.build_cross_domain_perm(domain_ids)

        if perm is None:
            self.total_single_domain_skips += 1

            return x

        self.total_applied += 1
        self.total_pairs += domain_ids.numel()

        source_domains = domain_ids
        donor_domains = domain_ids[perm]

        same = int((source_domains == donor_domains).sum().item())

        self.total_same_domain_pairs += same

        if same != 0:
            raise RuntimeError(f"Illegal same-domain pairs: {same}")

        self.last_source_domains = source_domains.detach().cpu()

        self.last_donor_domains = donor_domains.detach().cpu()

        # ----------------------------------------------------
        # Feature mean/std
        # ----------------------------------------------------

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

        mu = mu.detach()
        sigma = sigma.detach()

        x_norm = (x - mu) / sigma

        # Donor statistics
        mu_other = mu[perm]
        sigma_other = sigma[perm]

        # ----------------------------------------------------
        # lambda ~ Beta(alpha, alpha)
        # ----------------------------------------------------

        B = x.shape[0]

        lam = self.beta.sample((B, 1, 1, 1)).to(
            device=x.device,
            dtype=x.dtype,
        )

        mu_mix = lam * mu + (1.0 - lam) * mu_other

        sigma_mix = lam * sigma + (1.0 - lam) * sigma_other

        return x_norm * sigma_mix + mu_mix


class C3k2CrossDomainMixStyle(C3k2):
    """YOLO26 C3k2 + Cross-Domain MixStyle."""

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
        mixstyle_balanced=True,
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

        self.mixstyle = CrossDomainMixStyle(
            p=mixstyle_p,
            alpha=mixstyle_alpha,
            balanced=False,
        )

    def forward(self, x):

        x = super().forward(x)

        return self.mixstyle(x)
