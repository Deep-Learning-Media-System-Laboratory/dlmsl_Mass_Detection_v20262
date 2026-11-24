from __future__ import annotations

import math

import torch
from torch import nn


class AdaptiveCrossDomainMixStyle(nn.Module):
    """Adaptive Cross-Domain MixStyle.

    Main ideas:
    1. Only mix samples from different domains.
    2. Select partners based on feature-style distance.
    3. Adapt mixing strength according to style distance.
    4. Disable mixing during validation/inference.
    """

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.1,
        target_quantile: float = 0.65,
        partner_temperature: float = 0.50,
        adaptation_strength: float = 0.75,
        distance_gamma: float = 1.0,
        use_adaptive_lambda: bool = True,
        eps: float = 1e-6,
        debug: bool = False,
    ):
        super().__init__()

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        if not 0.0 <= target_quantile <= 1.0:
            raise ValueError(f"target_quantile must be in [0, 1], got {target_quantile}")

        if partner_temperature <= 0.0:
            raise ValueError(f"partner_temperature must be > 0, got {partner_temperature}")

        if not 0.0 <= adaptation_strength <= 1.0:
            raise ValueError(f"adaptation_strength must be in [0, 1], got {adaptation_strength}")

        if distance_gamma <= 0.0:
            raise ValueError(f"distance_gamma must be > 0, got {distance_gamma}")

        self.p = float(p)
        self.alpha = float(alpha)

        self.target_quantile = float(target_quantile)

        self.partner_temperature = float(partner_temperature)

        self.adaptation_strength = float(adaptation_strength)

        self.distance_gamma = float(distance_gamma)

        self.use_adaptive_lambda = bool(use_adaptive_lambda)

        self.eps = float(eps)
        self.debug = bool(debug)

        self.last_debug = {}

    def extra_repr(self):
        return (
            f"p={self.p}, "
            f"alpha={self.alpha}, "
            f"target_quantile={self.target_quantile}, "
            f"partner_temperature={self.partner_temperature}, "
            f"adaptation_strength={self.adaptation_strength}, "
            f"distance_gamma={self.distance_gamma}"
        )

    @staticmethod
    def _prepare_domain_ids(
        domain_ids,
        batch_size,
        device,
    ):
        if not torch.is_tensor(domain_ids):
            domain_ids = torch.as_tensor(domain_ids)

        domain_ids = domain_ids.to(device=device).reshape(-1)

        if domain_ids.numel() != batch_size:
            raise ValueError(
                "Number of domain IDs must equal "
                "the batch size. "
                f"Got {domain_ids.numel()} IDs "
                f"for batch size {batch_size}."
            )

        return domain_ids

    def _style_statistics(self, x):
        mu = x.mean(
            dim=(2, 3),
            keepdim=True,
        ).detach()

        var = x.var(
            dim=(2, 3),
            unbiased=False,
            keepdim=True,
        ).detach()

        sigma = (var + self.eps).sqrt()

        x_norm = (x - mu) / sigma

        return (
            mu,
            sigma,
            x_norm,
        )

    def _style_embedding(
        self,
        mu,
        sigma,
    ):
        style = torch.cat(
            [
                mu.flatten(start_dim=1),
                sigma.flatten(start_dim=1),
            ],
            dim=1,
        ).float()

        center = style.mean(
            dim=0,
            keepdim=True,
        )

        scale = style.std(
            dim=0,
            unbiased=False,
            keepdim=True,
        ).clamp_min(self.eps)

        style = (style - center) / scale

        return style

    def _select_cross_domain_partners(
        self,
        style,
        domain_ids,
    ):
        style.shape[0]

        distances = torch.cdist(
            style,
            style,
            p=2,
        )

        distances = distances / math.sqrt(max(style.shape[1], 1))

    def _select_cross_domain_partners(
        self,
        style,
        domain_ids,
    ):
        batch_size = style.shape[0]
        device = style.device

        distances = torch.cdist(
            style,
            style,
            p=2,
        )

        distances = distances / math.sqrt(max(style.shape[1], 1))

        cross_domain = domain_ids[:, None] != domain_ids[None, :]

        valid_distances = distances[cross_domain]

        if valid_distances.numel() == 0:
            idx = torch.arange(
                batch_size,
                device=device,
            )

            zeros = torch.zeros(
                batch_size,
                device=device,
            )

            return (
                idx,
                zeros,
                zeros,
            )

        global_min = valid_distances.min()

        global_max = valid_distances.max()

        global_range = (global_max - global_min).clamp_min(self.eps)

        partner_idx = torch.empty(
            batch_size,
            dtype=torch.long,
            device=device,
        )

        selected_distance = torch.empty(
            batch_size,
            dtype=torch.float32,
            device=device,
        )

        for i in range(batch_size):
            candidates = torch.nonzero(
                cross_domain[i],
                as_tuple=False,
            ).flatten()

            candidate_dist = distances[
                i,
                candidates,
            ]

            if candidates.numel() == 1:
                chosen_local = torch.tensor(
                    0,
                    device=device,
                )

            else:
                target = torch.quantile(
                    candidate_dist,
                    self.target_quantile,
                )

                local_scale = candidate_dist.std(unbiased=False).clamp_min(self.eps)

                score = -torch.abs(candidate_dist - target)

                score = score / (self.partner_temperature * local_scale + self.eps)

                probability = torch.softmax(
                    score,
                    dim=0,
                )

                if not torch.isfinite(probability).all() or probability.sum() <= 0:
                    probability = torch.ones_like(probability) / probability.numel()

                chosen_local = torch.multinomial(
                    probability,
                    num_samples=1,
                    replacement=True,
                ).squeeze(0)
            chosen = candidates[chosen_local]

            partner_idx[i] = chosen

            selected_distance[i] = distances[i, chosen]

        normalized_distance = ((selected_distance - global_min) / global_range).clamp(
            0.0,
            1.0,
        )

        return (
            partner_idx,
            selected_distance,
            normalized_distance,
        )

    def _sample_base_lambda(
        self,
        batch_size,
        device,
    ):
        concentration = torch.tensor(
            self.alpha,
            device=device,
            dtype=torch.float32,
        )

        beta = torch.distributions.Beta(
            concentration,
            concentration,
        )

        base_lambda = beta.sample((batch_size,))

        base_lambda = base_lambda.view(
            batch_size,
            1,
            1,
            1,
        )

        return base_lambda

    def _adaptive_lambda(
        self,
        base_lambda,
        normalized_distance,
    ):
        distance_weight = normalized_distance.pow(self.distance_gamma)

        distance_weight = distance_weight.view(
            -1,
            1,
            1,
            1,
        )

        adaptive_lambda = base_lambda + (1.0 - base_lambda) * self.adaptation_strength * distance_weight

        return adaptive_lambda.clamp(
            0.0,
            1.0,
        )

    def forward(
        self,
        x,
        domain_ids=None,
    ):
        self.last_debug = {"applied": False}

        if x.ndim != 4:
            raise ValueError(f"AdaptiveCrossDomainMixStyle expects BCHW features. Got shape {tuple(x.shape)}")

        if not self.training:
            self.last_debug = {
                "applied": False,
                "reason": "eval_mode",
            }

            return x

        batch_size = x.shape[0]

        if batch_size < 2:
            self.last_debug = {
                "applied": False,
                "reason": "batch_too_small",
            }

            return x

        if domain_ids is None:
            raise ValueError("domain_ids is required while AdaptiveCrossDomainMixStyle is in training mode.")

        domain_ids = self._prepare_domain_ids(
            domain_ids,
            batch_size,
            x.device,
        )

        if torch.unique(domain_ids).numel() < 2:
            self.last_debug = {
                "applied": False,
                "reason": "single_domain_batch",
            }

            return x

        if (
            torch.rand(
                (),
                device=x.device,
            )
            > self.p
        ):
            self.last_debug = {
                "applied": False,
                "reason": "probability_gate",
            }

            return x

        (
            mu,
            sigma,
            x_norm,
        ) = self._style_statistics(x)

        style = self._style_embedding(
            mu,
            sigma,
        )

        with torch.no_grad():
            (
                partner_idx,
                selected_distance,
                normalized_distance,
            ) = self._select_cross_domain_partners(
                style,
                domain_ids,
            )

            base_lambda = self._sample_base_lambda(
                batch_size,
                x.device,
            )

            if self.use_adaptive_lambda:
                adaptive_lambda = self._adaptive_lambda(
                    base_lambda,
                    normalized_distance,
                )

            else:
                # Adaptive Partner Only:
                # keep the original MixStyle lambda.
                adaptive_lambda = base_lambda.clone()

        partner_domain_ids = domain_ids[partner_idx]

        if torch.any(partner_domain_ids == domain_ids):
            raise RuntimeError("Adaptive Cross-Domain MixStyle selected a same-domain partner.")

        adaptive_lambda = adaptive_lambda.to(dtype=x.dtype)

        mu_partner = mu[partner_idx]

        sigma_partner = sigma[partner_idx]

        mu_mix = adaptive_lambda * mu + (1.0 - adaptive_lambda) * mu_partner

        sigma_mix = adaptive_lambda * sigma + (1.0 - adaptive_lambda) * sigma_partner

        out = x_norm * sigma_mix + mu_mix

        self.last_debug = {
            "applied": True,
            "use_adaptive_lambda": self.use_adaptive_lambda,
            "source_domains": domain_ids.detach().cpu().tolist(),
            "partner_domains": partner_domain_ids.detach().cpu().tolist(),
            "partner_indices": partner_idx.detach().cpu().tolist(),
            "style_distance": selected_distance.detach().cpu().tolist(),
            "normalized_distance": normalized_distance.detach().cpu().tolist(),
            "base_lambda": base_lambda.flatten().detach().cpu().tolist(),
            "adaptive_lambda": adaptive_lambda.flatten().detach().float().cpu().tolist(),
        }
        if self.debug:
            self._print_debug()

        return out

    def _print_debug(
        self,
        max_samples=8,
    ):
        if not self.last_debug.get(
            "applied",
            False,
        ):
            print(
                "[AD-CDMS] not applied:",
                self.last_debug.get(
                    "reason",
                    "unknown",
                ),
            )
            return

        n = min(
            len(self.last_debug["source_domains"]),
            max_samples,
        )

        print("\n[AD-CDMS] adaptive mixing")

        print("-" * 78)

        for i in range(n):
            print(
                f"sample={i:02d} "
                f"src="
                f"{self.last_debug['source_domains'][i]} "
                f"partner="
                f"{self.last_debug['partner_domains'][i]} "
                f"d="
                f"{self.last_debug['style_distance'][i]:.4f} "
                f"d_norm="
                f"{self.last_debug['normalized_distance'][i]:.4f} "
                f"lambda_base="
                f"{self.last_debug['base_lambda'][i]:.4f} "
                f"lambda_adapt="
                f"{self.last_debug['adaptive_lambda'][i]:.4f}"
            )

        print("-" * 78)
