from __future__ import annotations

from .adaptive_cd_mixstyle import AdaptiveCrossDomainMixStyle
from .block import C3k2


class C3k2AdaptiveCDMixStyle(C3k2):
    """YOLO C3k2 followed by Adaptive Cross-Domain MixStyle.

    This class is intentionally separate from the existing Random MixStyle / Cross-Domain MixStyle implementation.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
        # Adaptive MixStyle settings
        mix_p=0.5,
        mix_alpha=0.1,
        target_quantile=0.65,
        partner_temperature=0.50,
        adaptation_strength=0.75,
        distance_gamma=1.0,
        use_adaptive_lambda=True,
        debug=False,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            g=g,
            shortcut=shortcut,
        )

        self.adaptive_mixstyle = AdaptiveCrossDomainMixStyle(
            p=mix_p,
            alpha=mix_alpha,
            target_quantile=target_quantile,
            partner_temperature=partner_temperature,
            adaptation_strength=adaptation_strength,
            distance_gamma=distance_gamma,
            use_adaptive_lambda=use_adaptive_lambda,
            debug=debug,
        )

        # Trainer will populate this later.
        self._domain_ids = None

    def set_domain_ids(self, domain_ids):
        """Store domain IDs for the current training batch.

        Examples:
            [0, 0, 1, 1, 2, 2, 3, 3]
        """
        self._domain_ids = domain_ids

    def clear_domain_ids(self):
        self._domain_ids = None

    def forward(self, x):
        # Normal C3k2 forward
        x = super().forward(x)

        # Validation / inference:
        # never apply Adaptive MixStyle.
        if not self.training:
            return x

        # Important:
        # Ultralytics may run dummy forward passes while
        # constructing the model before domain IDs exist.
        if self._domain_ids is None:
            return x

        # Adaptive Cross-Domain MixStyle
        x = self.adaptive_mixstyle(
            x,
            domain_ids=self._domain_ids,
        )

        return x

    @property
    def mixstyle_debug(self):
        return self.adaptive_mixstyle.last_debug
