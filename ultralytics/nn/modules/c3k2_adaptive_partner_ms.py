from __future__ import annotations

from .c3k2_adaptive_cdms import (
    C3k2AdaptiveCDMixStyle,
)


class C3k2AdaptivePartnerMixStyle(
    C3k2AdaptiveCDMixStyle
):
    """
    Adaptive Partner MixStyle.

    Adaptive partner selection: ON
    Adaptive lambda: OFF

    Lambda remains standard MixStyle:
        lambda ~ Beta(alpha, alpha)
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

        mix_p=0.5,
        mix_alpha=0.1,

        target_quantile=0.65,
        partner_temperature=0.50,

        adaptation_strength=0.75,
        distance_gamma=1.0,

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

            mix_p=mix_p,
            mix_alpha=mix_alpha,

            target_quantile=target_quantile,

            partner_temperature=(
                partner_temperature
            ),

            adaptation_strength=(
                adaptation_strength
            ),

            distance_gamma=(
                distance_gamma
            ),

            # --------------------------------
            # IMPORTANT ABLATION
            # --------------------------------
            use_adaptive_lambda=False,

            debug=debug,
        )