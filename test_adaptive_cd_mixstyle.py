import torch

from ultralytics.nn.modules.adaptive_cd_mixstyle import (
    AdaptiveCrossDomainMixStyle,
)

torch.manual_seed(42)


# ============================================================
# FAKE YOLO FEATURE MAP
# ============================================================

x = torch.randn(
    8,
    128,
    20,
    20,
    requires_grad=True,
)


# ============================================================
# FAKE DOMAIN LABELS
# ============================================================

# Example:
#
# 0 = CMMD
# 1 = CBIS
# 2 = Hospital
# 3 = BCS-DBT

domain_ids = torch.tensor(
    [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
)


# ============================================================
# CREATE MODULE
# ============================================================

mixstyle = AdaptiveCrossDomainMixStyle(
    p=1.0,
    alpha=0.1,
    target_quantile=0.65,
    partner_temperature=0.50,
    adaptation_strength=0.75,
    distance_gamma=1.0,
    debug=True,
)


# ============================================================
# TRAIN MODE
# ============================================================

mixstyle.train()


# ============================================================
# FORWARD
# ============================================================

y = mixstyle(
    x,
    domain_ids=domain_ids,
)


print()
print(
    "Input shape :",
    x.shape,
)

print(
    "Output shape:",
    y.shape,
)


# ============================================================
# BASIC CHECK
# ============================================================

assert x.shape == y.shape

print("Shape test: PASS")


# ============================================================
# CHECK NUMBERS
# ============================================================

assert torch.isfinite(y).all()

print("Finite output: PASS")


# ============================================================
# GRADIENT CHECK
# ============================================================

loss = y.mean()

loss.backward()


assert x.grad is not None

assert torch.isfinite(x.grad).all()

print("Gradient test: PASS")


# ============================================================
# CROSS-DOMAIN CHECK
# ============================================================

debug = mixstyle.last_debug

source = debug["source_domains"]

partner = debug["partner_domains"]


for src, dst in zip(
    source,
    partner,
):
    assert src != dst


print("Cross-domain test: PASS")


# ============================================================
# EVAL MODE TEST
# ============================================================

mixstyle.eval()

with torch.no_grad():
    y_eval = mixstyle(
        x.detach(),
        domain_ids=domain_ids,
    )


assert torch.equal(
    x.detach(),
    y_eval,
)

print("Eval identity test: PASS")


print()
print("======================================")

print("ALL ADAPTIVE CD MIXSTYLE TESTS PASSED")

print("======================================")
