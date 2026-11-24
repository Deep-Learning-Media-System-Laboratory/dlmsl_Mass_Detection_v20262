import torch

from ultralytics.nn.modules.crossdomain_mixstyle import (
    CrossDomainMixStyle,
)

# ============================================================
# Intentionally imbalanced
# ============================================================

domain_ids = torch.tensor(
    [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        3,
    ],
    dtype=torch.long,
)


module = CrossDomainMixStyle(
    p=1.0,
    alpha=0.1,
)


# ============================================================
# DONOR TEST
# ============================================================

same_domain = 0
total_pairs = 0


for _ in range(10000):
    perm = module.build_cross_domain_perm(domain_ids)

    source = domain_ids
    donor = domain_ids[perm]

    same_domain += int((source == donor).sum().item())

    total_pairs += len(domain_ids)


print(
    "Total pairs:",
    total_pairs,
)

print(
    "Same-domain pairs:",
    same_domain,
)


assert same_domain == 0


# ============================================================
# FEATURE TEST
# ============================================================

x = torch.randn(
    len(domain_ids),
    128,
    80,
    80,
)


module.train()

module.set_strict_runtime(True)

module.set_domain_ids(domain_ids)

y = module(x)


assert x.shape == y.shape
assert not torch.equal(
    x,
    y,
)


print(
    "Feature shape:",
    y.shape,
)


# ============================================================
# EVAL TEST
# ============================================================

module.eval()

z = module(x)


assert torch.equal(
    x,
    z,
)


print()
print("PASS:")
print("- no same-domain donor")
print("- feature changed during training")
print("- feature unchanged during evaluation")
