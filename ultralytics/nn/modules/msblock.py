"""Multi-scale block used by the YOLO26-MS neck experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .conv import Conv

__all__ = ("MSBlock",)


class MSBlockLayer(nn.Module):
    """Large-kernel grouped expansion followed by 1x1 projection.

    This implements the YOLO-MS MSBlock_kxk_1x1_Layer idea:

        input
          -> kxk grouped convolution with channel expansion
          -> 1x1 projection back to the original branch channels
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        expand_ratio: float = 2.0,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, received {channels}")

        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError(f"MSBlockLayer kernel_size must be an odd integer larger than 1, received {kernel_size}")

        if expand_ratio < 1.0:
            raise ValueError(f"expand_ratio must be >= 1.0, received {expand_ratio}")

        expanded_channels = round(channels * expand_ratio)

        # The first convolution uses groups=channels. Therefore its output
        # channels must be divisible by channels.
        multiplier = max(1, round(expanded_channels / channels))
        expanded_channels = channels * multiplier

        # This is equivalent to a depthwise convolution with a channel
        # multiplier when expanded_channels > channels.
        self.large_kernel_conv = Conv(
            channels,
            expanded_channels,
            k=kernel_size,
            s=1,
            g=channels,
        )

        self.pointwise_conv = Conv(
            expanded_channels,
            channels,
            k=1,
            s=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the large-kernel grouped convolution and projection."""
        return self.pointwise_conv(self.large_kernel_conv(x))


class MSBlock(nn.Module):
    """Ultralytics implementation of the YOLO-MS multi-scale block.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        kernel_sizes: Kernel size for each feature branch. The official YOLO-MS neck uses three branches such as (1, 3,
            3), (1, 5, 5), or (1, 7, 7).
        down_ratio: Number of channels in each branch relative to c1. With three branches and down_ratio=0.5, the
            internal projection has 3 * 0.5 * c1 channels.
        mid_expand_ratio: Expansion ratio inside each large-kernel branch.
        layers_num: Number of MSBlockLayer modules in each processed branch.
        start_branch_id: First branch that receives the preceding branch's output before its own processing. Branch
            indices are zero-based.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_sizes: Sequence[int] = (1, 3, 3),
        down_ratio: float = 0.5,
        mid_expand_ratio: float = 2.0,
        layers_num: int = 1,
        start_branch_id: int = 2,
    ) -> None:
        super().__init__()

        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        number_of_branches = len(kernel_sizes)

        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"c1 and c2 must be positive, received c1={c1}, c2={c2}")

        if number_of_branches < 2:
            raise ValueError(f"MSBlock requires at least two branches, received {number_of_branches}")

        if down_ratio <= 0:
            raise ValueError(f"down_ratio must be positive, received {down_ratio}")

        if layers_num < 1:
            raise ValueError(f"layers_num must be at least 1, received {layers_num}")

        if not 0 <= start_branch_id < number_of_branches:
            raise ValueError(
                "start_branch_id must be within the branch range, received "
                f"{start_branch_id} for {number_of_branches} branches"
            )

        self.kernel_sizes = kernel_sizes
        self.number_of_branches = number_of_branches
        self.start_branch_id = start_branch_id

        # For the official equal three-way channel split:
        #
        # internal channels =
        #     c1 * number_of_branches * down_ratio
        #
        # Therefore each branch gets:
        #     c1 * down_ratio
        self.branch_channels = max(1, int(c1 * down_ratio))
        internal_channels = self.branch_channels * number_of_branches

        self.input_projection = Conv(
            c1,
            internal_channels,
            k=1,
            s=1,
        )

        branches: list[nn.Module] = []

        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                branches.append(nn.Identity())
                continue

            branches.append(
                nn.Sequential(
                    *[
                        MSBlockLayer(
                            channels=self.branch_channels,
                            kernel_size=kernel_size,
                            expand_ratio=mid_expand_ratio,
                        )
                        for _ in range(layers_num)
                    ]
                )
            )

        self.branches = nn.ModuleList(branches)

        self.output_projection = Conv(
            internal_channels,
            c2,
            k=1,
            s=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input through the multi-scale feature branches."""
        projected = self.input_projection(x)

        branch_inputs = projected.split(self.branch_channels, dim=1)

        if len(branch_inputs) != self.number_of_branches:
            raise RuntimeError(
                "Unexpected MSBlock channel split: expected "
                f"{self.number_of_branches} branches but received "
                f"{len(branch_inputs)}"
            )

        outputs: list[torch.Tensor] = []

        for branch_id, (branch_input, branch) in enumerate(zip(branch_inputs, self.branches)):
            # Cross-branch information flow from YOLO-MS.
            if branch_id >= self.start_branch_id:
                branch_input = branch_input + outputs[branch_id - 1]

            outputs.append(branch(branch_input))

        fused = torch.cat(outputs, dim=1)
        return self.output_projection(fused)
