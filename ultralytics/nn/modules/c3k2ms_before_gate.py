"""C3k2-MS hybrid block for YOLO26."""

from __future__ import annotations

import math

import torch
from torch import nn

from .block import C2f, PSABlock
from .conv import Conv

__all__ = ("MSKernelLayer", "MSBottleneck", "C3k2MS")


class MSKernelLayer(nn.Module):
    """Large-kernel grouped convolution followed by 1x1 projection.

    This follows the YOLO-MS kxk -> 1x1 branch-processing idea.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        expand_ratio: float = 2.0,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, received {channels}"
            )

        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be an odd integer larger than 1, "
                f"received {kernel_size}"
            )

        if expand_ratio < 1.0:
            raise ValueError(
                f"expand_ratio must be >= 1.0, received {expand_ratio}"
            )

        expanded_channels = int(channels * expand_ratio)

        # groups=channels requires output channels to be divisible
        # by the number of input channels.
        multiplier = max(
            1,
            math.ceil(expanded_channels / channels),
        )
        expanded_channels = channels * multiplier

        self.large_kernel = Conv(
            channels,
            expanded_channels,
            k=kernel_size,
            s=1,
            g=channels,
        )

        self.pointwise = Conv(
            expanded_channels,
            channels,
            k=1,
            s=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped large-kernel convolution and projection."""
        return self.pointwise(self.large_kernel(x))


class MSBottleneck(nn.Module):
    """Compact hierarchical MS operator for use inside C3k2.

    Architecture:

        input
          ↓
        1x1 projection
          ↓
        split into three branches
          ├── branch 0: identity
          ├── branch 1: kxk -> 1x1
          └── branch 2:
                input + branch 1 output
                -> kxk -> 1x1
          ↓
        concatenate
          ↓
        1x1 output projection
          ↓
        optional residual connection

    This preserves the central YOLO-MS behavior:

        branch_2_input = branch_2_input + branch_1_output
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 5,
        down_ratio: float = 0.5,
        expand_ratio: float = 2.0,
        layers_num: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        if down_ratio <= 0:
            raise ValueError(
                f"down_ratio must be positive, received {down_ratio}"
            )

        if layers_num < 1:
            raise ValueError(
                f"layers_num must be >= 1, received {layers_num}"
            )

        branch_channels = max(
            1,
            int(c1 * down_ratio),
        )
        internal_channels = branch_channels * 3

        self.branch_channels = branch_channels

        self.input_projection = Conv(
            c1,
            internal_channels,
            k=1,
            s=1,
        )

        self.branch_1 = nn.Sequential(
            *[
                MSKernelLayer(
                    channels=branch_channels,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                )
                for _ in range(layers_num)
            ]
        )

        self.branch_2 = nn.Sequential(
            *[
                MSKernelLayer(
                    channels=branch_channels,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                )
                for _ in range(layers_num)
            ]
        )

        self.output_projection = Conv(
            internal_channels,
            c2,
            k=1,
            s=1,
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run hierarchical multi-scale processing."""
        projected = self.input_projection(x)

        branch_0, branch_1, branch_2 = projected.split(
            self.branch_channels,
            dim=1,
        )

        # First branch remains relatively unprocessed.
        output_0 = branch_0

        # Second branch receives large-kernel processing.
        output_1 = self.branch_1(branch_1)

        # Hierarchical cross-branch propagation from YOLO-MS.
        output_2 = self.branch_2(
            branch_2 + output_1
        )

        output = torch.cat(
            (output_0, output_1, output_2),
            dim=1,
        )

        output = self.output_projection(output)

        return x + output if self.add else output


class C3k2MS(C2f):
    """C3k2 outer structure with MS-style internal processing.

    The outer C2f flow remains unchanged. Only the modules stored in
    ``self.m`` are replaced by MSBottleneck modules.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of repeated internal blocks.
        kernel_size: Large kernel used in the MS branches.
        e: C2f hidden-channel expansion ratio.
        attn: Add YOLO26 PSA after each MSBottleneck.
        g: Retained for C2f compatibility.
        shortcut: Enable residual connections.
        down_ratio: Width of each MS branch relative to hidden channels.
        expand_ratio: Expansion inside each large-kernel branch.
        layers_num: Repeated large-kernel layers inside each branch.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        kernel_size: int = 5,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        down_ratio: float = 0.5,
        expand_ratio: float = 2.0,
        layers_num: int = 1,
    ) -> None:
        # Construct the original C2f outer structure:
        # cv1, cv2, hidden channel count, and normal forward flow.
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            shortcut=shortcut,
            g=g,
            e=e,
        )

        blocks: list[nn.Module] = []

        for _ in range(n):
            ms_block = MSBottleneck(
                c1=self.c,
                c2=self.c,
                kernel_size=kernel_size,
                down_ratio=down_ratio,
                expand_ratio=expand_ratio,
                layers_num=layers_num,
                shortcut=shortcut,
            )

            if attn:
                blocks.append(
                    nn.Sequential(
                        ms_block,
                        PSABlock(
                            self.c,
                            attn_ratio=0.5,
                            num_heads=max(self.c // 64, 1),
                        ),
                    )
                )
            else:
                blocks.append(ms_block)

        self.m = nn.ModuleList(blocks)