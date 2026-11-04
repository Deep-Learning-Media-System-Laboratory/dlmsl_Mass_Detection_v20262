"""YOLO-MS modules for an Ultralytics YOLOv8 port.

This is a clean PyTorch/Ultralytics reimplementation of the architecture flow
released in the official MMYOLO-based YOLO-MS repository:

- C3-K3579-A234 backbone
- shared Global Query Layer (GQL) query across stages 2, 3, and 4
- MSBlock-based PAFPN aggregation
- scale-dependent 3x3 / 5x5 / 7x7 detection towers
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn

from ultralytics.nn.modules.block import SPPF
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.ops import make_divisible

__all__ = ("GQL", "MSBlock", "MSIndex", "YOLOMSBackbone", "YOLOMSDetect")


def _set_bn_cfg(module: nn.Module, momentum: float = 0.03, eps: float = 0.001) -> nn.Module:
    """Apply the normalization settings used by the official YOLO-MS config."""
    for layer in module.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.momentum = momentum
            layer.eps = eps
    return module


class GQL(nn.Module):
    """Global Query Layer used by YOLO-MS.

    The query is owned by YOLOMSBackbone and shared across the attention-enabled MSBlocks in backbone stages 2, 3, and
    4.
    """

    def __init__(self, dim: int, length: int = 3, size: int = 4) -> None:
        super().__init__()
        if dim % length != 0:
            raise ValueError(f"GQL dim={dim} must be divisible by query length={length}.")
        self.length = int(length)
        self.pool = nn.AdaptiveAvgPool2d((size, size))
        self.down_conv = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if query is None:
            raise ValueError("GQL requires the shared backbone query tensor.")

        batch, channels, _, _ = inputs.shape
        query_count, query_dim = query.shape
        pooled = self.down_conv(self.pool(inputs)).reshape(batch, -1, 1)

        if pooled.shape[1] != query_dim:
            raise RuntimeError(
                f"GQL query dimension mismatch: query has {query_dim}, pooled feature has {pooled.shape[1]}."
            )
        if channels % query_count != 0:
            raise RuntimeError(f"GQL input channels={channels} must be divisible by query count={query_count}.")

        weights = torch.matmul(query.unsqueeze(0).expand(batch, -1, -1), pooled).sigmoid()
        weights = weights.repeat(1, channels // query_count, 1).unsqueeze(-1)
        return inputs * weights


class MSBlockKxK1x1Layer(nn.Module):
    """Official YOLO-MS branch operator: grouped kxk convolution, then 1x1 projection."""

    def __init__(
        self,
        channels: int,
        expanded_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if channels <= 0 or expanded_channels <= 0:
            raise ValueError("channels and expanded_channels must be positive.")
        if expanded_channels % channels != 0:
            raise ValueError(
                f"expanded_channels={expanded_channels} must be divisible by channels={channels} "
                "because the official operator uses groups=channels."
            )

        self.large_kernel = Conv(
            channels,
            expanded_channels,
            k=kernel_size,
            s=1,
            g=channels,
        )
        self.project = Conv(expanded_channels, channels, k=1, s=1)
        _set_bn_cfg(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.large_kernel(x))


class MSBlock(nn.Module):
    """Multi-Scale Block following the official MMYOLO YOLO-MS flow.

    Args:
        c1: Input channels.
        c2: Output channels.
        kernel_sizes: One kernel per branch, e.g. (1, 5, 5).
        down_ratio: Internal channel ratio. Backbone uses 1.0; neck uses 0.5.
        mid_expand_ratio: Expansion used by each processed branch.
        layers_num: Number of repeated branch operators.
        start_branch_id: First branch that receives the preceding branch output.
        use_gql: Enable GQL. Used only inside the YOLO-MS backbone.
        gql_length: Number of global queries/channel groups.
        gql_size: Pooled spatial size.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_sizes: Sequence[int] = (1, 3, 3),
        down_ratio: float = 1.0,
        mid_expand_ratio: float = 2.0,
        layers_num: int = 1,
        start_branch_id: int = 2,
        use_gql: bool = False,
        gql_length: int = 3,
        gql_size: int = 4,
    ) -> None:
        super().__init__()

        kernels = tuple(int(k) for k in kernel_sizes)
        branch_count = len(kernels)
        if branch_count == 0:
            raise ValueError("MSBlock requires at least one branch.")
        if not 0 <= start_branch_id < branch_count:
            raise ValueError(f"start_branch_id={start_branch_id} is invalid for {branch_count} branches.")
        if layers_num < 1:
            raise ValueError("layers_num must be at least 1.")

        # Official formula:
        # self.in_channel = int((in_channel * len(kernel_sizes)) * down_ratio)
        internal_channels = int((c1 * branch_count) * float(down_ratio))
        if internal_channels < branch_count:
            raise ValueError(f"Internal channels={internal_channels} are too small for {branch_count} branches.")

        # Official configs use equal [1, 1, 1] branch ratios. Allocate any
        # rounding remainder to the final branch for robustness.
        base = internal_channels // branch_count
        branch_channels = [base] * branch_count
        branch_channels[-1] += internal_channels - sum(branch_channels)

        # Cross-branch addition requires equal adjacent channel counts.
        for i in range(start_branch_id, branch_count):
            if branch_channels[i] != branch_channels[i - 1]:
                raise ValueError(
                    f"Cross-branch addition requires equal adjacent branch channels; received {branch_channels}."
                )

        self.start_branch_id = int(start_branch_id)
        self.branch_channels = branch_channels
        self.in_projection = Conv(c1, internal_channels, k=1, s=1)

        branches: list[nn.Module] = []
        for kernel, channels in zip(kernels, branch_channels):
            if kernel == 1:
                branches.append(nn.Identity())
                continue

            expanded = int(channels * float(mid_expand_ratio))
            # The official "auto" grouping uses groups=channels.
            if expanded % channels != 0:
                expanded = channels * max(1, round(expanded / channels))

            branches.append(
                nn.Sequential(
                    *[
                        MSBlockKxK1x1Layer(
                            channels=channels,
                            expanded_channels=expanded,
                            kernel_size=kernel,
                        )
                        for _ in range(int(layers_num))
                    ]
                )
            )

        self.branches = nn.ModuleList(branches)
        self.attention = GQL(internal_channels, gql_length, gql_size) if use_gql else None
        self.out_projection = Conv(internal_channels, c2, k=1, s=1)
        _set_bn_cfg(self)

    def forward(self, x: torch.Tensor, query: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.in_projection(x)
        split_features = torch.split(projected, self.branch_channels, dim=1)
        outputs: list[torch.Tensor] = []

        for branch_id, (feature, branch) in enumerate(zip(split_features, self.branches)):
            if branch_id >= self.start_branch_id:
                feature = feature + outputs[branch_id - 1]
            outputs.append(branch(feature))

        fused = torch.cat(outputs, dim=1)
        if self.attention is not None:
            fused = self.attention(fused, query)
        return self.out_projection(fused)


class YOLOMSStage(nn.Module):
    """One YOLO-MS backbone stage: stride-2 Conv -> MSBlock -> optional SPPF."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int,
        layers_num: int,
        use_attention: bool,
        use_sppf: bool,
        query_length: int,
        query_size: int,
        mid_expand_ratio: float,
        start_branch_id: int,
    ) -> None:
        super().__init__()
        self.downsample = Conv(c1, c2, k=3, s=2)
        self.msblock = MSBlock(
            c2,
            c2,
            kernel_sizes=(1, kernel_size, kernel_size),
            down_ratio=1.0,
            mid_expand_ratio=mid_expand_ratio,
            layers_num=layers_num,
            start_branch_id=start_branch_id,
            use_gql=use_attention,
            gql_length=query_length,
            gql_size=query_size,
        )
        self.sppf = SPPF(c2, c2, k=5) if use_sppf else nn.Identity()
        _set_bn_cfg(self)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.msblock(x, query if self.msblock.attention is not None else None)
        return self.sppf(x)


class YOLOMSBackbone(nn.Module):
    """YOLOv8-MS C3-K3579-A234 backbone.

    Returns:
        Tuple containing P3/8, P4/16, and P5/32 feature maps.
        Keeping the complete backbone inside one Ultralytics module is deliberate: the MMYOLO implementation creates one
        trainable query and shares it among the GQL modules in stages 2, 3, and 4.
    """

    def __init__(
        self,
        c1: int,
        depth: float = 1 / 3,
        width: float = 0.16,
        last_stage_channels: int = 1280,
        query_length: int = 3,
        query_size: int = 4,
        msblock_layers_num: int = 3,
        mid_expand_ratio: float = 2.0,
        start_branch_id: int = 2,
    ) -> None:
        super().__init__()

        def scaled(channels: int) -> int:
            return make_divisible(channels * width, 8)

        c_stem = scaled(80)
        c_stage1 = scaled(160)
        c_stage2 = scaled(320)
        c_stage3 = scaled(640)
        c_stage4 = scaled(last_stage_channels)
        branch_layers = max(int(msblock_layers_num * depth), 1)

        self.out_channels = (c_stage2, c_stage3, c_stage4)
        self.query = nn.Parameter(torch.randn(query_length, query_size * query_size))

        self.stem = Conv(c1, c_stem, k=3, s=2)
        self.stage1 = YOLOMSStage(
            c_stem,
            c_stage1,
            kernel_size=3,
            layers_num=branch_layers,
            use_attention=False,
            use_sppf=False,
            query_length=query_length,
            query_size=query_size,
            mid_expand_ratio=mid_expand_ratio,
            start_branch_id=start_branch_id,
        )
        self.stage2 = YOLOMSStage(
            c_stage1,
            c_stage2,
            kernel_size=5,
            layers_num=branch_layers,
            use_attention=True,
            use_sppf=False,
            query_length=query_length,
            query_size=query_size,
            mid_expand_ratio=mid_expand_ratio,
            start_branch_id=start_branch_id,
        )
        self.stage3 = YOLOMSStage(
            c_stage2,
            c_stage3,
            kernel_size=7,
            layers_num=branch_layers,
            use_attention=True,
            use_sppf=False,
            query_length=query_length,
            query_size=query_size,
            mid_expand_ratio=mid_expand_ratio,
            start_branch_id=start_branch_id,
        )
        self.stage4 = YOLOMSStage(
            c_stage3,
            c_stage4,
            kernel_size=9,
            layers_num=branch_layers,
            use_attention=True,
            use_sppf=True,
            query_length=query_length,
            query_size=query_size,
            mid_expand_ratio=mid_expand_ratio,
            start_branch_id=start_branch_id,
        )
        _set_bn_cfg(self)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x, self.query)
        p3 = self.stage2(x, self.query)
        p4 = self.stage3(p3, self.query)
        p5 = self.stage4(p4, self.query)
        return p3, p4, p5


class MSIndex(nn.Module):
    """Select one feature map from the tuple returned by YOLOMSBackbone."""

    def __init__(self, c1: int, c2: int, index: int) -> None:
        super().__init__()
        self.index = int(index)
        self.out_channels = int(c2)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return x[self.index]


class MSHeadConv(nn.Module):
    """YOLO-MS head convolution.

    The official implementation uses groups=1 for 3x3 kernels and the configured groups (4) for the 5x5 and 7x7 kernels.
    """

    def __init__(self, c1: int, c2: int, kernel_size: int, groups: int = 4) -> None:
        super().__init__()
        actual_groups = 1 if int(kernel_size) == 3 else int(groups)
        if c1 % actual_groups != 0 or c2 % actual_groups != 0:
            raise ValueError(f"Head convolution channels ({c1}, {c2}) must be divisible by groups={actual_groups}.")
        self.conv = Conv(c1, c2, k=int(kernel_size), s=1, g=actual_groups)
        _set_bn_cfg(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class YOLOMSDetect(Detect):
    """Ultralytics Detect head with the scale-dependent YOLO-MS towers."""

    def __init__(
        self,
        nc: int = 80,
        groups: Sequence[int] = (4, 4, 4),
        reg_kernel_sizes: Sequence[Sequence[int]] = ((3, 3), (5, 3), (7, 3)),
        cls_kernel_sizes: Sequence[Sequence[int]] = ((3, 3), (5, 3), (7, 3)),
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        if len(ch) != 3:
            raise ValueError(f"YOLOMSDetect expects three feature levels, received ch={ch}.")
        if not (len(groups) == len(reg_kernel_sizes) == len(cls_kernel_sizes) == len(ch)):
            raise ValueError("groups and kernel-size lists must match the number of feature levels.")

        # Initialize all inherited Detect attributes and inference methods.
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)

        reg_channels = max(16, ch[0] // 4, self.reg_max * 4)
        cls_channels = max(ch[0], self.nc)  # matches MMYOLO YOLOv8MSHeadModule

        self.cv2 = nn.ModuleList(
            self._make_tower(
                input_channels=input_channels,
                hidden_channels=reg_channels,
                kernels=reg_kernel_sizes[level],
                groups=int(groups[level]),
                output_channels=4 * self.reg_max,
            )
            for level, input_channels in enumerate(ch)
        )
        self.cv3 = nn.ModuleList(
            self._make_tower(
                input_channels=input_channels,
                hidden_channels=cls_channels,
                kernels=cls_kernel_sizes[level],
                groups=int(groups[level]),
                output_channels=self.nc,
            )
            for level, input_channels in enumerate(ch)
        )

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

        _set_bn_cfg(self)

    @staticmethod
    def _make_tower(
        input_channels: int,
        hidden_channels: int,
        kernels: Sequence[int],
        groups: int,
        output_channels: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_channels = input_channels
        for kernel in kernels:
            layers.append(MSHeadConv(current_channels, hidden_channels, int(kernel), groups))
            current_channels = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, output_channels, kernel_size=1))
        return nn.Sequential(*layers)
