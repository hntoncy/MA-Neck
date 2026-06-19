"""Core MA-Neck modules.

The implementation is distilled from the project code corresponding to
"MA-Neck: Mutual attention-based feature enhancement for lightweight
object detection" and uses the paper names:

- SAE: self-attention enhancement for single feature maps.
- MGCA: mutual graph channel attention for paired feature maps.
- MSACA: multi-scale attention concatenation used where PANet would concat.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import init


class CoAttention(nn.Module):
    """Co-attention over two channel-first flattened feature tensors.

    Inputs are expected as ``(B, C, N)`` and the returned tensors keep the
    same shape. This helper is used inside MGCA to refine paired attention
    maps before channel reweighting.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(channels, channels, 1, groups=channels, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        u1 = self.linear(x1).transpose(1, 2)
        affinity = u1 @ x2.flatten(2)
        attn_12 = self.softmax(affinity)
        attn_21 = self.softmax(affinity.transpose(1, 2))
        x2_attn = x1 @ attn_12
        x1_attn = x2 @ attn_21
        return x1_attn, x2_attn


class MGCA(nn.Module):
    """Mutual graph channel attention.

    The module learns channel-wise graph attention for two same-scale feature
    maps and returns the two reweighted maps. Both inputs must have the same
    channel count.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.a2 = nn.Parameter(torch.empty(channels, channels))
        init.constant_(self.a2, 1e-6)

        self.conv3 = nn.Conv1d(channels, channels, 1, bias=False)
        self.conv4 = nn.Conv1d(channels, channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.co_attn = CoAttention(channels)

        self.p = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.p_inv = nn.Sequential(
            nn.Conv1d(channels, channels, 1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        y1 = self.avg_pool(x1).flatten(2)
        y2 = self.avg_pool(x2).flatten(2)
        b, c, _ = y1.shape

        a1 = self.p(y1)
        d = self.p_inv(torch.diag_embed(a1.squeeze(2)))
        a2 = self.softmax(d) @ y2

        a1 = d.to(a1.device) * a1 + self.a2
        a2 = d.to(a2.device) * a2 + self.a2
        a1, a2 = self.co_attn(a1, a2)

        y1 = torch.matmul(a1.to(y1.dtype), y1)
        y2 = torch.matmul(a2.to(y2.dtype), y2)

        w1 = self.sigmoid(self.relu(self.conv3(y1)).view(b, c, 1, 1))
        w2 = self.sigmoid(self.relu(self.conv4(y2)).view(b, c, 1, 1))
        return x1 * w1, x2 * w2


class MSACA(nn.Module):
    """Mutual scale-aware concatenation for two feature maps.

    This replaces a plain concat at PANet fusion points. It first applies MGCA,
    projects both attention-refined maps with 1x1 convolutions, adds residual
    paths, then concatenates along the channel dimension.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mgca = MGCA(channels)
        self.cv1 = nn.Conv2d(channels, channels, 1)
        self.cv2 = nn.Conv2d(channels, channels, 1)

    def forward(self, xs: list[Tensor] | tuple[Tensor, Tensor]) -> Tensor:
        x1, x2 = xs
        u1, u2 = self.mgca(x1, x2)
        u1 = self.cv1(u1)
        u2 = self.cv2(u2)
        return torch.cat((x1 + u1, x2 + u2), dim=1)


class CoConv(nn.Module):
    """Channel-wise convolution used to merge paired co-attended maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
        self.proj = nn.Conv2d(channels, channels, 1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.01)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.gate(self.act(self.bn(self.conv(x))))
        return self.proj(x)


class SpatialCoAttention(nn.Module):
    """Spatial co-attention for two same-shaped feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=1)
        self.conv = CoConv(channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        b, c, h, w = x1.shape
        x1_flat = x1.flatten(2)
        x2_flat = x2.flatten(2)
        u1 = self.linear(x1_flat).transpose(1, 2)
        affinity = u1 @ x2_flat
        attn_12 = self.softmax(affinity)
        attn_21 = self.softmax(affinity.transpose(1, 2))
        x2_attn = (x1_flat @ attn_12).view(b, c, h, w)
        x1_attn = (x2_flat @ attn_21).view(b, c, h, w)
        return self.conv(x1_attn, x2_attn)


class SAE(nn.Module):
    """Self-attention enhancement for a single feature map.

    The original project named this module ``SCoA`` in YAML/logs. The paper
    refers to it as SAE, so this clean implementation exposes the paper name.
    ``SCoA`` is kept as an alias for drop-in YOLOv5 compatibility.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 11, padding=5, groups=channels),
        )
        self.coattn = SpatialCoAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        context = self.local_context(x)
        return self.coattn(context, x) + x


SCoA = SAE
