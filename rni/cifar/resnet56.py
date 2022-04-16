import copy
import math
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
from rni.cifar.sigbn import SigBatchNorm2d


class PadChannels(nn.Module):
    def __init__(self, out_channels, stride=1):
        """
        :param stride: downsample last two dims
        :param out_channels: nr of channels output should have
        """

        super().__init__()
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, X):
        missing_channels = self.out_channels - X.shape[1]
        if missing_channels < 0:
            return X
        X = F.pad(
            X[:, :, :: self.stride, :: self.stride],
            (0, 0, 0, 0, missing_channels // 2, missing_channels // 2),
            "constant",
            0,
        )
        if X.shape[1] != self.out_channels:
            # assymetric case
            missing_channels = self.out_channels - X.shape[1]
            padding = torch.zeros(
                X.shape[0], missing_channels, X.shape[2], X.shape[3], device=X.device
            )
            X = torch.cat([X, padding], dim=1)

        return X

    def __repr__(self):
        return f"PadChannels(out_channels={self.out_channels}, stride={self.stride})"


class ResidualBlock(nn.Module):
    """
    Option A: parameter free zero padding on the channel dimension
    Structure: [3x3xC, 3x3xC]
    """

    def __init__(self, in_channels, out_channels, stride=1, bn_cls=nn.BatchNorm2d):
        super().__init__()

        skip = OrderedDict([])
        if (in_channels != out_channels) or (stride > 1):
            skip = OrderedDict(
                [("pad", PadChannels(out_channels=out_channels, stride=stride))]
            )
        self.skip = nn.Sequential(skip)

        self.residual = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            stride=stride,
                        ),
                    ),
                    ("bn1", bn_cls(out_channels)),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn2", bn_cls(out_channels)),
                ]
            )
        )
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.relu(self.residual(X) + self.skip(X))
        return out


class Resnet(nn.Sequential):
    def __init__(
        self,
        n_classes=10,
        bn_cls=nn.BatchNorm2d,
    ):
        layers = [
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                bn_cls(16),
            )
        ]
        prev_out = 16
        new_out = 16
        for stage in range(3):
            layers += self.create_stage(
                n_blocks=9,
                in_channels=prev_out,
                out_channels=new_out,
                stride=1 if stage == 0 else 2,
                bn_cls=bn_cls,
            )
            prev_out = new_out
            new_out = prev_out * 2

        layers += [
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(),
                nn.Linear(prev_out, n_classes),
            )
        ]
        super().__init__(*layers)

    @staticmethod
    def create_stage(
        n_blocks,
        in_channels,
        out_channels,
        stride,
        bn_cls=nn.BatchNorm2d,
    ) -> list:
        res = [
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bn_cls=bn_cls,
            )
        ]
        for _ in range(n_blocks - 1):
            res += [
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    bn_cls=bn_cls,
                )
            ]
        return res


def get_model(dataset="cifar10", bn="bn"):
    assert dataset in ["cifar10", "cifar100"]
    assert bn in ["bn", "sigbn"]

    bn_cls = nn.BatchNorm2d if bn == "bn" else SigBatchNorm2d
    n_classes = 10 if dataset == "cifar10" else 100

    model = Resnet(n_classes=n_classes, bn_cls=bn_cls)
    model = init_model(model)
    return model


def init_model(model):
    """
    Regular BN will be initialised to 0.5 weight and 0 bias.
    SigBN weights drawn from standard normal.
    BNs have to be initialised last to keep conv inits the same.
    """
    for m in [m for m in model.modules() if isinstance(m, nn.Conv2d)]:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))

    for m in [m for m in model.modules() if isinstance(m, nn.Linear)]:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    for m in [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]:
        if isinstance(m, SigBatchNorm2d):
            nn.init.normal_(m.weight, mean=0, std=1)
        else:
            nn.init.constant_(m.weight, val=0.5)
            nn.init.constant_(m.bias, val=0.0)
    return model


def get_prune_masks(
    model, percentage=0.5, min_filters_per_layer: int = 3, scope="global"
) -> list:
    """Returns a list of masks of which filters to keep"""
    assert scope in ["local", "global"]
    res = []
    bns = [block.residual.bn1 for block in model if isinstance(block, ResidualBlock)]

    weights = torch.cat([bn.weight.view(-1) for bn in bns]).detach().cpu()
    scores = weights.sigmoid() if isinstance(bns[0], SigBatchNorm2d) else weights.abs()
    global_th = np.quantile(scores, percentage)

    for bn in bns:
        scores = (
            (bn.weight.sigmoid() if isinstance(bn, SigBatchNorm2d) else bn.weight.abs())
            .detach()
            .cpu()
        )
        local_th = global_th if scope == "global" else np.quantile(scores, percentage)
        if (scores >= local_th).float().sum() < min_filters_per_layer:
            local_th = scores.sort()[0][-min_filters_per_layer - 1].item()
        mask = (scores > local_th).float().detach()  # filters to keep
        res.append(mask)
    return res


def prune_bn(bn, mask):
    old_bn = copy.deepcopy(bn)
    new_bn = old_bn.__class__(mask.sum().long().detach().cpu().item())
    new_bn.weight = nn.Parameter(old_bn.weight[mask.bool()])
    if old_bn.bias is not None:
        new_bn.bias = nn.Parameter(old_bn.bias[mask.bool()])
    new_bn.running_mean = old_bn.running_mean[mask.bool()]
    new_bn.running_var = old_bn.running_var[mask.bool()]
    return new_bn


def prune_conv(conv, mask=None, prev_mask=None):
    old_conv = copy.deepcopy(conv)
    if mask is None:
        mask = torch.ones(old_conv.out_channels)
    if prev_mask is None:
        prev_mask = torch.ones(old_conv.in_channels)

    new_conv = nn.Conv2d(
        prev_mask.sum().long(),
        mask.sum().long(),
        kernel_size=old_conv.kernel_size,
        bias=False,
        padding=old_conv.padding,
        stride=old_conv.stride,
    )
    new_conv.weight = nn.Parameter(
        old_conv.weight[mask.bool(), :, :, :][:, prev_mask.bool(), :, :]
    )
    return new_conv


def prune_hard(model, masks):
    """
    Pruning based only on b1 has implications on both conv1 and conv2
    """
    model = copy.deepcopy(model)
    blocks = [block for block in model if isinstance(block, ResidualBlock)]
    if len(blocks) != len(masks):
        raise ValueError("nr of residual blocks and masks doesnt match.")
    for block, mask in zip(blocks, masks):
        block.residual.conv1 = prune_conv(block.residual.conv1, mask)
        block.residual.bn1 = prune_bn(block.residual.bn1, mask)
        block.residual.conv2 = prune_conv(
            block.residual.conv2, mask=None, prev_mask=mask
        )
    return model
