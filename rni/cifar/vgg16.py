import copy
import math
import torch
import numpy as np
from torch import nn
from torchvision.models import vgg16_bn
from rni.cifar.sigbn import SigBatchNorm2d


def get_model(dataset="cifar10", bn="bn"):
    assert dataset in ["cifar10", "cifar100"]
    assert bn in ["bn", "sigbn"]

    n_classes = 10 if dataset == "cifar10" else 100
    model = vgg16_bn(pretrained=False)
    model.features = model.features[:-1]  # dropping the last maxpool
    if bn == "sigbn":
        model.features = nn.Sequential(
            *[
                SigBatchNorm2d(l.num_features) if isinstance(l, nn.BatchNorm2d) else l
                for l in model.features
            ]
        )

    model.avgpool = nn.AvgPool2d(kernel_size=2)
    model.classifier = nn.Sequential(nn.Linear(512, n_classes))
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
    bns = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)]

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


def prune_hard(model, masks):
    """Create a new model with a pruned architecture"""
    model = copy.deepcopy(model.cpu())
    mask_ix = 0
    with torch.no_grad():
        for layer_ix in range(len(model.features)):
            layer = model.features[layer_ix]
            if not isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                continue
            mask_as_indices = masks[mask_ix].nonzero().view(-1)
            if isinstance(layer, nn.Conv2d):
                previous_mask_as_indices = (
                    torch.tensor(range(layer.weight.shape[1]))
                    if mask_ix == 0
                    else masks[mask_ix - 1].nonzero().view(-1)
                )
                new_conv = nn.Conv2d(
                    len(previous_mask_as_indices),
                    len(mask_as_indices),
                    kernel_size=layer.kernel_size,
                    padding=layer.padding,
                )
                new_conv.weight = nn.Parameter(
                    layer.weight[mask_as_indices, :, :, :][
                        :, previous_mask_as_indices, :, :
                    ]
                )
                new_conv.bias = (
                    None
                    if layer.bias is None
                    else nn.Parameter(layer.bias[mask_as_indices])
                )
                model.features[layer_ix] = new_conv
            if isinstance(layer, nn.BatchNorm2d):
                cls = (
                    SigBatchNorm2d
                    if isinstance(layer, SigBatchNorm2d)
                    else nn.BatchNorm2d
                )
                new_bn = cls(len(mask_as_indices))
                new_bn.weight = nn.Parameter(layer.weight[mask_as_indices])
                if layer.bias is not None:
                    new_bn.bias = nn.Parameter(layer.bias[mask_as_indices])
                new_bn.running_mean = layer.running_mean[mask_as_indices]
                new_bn.running_var = layer.running_var[mask_as_indices]
                model.features[layer_ix] = new_bn
                mask_ix += 1

        lin_mask = masks[-1].bool()
        old_lin = model.classifier[0]
        new_lin = nn.Linear(lin_mask.sum(), old_lin.out_features)
        new_lin.weight = nn.Parameter(old_lin.weight[:, lin_mask])
        new_lin.bias = nn.Parameter(old_lin.bias)
        model.classifier[0] = new_lin
    return model
