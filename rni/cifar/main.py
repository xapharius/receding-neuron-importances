import copy
import torch
from tqdm import tqdm
from torch import nn, optim
from collections import defaultdict

from rni.cifar import utils, vgg16, resnet56
from rni.cifar.sigbn import SigBatchNorm2d


def get_optimiser(model, lr=1e-1, weight_decay=1e-4):
    """
    If using SigBN increase lr and disable weight decay.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_params = [param for layer in bns for param in layer.parameters()]
    other_params = [
        param
        for layer in model.modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
        for param in layer.parameters()
    ]

    using_sigbn = isinstance(bns[0], SigBatchNorm2d)
    bn_lr = 10 * lr if using_sigbn else lr
    bn_wd = 0 if using_sigbn else weight_decay

    optimiser = optim.SGD(
        [
            {
                "params": other_params,
                "lr": lr,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": weight_decay,
            },
            {
                "params": bn_params,
                "lr": bn_lr,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": bn_wd,
            },
        ]
    )
    return optimiser


def get_bn_weights(model, cat=True):
    res = [
        bn.weight.view(-1) for bn in model.modules() if isinstance(bn, nn.BatchNorm2d)
    ]
    if cat:
        res = torch.cat(res)
    return res


def rni_loss(weights, b=3):
    """
    This is it.
    """
    weights = (weights + b).sigmoid()
    loss = weights * (1 - weights.log())
    return loss.sum()


def get_loss_func(method="rni", b=3, reg_weight=1e-4):
    """
    Return callable whose parameters have already been set.
    """
    if method is None or method == "ucs" or reg_weight == 0:
        return None
    elif method == "l1":
        return lambda model: reg_weight * get_bn_weights(model).abs().sum()
    elif method == "rni":
        return lambda model: reg_weight * rni_loss(get_bn_weights(model), b=b)
    assert ValueError("Not a valid method")


def _train(
    model: nn.Module,
    train_loader: utils.DataLoader,
    test_loader: utils.DataLoader,
    reg_func=None,
    epochs=160,
    lr=0.1,
    seed=0,
    device="cuda:0",
):
    utils.set_torch_seed(seed)
    model = model.to(device)

    logs = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    optimiser = get_optimiser(model, lr)
    milestones = [int(epochs * 0.5), int(epochs * 0.75)]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=milestones, gamma=0.1
    )

    best_model = None
    best_acc = -1
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            if reg_func is not None:
                loss += reg_func(model)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        lr_scheduler.step()

        logs["test_acc"].append(utils.get_accuracy(model, test_loader, device=device))
        pbar.set_description("Test Acc: {:.4f}%".format(logs["test_acc"][-1] * 100))

        if logs["test_acc"][-1] > best_acc:
            best_acc = logs["test_acc"][-1]
            best_model = copy.deepcopy(model).cpu()
            logs["best_epoch"] = epoch

    return best_model, logs


def train(
    dataset="cifar10",
    arch="vgg16",
    method="rni",
    reg_weight=1e-4,
    b=3,
    epochs=160,
    lr=0.1,
    batch_size=64,
    seed=0,
    device="cuda:0",
) -> (nn.Module, dict):
    """
    Returns best model and training logs
    :param dataset: "cifar10" or "cifar100"
    :param arch: "vgg16" or "resnet56"
    :param method: "rni", "ucs", "l1"
    :param reg_weight: strength of sparsity regularisation, will be turned off for ucs
    :param b: "shift" hyper-parameter for rni, won't do anything for l1 or ucs
    :param lr: scheduled to be divided by 10 at 50% and 75% of epochs.
    """
    assert dataset in ["cifar10", "cifar100"]
    assert method in ["rni", "l1", "ucs"]
    assert arch in ["vgg16", "resnet56"]

    bn = "sigbn" if method == "rni" else "bn"

    train_loader, test_loader = utils.get_loaders(dataset, batch_size)
    model = eval(arch).get_model(dataset=dataset, bn=bn)
    reg_func = get_loss_func(method=method, reg_weight=reg_weight, b=b)

    model, logs = _train(
        model,
        train_loader,
        test_loader,
        reg_func,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    return model, logs


def prune_finetune(
    model: nn.Module,
    dataset="cifar10",
    method="rni",
    prune_pc=0.5,
    epochs=160,
    lr=0.1,
    batch_size=64,
    seed=0,
    device="cuda:0",
):
    """
    Prunes given model and finetunes it. Pruning strategy (local/global) depends on method.
    :param model: network returned by "train"
    :param dataset: "cifar10" or "cifar100"
    :param arch: "vgg16" or "resnet56"
    :param method: "rni", "ucs", "l1"
    :param prune_pc: percentage or filters to prune
    :param lr: scheduled to be divided by 10 at 50% and 75% of epochs.
    """
    assert dataset in ["cifar10", "cifar100"]
    assert method in ["rni", "l1", "ucs"]

    prune_scope = "local" if method == "ucs" else "global"
    arch = vgg16 if isinstance(model, vgg16.get_model().__class__) else resnet56
    train_loader, test_loader = utils.get_loaders(dataset, batch_size)

    prune_masks = arch.get_prune_masks(model, percentage=prune_pc, scope=prune_scope)
    pruned_model = arch.prune_hard(model, prune_masks)
    finetuned_model, logs = _train(
        pruned_model,
        train_loader,
        test_loader,
        reg_func=None,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    return finetuned_model, logs
