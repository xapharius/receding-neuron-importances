import os
import torch
import random
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader


DATASET_PATHS = {
    "cifar10": "/data/datasets/CIFAR10",
    "cifar100": "/data/datasets/CIFAR100",
}

NORMVALS = {
    "mean": {
        "cifar10": [0.4914, 0.4822, 0.4465],
        "cifar100": [0.5071, 0.4867, 0.4408],
    },
    "std": {
        "cifar10": [0.2023, 0.1994, 0.2010],
        "cifar100": [0.2675, 0.2565, 0.2761],
    },
}


def set_dataset_path(dataset, path):
    global DATASET_PATHS
    DATASET_PATHS[dataset] = path


def get_transforms(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORMVALS["mean"][dataset], NORMVALS["std"][dataset]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMVALS["mean"][dataset], NORMVALS["std"][dataset]),
        ]
    )
    return train_transform, test_transform


def get_loaders(dataset="cifar10", batch_size=64):
    assert dataset in ["cifar10", "cifar100"]
    train_transform, test_transform = get_transforms(dataset)
    ds_cls = CIFAR10 if dataset == "cifar10" else CIFAR100

    train_ds = ds_cls(
        root=DATASET_PATHS[dataset], transform=train_transform, train=True
    )
    test_ds = ds_cls(root=DATASET_PATHS[dataset], transform=test_transform, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    return train_loader, test_loader


def get_accuracy(model, loader, device="cuda:0") -> float:
    correct = 0
    n_obs = 0

    model.eval()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            out = model(X)
        correct += (out.argmax(dim=1) == y).float().sum()
        n_obs += len(X)
    model.train()

    accuracy = correct / n_obs
    return accuracy.item()


def set_torch_seed(seed: int = 0):
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False  # Needs to stay false for determinism
    torch.backends.cudnn.deterministic = True
