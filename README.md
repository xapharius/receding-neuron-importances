# Receding Neuron Importances for Structured Pruning
Official repository of the paper: Receding Neuron Importances for Structured Pruning, by Mihai Suteu and Yike Guo 2022 https://arxiv.org/abs/2204.06404

## Abstract
Structured pruning efficiently compresses networks by identifying and removing unimportant neurons. While this can be elegantly achieved by applying sparsity-inducing regularisation on BatchNorm parameters, an L1 penalty would shrink all scaling factors rather than just those of superfluous neurons. To tackle this issue, we introduce a simple BatchNorm variation with bounded scaling parameters, based on which we design a novel regularisation term that suppresses only neurons with low importance. Under our method, the weights of unnecessary neurons effectively recede, producing a polarised bimodal distribution of importances. We show that neural networks trained this way can be pruned to a larger extent and with less deterioration. We one-shot prune VGG and ResNet architectures at different ratios on CIFAR and ImagenNet datasets. In the case of VGG-style networks, our method significantly outperforms existing approaches particularly under a severe pruning regime.

## Experiments and Usage
- Datasets: CIFAR10, CIFAR100
- Methods: RNI (ours), UCS (baseline), L1 (Slimming)
- Architectures: VGG-16, ResNet-56
- In the hope to make things more user-friendly all training, pruning and fine-tuning can be run from a Ipython Notebook using only two functions.
- Both Training and pruning-finetuning return the best model and training logs.
- Sample usage over CIFAR scenarios can be found in Results.ipynb

### Training
- Trains a new full/baseline model from scratch, which can be pruned later.
```
model, logs = train(
    dataset="cifar10",  # "cifar10", "cifar100"
    arch="vgg16",       # "vgg16", "resnet56"
    method="rni",       # "rni", "ucs", "l1"
    reg_weight=1e-4,    # strength of sparsity regularisation, will be turned off for ucs
    b=3,                # "shift" hyper-parameter for rni, won't do anything for l1 or ucs
    epochs=160,
    lr=0.1,             # scheduled to be divided by 10 at 50% and 75% of epochs.
    batch_size=64,
    seed=0,
    device="cuda:0",
)
```

### Pruning and Finetuning
- Prunes and finetunes a given model.
- Pruning strategy is dictated by method used. local: ucs, global: l1, rni

```
model, logs = prune_finetune(
    model,
    dataset="cifar10",  
    method="rni",       
    prune_pc=0.5,       # percentage of filters to prune
    epochs=160,
    lr=0.1,             
    batch_size=64,
    seed=0,
    device="cuda:0",
)
```


# Implementation Details
- Written in Pytorch
- Very simple implementation using a single underlying trainer function for all methods, architectures and datasets.
- Pruning logic is defined in the model modules, as it depends on the architecture.
- Sigmoid BatchNorm implementation is based on Pytorch's BatchNorm code.