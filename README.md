# Causal Pruning

This project introduces a novel way of pruning deep models. Current state of 
the art is limited to local techniques like L1pruning (also called magnitude
pruning).

This method relies on the change in loss between different epochs to establish
a causal link between the weights and the loss value -- hence the name: causal
pruning.

## Setup

First setup the conda environment using `environment.yml` as follows.

`conda create -f environment.yml`

Next, activate the environment using

`conda activate cpn`

Now, make sure to download the datasets by running `setup.sh`. 
`setup.sh` does two things -- it untars `data/tinyimagenet200.tar.gz`, and it downloads
`FashionMNIST` and `CIFAR10` datasets using pytorch utils.

## Experiments

There are three main files in `tests` that contain the experiments -- one for each model 
architecture

1. `LeNet`
2. `AlexNet`
3. `ResNet18`

We run each model architecture for three datasets

1. `Fashion-MNIST` -- 60000 train images, 10000 test images, 10 classes, each image of size 28x28 pixels (grayscale)
2. `CIFAR10` -- 50000 train images, 10000 test images, 10 classes, each image of size 32x32 pixels (RGB)
3. `TinyImageNet` -- 100000 train images, 10000 test images, 200 classes, each image of size 64x64 (RGB)

Run 

* `tests/lenet.sh` for all `LeNet` experiments
* `tests/alexnet.sh` for all `AlexNet` experiments
* `tests/resnet.sh` for all `ResNet18` experiments
