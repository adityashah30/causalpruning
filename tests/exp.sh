mkdir -p ../checkpoints
mkdir -p ../tensorboard


# CIFAR10

## LeNet
python main.py --model=lenet --dataset=cifar10 --no-prune
python main.py --model=lenet --dataset=cifar10 --momentum=0.9 --no-prune
python main.py --model=lenet --dataset=cifar10 --prune --pruner=causalpruner
python main.py --model=lenet --dataset=cifar10 --momentum=0.9 \
    --prune --pruner=causalpruner
python main.py --model=lenet --dataset=cifar10 --prune --pruner=magpruner
python main.py --model=lenet --dataset=cifar10 --momentum=0.9 \
    --prune --pruner=magpruner

## Fully Connected
python main.py --model=fullyconnected --dataset=cifar10 --no-prune
python main.py --model=fullyconnected --dataset=cifar10 --momentum=0.9 \
    --no-prune
python main.py --model=fullyconnected --dataset=cifar10 --pruner=causalpruner
python main.py --model=fullyconnected --dataset=cifar10 --momentum=0.9 \
    --pruner=causalpruner
python main.py --model=fullyconnected --dataset=cifar10 --prune \
    --pruner=magpruner
python main.py --model=fullyconnected --dataset=cifar10 --momentum=0.9 \
    --prune --pruner=magpruner

## AlexNet
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 --no-prune
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 \
    --momentum=0.9 --no-prune
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 \
    --momentum=0.9 --prune --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 --prune \
     --pruner=magpruner
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 \
    --momentum=0.9 --prune --pruner=magpruner


# Fashion-MNIST

## LeNet
python main.py --model=lenet --dataset=fashionmnist --no-prune
python main.py --model=lenet --dataset=fashionmnist --momentum=0.9 --no-prune
python main.py --model=lenet --dataset=fashionmnist --prune \
    --pruner=causalpruner
python main.py --model=lenet --dataset=fashionmnist --momentum=0.9 --prune \
    --pruner=causalpruner
python main.py --model=lenet --dataset=fashionmnist --prune --pruner=magpruner
python main.py --model=lenet --dataset=fashionmnist --momentum=0.9 --prune \
    --pruner=magpruner

## Fully Connected
python main.py --model=fullyconnected --dataset=fashionmnist --no-prune
python main.py --model=fullyconnected --dataset=fashionmnist --momentum=0.9 \
    --no-prune
python main.py --model=fullyconnected --dataset=fashionmnist --prune \
    --pruner=causalpruner
python main.py --model=fullyconnected --dataset=fashionmnist \
    --momentum=0.9 --prune --pruner=causalpruner
python main.py --model=fullyconnected --dataset=fashionmnist --prune \
    --pruner=magpruner
python main.py --model=fullyconnected --dataset=fashionmnist \
    --momentum=0.9 --prune--prune --pruner=magpruner

## AlexNet
python main.py --model=alexnet --dataset=fashionmnist --no-prune
python main.py --model=alexnet --dataset=fashionmnist --momentum=0.9 --no-prune
python main.py --model=alexnet --dataset=fashionmnist --lr=0.1 --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=fashionmnist --lr=0.1 --momentum=0.9 \
    --prune --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=fashionmnist --lr=0.1 --prune \
    --pruner=magpruner
python main.py --model=alexnet --dataset=fashionmnist --lr=0.1 --momentum=0.9 \
    --prune --pruner=magpruner