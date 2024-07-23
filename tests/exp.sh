mkdir -p ../checkpoints
mkdir -p ../tensorboard


# Fashnion-MNIST

## LeNet
python main.py --model=lenet --dataset=fashionmnist --no-prune
python main.py --model=lenet --dataset=fashionmnist --prune \
    --pruner=causalpruner
python main.py --model=lenet --dataset=fashionmnist --prune --pruner=magpruner

## AlexNet
python main.py --model=alexnet --dataset=fashionmnist --no-prune
python main.py --model=alexnet --dataset=fashionmnist --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=fashionmnist --prune \
     --pruner=magpruner

## ResNet
python main.py --model=resnet --dataset=fashionmnist --no-prune
python main.py --model=resnet --dataset=fashionmnist --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=resnet --dataset=fashionmnist --prune \
     --pruner=magpruner


# CIFAR10

## LeNet
python main.py --model=lenet --dataset=cifar10 --no-prune
python main.py --model=lenet --dataset=cifar10 --prune --pruner=causalpruner
python main.py --model=lenet --dataset=cifar10 --prune --pruner=magpruner

## AlexNet
python main.py --model=alexnet --dataset=cifar10 --no-prune
python main.py --model=alexnet --dataset=cifar10 --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=cifar10 --prune \
     --pruner=magpruner

## ResNet18`
python main.py --model=resnet18 --dataset=cifar10 --no-prune
python main.py --model=resnet18 --dataset=cifar10 --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=resnet18 --dataset=cifar10 --prune \
     --pruner=magpruner


# Mini Imagenet

## LeNet
python main.py --model=lenet --dataset=miniimagenet --no-prune
python main.py --model=lenet --dataset=miniimagenet --prune --pruner=causalpruner
python main.py --model=lenet --dataset=miniimagenet --prune --pruner=magpruner

## AlexNet
python main.py --model=alexnet --dataset=miniimagenet --no-prune
python main.py --model=alexnet --dataset=miniimagenet --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=miniimagenet --prune \
     --pruner=magpruner

## ResNet18
python main.py --model=resnet18 --dataset=miniimagenet --no-prune
python main.py --model=resnet18 --dataset=miniimagenet --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=resnet18 --dataset=miniimagenet --prune \
     --pruner=magpruner
