mkdir -p ../checkpoint
rm -rf ../checkpoint/*

mkdir -p ../tensorboard
rm -rf ../tensorboard/*

# MNIST

# LeNet
python main.py --model=lenet --dataset=mnist --no-prune
python main.py --model=lenet --dataset=mnist --prune --pruner=causalpruner
python main.py --model=lenet --dataset=mnist --prune --pruner=magpruner

# Fully Connected
python main.py --model=fullyconnected --dataset=mnist --no-prune
python main.py --model=fullyconnected --dataset=mnist --prune \
     --pruner=causalpruner
python main.py --model=fullyconnected --dataset=mnist --prune --pruner=magpruner


# CIFAR10

# LeNet
python main.py --model=lenet --dataset=cifar10 --no-prune
python main.py --model=lenet --dataset=cifar10 --prune --pruner=causalpruner
python main.py --model=lenet --dataset=cifar10 --prune --pruner=magpruner

# Fully Connected
python main.py --model=fullyconnected --dataset=cifar10 --no-prune
python main.py --model=fullyconnected --dataset=cifar10--pruner=causalpruner
python main.py --model=fullyconnected --dataset=cifar10 --prune \
    --pruner=magpruner

# AlexNet
python main.py --model=alexnet --dataset=cifar10 --lr=0.1  --no-prune
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 --prune \
     --pruner=magpruner
