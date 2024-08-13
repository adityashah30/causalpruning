mkdir -p ../checkpoints
mkdir -p ../tensorboard


# CIFAR10

## LeNet
python main.py --model=lenet --dataset=cifar10 --no-prune

python main.py --model=lenet --dataset=cifar10 --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

python main.py --model=lenet --dataset=cifar10 --prune \
    --pruner=magpruner --mag_prune_amount=0.275

## AlexNet
python main.py --model=alexnet --dataset=cifar10 --no-prune

python main.py --model=alexnet --dataset=cifar10 --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16 \
    --causal_pruner_l1_regularization_coeff=1e-16

python main.py --model=alexnet --dataset=cifar10 --prune \
     --pruner=magpruner --mag_pruner_amount=0.275


# Fashnion-MNIST

## LeNet
python main.py --model=lenet --dataset=fashionmnist --no-prune

python main.py --model=lenet --dataset=fashionmnist --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

python main.py --model=lenet --dataset=fashionmnist --prune \
    --pruner=magpruner --mag_prune_amount=0.1

## AlexNet
python main.py --model=alexnet --dataset=fashionmnist --no-prune

python main.py --model=alexnet --dataset=fashionmnist --prune \
    --pruner=causalpruner --causal_pruner_batch_size=16 \
    --causal_pruner_l1_regularization_coeff=1e-16

# Tiny ImageNet

## LeNet
python main.py --model=lenet --dataset=tinyimagenet --no-prune

python main.py --model=lenet --dataset=tinyimagenet --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

python main.py --model=lenet --dataset=tinyimagenet --prune \
    --pruner=magpruner --mag_pruner_amount=0.275
