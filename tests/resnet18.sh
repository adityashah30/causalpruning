mkdir -p ../checkpoints
mkdir -p ../tensorboard

# Fashion-MNIST
uv run main.py --model=resnet18 --dataset=fashionmnist --no-prune

uv run main.py --model=resnet18 --dataset=fashionmnist --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-16

uv run main.py --model=resnet18 --dataset=fashionmnist --prune \
    --pruner=magpruner --mag_prune_amount=0.174

# CIFAR10
uv run main.py --model=resnet18 --dataset=cifar10 --no-prune

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-15

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
    --pruner=magpruner --mag_prune_amount=0.215


# TinyImageNet
uv run main.py --model=resnet18 --dataset=tinyimagenet --no-prune

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-15

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
    --pruner=magpruner --mag_prune_amount=0.14
