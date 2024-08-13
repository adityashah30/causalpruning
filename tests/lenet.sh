mkdir -p ../checkpoints
mkdir -p ../tensorboard

# Fashion-MNIST
python main.py --model=lenet --dataset=fashionmnist --no-prune

python main.py --model=lenet --dataset=fashionmnist --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

python main.py --model=lenet --dataset=fashionmnist --prune \
    --pruner=magpruner --mag_prune_amount=0.1

# CIFAR10
python main.py --model=lenet --dataset=cifar10 --no-prune

python main.py --model=lenet --dataset=cifar10 --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

python main.py --model=lenet --dataset=cifar10 --prune \
    --pruner=magpruner --mag_prune_amount=0.275


# TinyImageNet
python main.py --model=lenet --dataset=tinyimagenet --no-prune

python main.py --model=lenet --dataset=tinyimagenet --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

python main.py --model=lenet --dataset=tinyimagenet --prune \
    --pruner=magpruner --mag_prune_amount=0.275
