mkdir -p ../checkpoints
mkdir -p ../tensorboard

# Fashion-MNIST
python main.py --model=alexnet --dataset=fashionmnist --no-prune

python main.py --model=alexnet --dataset=fashionmnist --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-17

python main.py --model=alexnet --dataset=fashionmnist --prune \
    --pruner=magpruner --mag_prune_amount=0.275

# CIFAR10
python main.py --model=alexnet --dataset=cifar10 --no-prune

python main.py --model=alexnet --dataset=cifar10 --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-16

python main.py --model=alexnet --dataset=cifar10 --prune \
    --pruner=magpruner --mag_prune_amount=0.275


# TinyImageNet
python main.py --model=alexnet --dataset=tinyimagenet --no-prune

python main.py --model=alexnet --dataset=tinyimagenet --prune \
    --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-16

python main.py --model=alexnet --dataset=tinyimagenet --prune \
    --pruner=magpruner --mag_prune_amount=0.196
