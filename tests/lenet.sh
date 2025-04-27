mkdir -p ../checkpoints
mkdir -p ../tensorboard

# Fashion-MNIST
# uv run main.py --model=lenet --dataset=fashionmnist --no-prune

# uv run main.py --model=lenet --dataset=fashionmnist --prune \
#     --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

# uv run main.py --model=lenet --dataset=fashionmnist --prune \
#     --pruner=magpruner --mag_pruner_amount=0.1

# CIFAR10
uv run main.py --model=lenet --dataset=cifar10 --no-prune \
  --train_optimizer=sgd_momentum \
  --train_convergence_loss_tolerance=-100

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-12 \
  --train_optimizer=sgd_momentum \
  --train_convergence_loss_tolerance=-100

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --pruner=magpruner --mag_pruner_amount=0.16 \
  --train_optimizer=sgd_momentum \
  --train_convergence_loss_tolerance=-100

# TinyImageNet
# uv run main.py --model=lenet --dataset=tinyimagenet --no-prune

# uv run main.py --model=lenet --dataset=tinyimagenet --prune \
#     --pruner=causalpruner --causal_pruner_l1_regularization_coeff=1e-14

# uv run main.py --model=lenet --dataset=tinyimagenet --prune \
#     --pruner=magpruner --mag_pruner_amount=0.275
