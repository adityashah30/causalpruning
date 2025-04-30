mkdir -p ../checkpoints
mkdir -p ../tensorboard

# CIFAR10

# No pruning
uv run main.py --model=lenet --dataset=cifar10 --no-prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt

# Causal Pruning
uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.1

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.3

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.5

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.7

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.9

# Iterative Magnitude Pruning
uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.1

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.3

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.5

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.7

uv run main.py --model=lenet --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.9
