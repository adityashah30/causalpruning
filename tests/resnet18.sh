mkdir -p ../checkpoints
mkdir -p ../tensorboard

# CIFAR10

# No pruning
uv run main.py --model=resnet18 --dataset=cifar10 --no-prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt

# Causal Pruning
uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.1

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.3

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.5

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.7

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.9

# Iterative Magnitude Pruning
uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.1

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.3

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.5

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.7

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.9

# Tinyimagenet

# No pruning
uv run main.py --model=resnet18 --dataset=tinyimagenet --no-prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt

# Causal Pruning
uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.1

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.3

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.5

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.7

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.9

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.95

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=causalpruner --total_prune_amount=0.98

# Iterative Magnitude Pruning
uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.1

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.3

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.5

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.7

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.9

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.95

uv run main.py --model=resnet18 --dataset=tinyimagenet --prune \
  --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
  --pruner=magpruner --total_prune_amount=0.98
