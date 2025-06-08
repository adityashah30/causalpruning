mkdir -p ../checkpoints
mkdir -p ../tensorboard

# No pruning
uv run main.py --model=resnet18 --dataset=cifar10 --no-prune \
  --num_train_epochs=410

# Causal Pruning
uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --pruner=causalpruner --total_prune_amount=0.3 \
  --num_pre_prune_epochs=10 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=10 --num_prune_epochs=10 \
  --num_train_epochs=300

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --pruner=causalpruner --total_prune_amount=0.4 \
  --num_pre_prune_epochs=10 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=10 --num_prune_epochs=10 \
  --num_train_epochs=300

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --pruner=causalpruner --total_prune_amount=0.5 \
  --num_pre_prune_epochs=10 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=10 --num_prune_epochs=10 \
  --num_train_epochs=300

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --pruner=causalpruner --total_prune_amount=0.6 \
  --num_pre_prune_epochs=10 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=10 --num_prune_epochs=10 \
  --num_train_epochs=300

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --pruner=causalpruner --total_prune_amount=0.7 \
  --num_pre_prune_epochs=10 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=10 --num_prune_epochs=10 \
  --num_train_epochs=300

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --pruner=causalpruner --total_prune_amount=0.8 \
  --num_pre_prune_epochs=10 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=10 --num_prune_epochs=10 \
  --num_train_epochs=300

uv run main.py --model=resnet18 --dataset=cifar10 --prune \
  --pruner=causalpruner --total_prune_amount=0.9 \
  --num_pre_prune_epochs=10 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=10 --num_prune_epochs=10 \
  --num_train_epochs=300

# Iterative Magnitude Pruning
# uv run main.py --model=resnet18 --dataset=cifar10 --prune \
#   --pruner=magpruner --total_prune_amount=0.1 \
#   --num_pre_prune_epochs=10 --num_prune_iterations=10 \
#   --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
#   --num_train_epochs=300
#
# uv run main.py --model=resnet18 --dataset=cifar10 --prune \
#   --pruner=magpruner --total_prune_amount=0.3 \
#   --num_pre_prune_epochs=10 --num_prune_iterations=10 \
#   --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
#   --num_train_epochs=300
#
# uv run main.py --model=resnet18 --dataset=cifar10 --prune \
#   --pruner=magpruner --total_prune_amount=0.5 \
#   --num_pre_prune_epochs=10 --num_prune_iterations=10 \
#   --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
#   --num_train_epochs=300
#
# uv run main.py --model=resnet18 --dataset=cifar10 --prune \
#   --pruner=magpruner --total_prune_amount=0.7 \
#   --num_pre_prune_epochs=10 --num_prune_iterations=10 \
#   --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
#   --num_train_epochs=300
#
# uv run main.py --model=resnet18 --dataset=cifar10 --prune \
#   --pruner=magpruner --total_prune_amount=0.9 \
#   --num_pre_prune_epochs=10 --num_prune_iterations=10 \
#   --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
#   --num_train_epochs=300
