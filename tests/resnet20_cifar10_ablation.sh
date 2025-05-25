#!/bin/bash

# Create directories if they don't exist
mkdir -p ../checkpoints_ablation

mkdir -p ../checkpoints_ablation/prune_amount_0.98
mkdir -p ../tensorboard

# Define the variables
N_PRE_VALS=(0 10 20 30)
N_ITER_VALS=(1 5 10 20)
N_EPOCH_PRUNE_VALS=(5 10 20)

# Run all combinations
for N_PRE in "${N_PRE_VALS[@]}"; do
  mkdir -p "../checkpoints_ablation/prune_amount_0.98/N_PRE_${N_PRE}"
  for N_ITER in "${N_ITER_VALS[@]}"; do
    for N_EPOCH_PRUNE in "${N_EPOCH_PRUNE_VALS[@]}"; do
      echo "Running with N_PRE=$N_PRE, N_ITER=$N_ITER, N_EPOCH_PRUNE=$N_EPOCH_PRUNE"      
      
      uv run main.py --model=resnet20 --dataset=cifar10 --prune \
        --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
        --pruner=causalpruner --total_prune_amount=0.98 \
        --num_prune_iterations=$N_ITER \
        --num_pre_prune_epochs=$N_PRE \
        --num_prune_epochs=$N_EPOCH_PRUNE \
        --checkpoint_dir="../checkpoints_ablation/prune_amount_0.98/N_PRE_${N_PRE}"
    done
  done
done

mkdir -p ../checkpoints_ablation/prune_amount_0.9
mkdir -p ../tensorboard

# Define the variables
N_PRE_VALS=(0 10 20 30)
N_ITER_VALS=(1 5 10 20)
N_EPOCH_PRUNE_VALS=(5 10 20)

# Run all combinations
for N_PRE in "${N_PRE_VALS[@]}"; do
  mkdir -p "../checkpoints_ablation/prune_amount_0.9/N_PRE_${N_PRE}"
  for N_ITER in "${N_ITER_VALS[@]}"; do
    for N_EPOCH_PRUNE in "${N_EPOCH_PRUNE_VALS[@]}"; do
      echo "Running with N_PRE=$N_PRE, N_ITER=$N_ITER, N_EPOCH_PRUNE=$N_EPOCH_PRUNE"      
      
      uv run main.py --model=resnet20 --dataset=cifar10 --prune \
        --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
        --pruner=causalpruner --total_prune_amount=0.9 \
        --num_prune_iterations=$N_ITER \
        --num_pre_prune_epochs=$N_PRE \
        --num_prune_epochs=$N_EPOCH_PRUNE \
        --checkpoint_dir="../checkpoints_ablation/prune_amount_0.9/N_PRE_${N_PRE}"
    done
  done
done