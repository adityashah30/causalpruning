#!/bin/bash

mkdir -p ../tensorboard
mkdir -p ../checkpoints/lenet_cifar10/

PRUNE_AMT_VALS=(0.3 0.5 0.7 0.8 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 0.995 0.999)

for i in {1..6}; do
  # Create repetition-specific directory
  mkdir -p "../checkpoints/lenet_cifar10/lenetcifar10_${i}"
  
  for PRUNE_AMT in "${PRUNE_AMT_VALS[@]}"; do
    uv run main.py --model=lenet --dataset=cifar10 --prune \
    --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
      --pruner=causalpruner --total_prune_amount=$PRUNE_AMT --num_prune_iterations=5 \
      --checkpoint_dir="../checkpoints/lenetcifar10_${i}"
  done

  for PRUNE_AMT in "${PRUNE_AMT_VALS[@]}"; do
    uv run main.py --model=lenet --dataset=cifar10 --prune \
    --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
    --pruner=magpruner --total_prune_amount=$PRUNE_AMT --num_prune_iterations=5 \
    --checkpoint_dir="../checkpoints/lenetcifar10_${i}"
  done
done