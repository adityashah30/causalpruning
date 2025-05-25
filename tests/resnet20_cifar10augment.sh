#!/bin/bash

PRUNE_AMT_VALS=(0.1 0.3 0.5 0.7 0.9 0.95 0.98)
mkdir -p "../checkpoints/resnet20_cifar10"

for PRUNE_AMT in "${PRUNE_AMT_VALS[@]}"; do
    uv run main.py --model=resnet20 --dataset=cifar10 --prune \
    --train_lr=1e-3 --max_train_lr=0.1 --no-run_lrrt \
        --pruner=causalpruner --total_prune_amount=$PRUNE_AMT --num_prune_iterations=10 \
        --checkpoint_dir="../checkpoints/resnet20_cifar10/"
done