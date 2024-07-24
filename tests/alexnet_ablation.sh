mkdir -p ../checkpoints
mkdir -p ../tensorboard


num_iterations=(2 5 10 20)
num_prune_epochs=(2 5 10 20)
alphas=(1e-16 1e-17 1e-18 1e-19)

echo "Running Alexnet ablation studies on CIFAR10"

for iter in ${num_iterations[@]};
do
    for prune_epochs in ${num_prune_epochs[@]};
    do
        for alpha in ${alphas[@]};
        do
            echo "Num iterations: ${iter}; Num prune epochs: ${prune_epochs}; alpha: ${alpha}";
            python main.py --model=alexnet --dataset=cifar10 \
                --prune --pruner=causalpruner \
                --causal_pruner_batch_size=16 \
                --num_prune_iterations="${iter}" \
                --num_prune_epochs="${prune_epochs}" \
                --causal_pruner_l1_regularization_coeff="${alpha}";
        done
    done
done
