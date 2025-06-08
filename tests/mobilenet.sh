mkdir -p ../checkpoints
mkdir -p ../tensorboard

# Causal Pruning
uv run main.py --model=mobilenet_trained --dataset=imagenet --prune \
  --batch_size=1024 --batch_size_while_pruning=1024 \
  --pruner=causalpruner --total_prune_amount=0.3 \
  --num_pre_prune_epochs=0 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=12 --num_prune_epochs=1 \
  --num_train_epochs=50

uv run main.py --model=mobilenet_trained --dataset=imagenet --prune \
  --batch_size=1024 --batch_size_while_pruning=1024 \
  --pruner=causalpruner --total_prune_amount=0.4 \
  --num_pre_prune_epochs=0 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=12 --num_prune_epochs=1 \
  --num_train_epochs=50

uv run main.py --model=mobilenet_trained --dataset=imagenet --prune \
  --batch_size=1024 --batch_size_while_pruning=1024 \
  --pruner=causalpruner --total_prune_amount=0.5 \
  --num_pre_prune_epochs=0 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=12 --num_prune_epochs=1 \
  --num_train_epochs=50

uv run main.py --model=mobilenet_trained --dataset=imagenet --prune \
  --batch_size=1024 --batch_size_while_pruning=1024 \
  --pruner=causalpruner --total_prune_amount=0.6 \
  --num_pre_prune_epochs=0 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=12 --num_prune_epochs=1 \
  --num_train_epochs=50

uv run main.py --model=mobilenet_trained --dataset=imagenet --prune \
  --batch_size=1024 --batch_size_while_pruning=1024 \
  --pruner=causalpruner --total_prune_amount=0.7 \
  --num_pre_prune_epochs=0 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=12 --num_prune_epochs=1 \
  --num_train_epochs=50

uv run main.py --model=mobilenet_trained --dataset=imagenet --prune \
  --batch_size=1024 --batch_size_while_pruning=1024 \
  --pruner=causalpruner --total_prune_amount=0.8 \
  --num_pre_prune_epochs=0 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=12 --num_prune_epochs=1 \
  --num_train_epochs=50

uv run main.py --model=mobilenet_trained --dataset=imagenet --prune \
  --batch_size=1024 --batch_size_while_pruning=1024 \
  --pruner=causalpruner --total_prune_amount=0.9 \
  --num_pre_prune_epochs=0 --num_prune_iterations=10 \
  --num_train_epochs_before_pruning=12 --num_prune_epochs=1 \
  --num_train_epochs=50
