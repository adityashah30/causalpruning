rm -rf ../checkpoint/*
rm -rf ../tensorboard/*

python main.py --model=lenet --dataset=mnist --no-prune
python main.py --model=lenet --dataset=mnist
python main.py --model=lenet --dataset=cifar10 --no-prune
python main.py --model=lenet --dataset=cifar10
python main.py --model=alexnet --dataset=cifar10 --lr=0.1  --no-prune
python main.py --model=alexnet --dataset=cifar10 --lr=0.1 --causal_pruner_batch_size=16
python main.py --model=fullyconnected --dataset=cifar10 --no-prune
python main.py --model=fullyconnected --dataset=cifar10
