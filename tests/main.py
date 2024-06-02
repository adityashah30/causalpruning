import argparse

from .context import causalpruner


def main(args):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal Pruning")
    parser.add_argument("--model", type=str,
                        default="lenet", help="Model name")
    parser.add_argument("--dataset", type=str,
                        default="cifar10", help="Dataset name")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="Batch size")
    parser.add_argument("--nepochs_pretrain", type=int, default=5,
                        help="Number of epochs for pretraining")
    parser.add_argument("--nepochs_prune", type=int, default=10,
                        help="Number of epochs for pruning")
    parser.add_argument("--nepochs_post_prune", type=int, default=100,
                        help="Number of epochs to run post training")
    parser.add_argument("--num_iter", type=int, default=10,
                        help="Number of iterations")
    parser.add_argument("--optimizer", type=str,
                        default="sgd", help="Optimizier", choices=["sgd"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum")
    parser.add_argument("--method", type=str, default="causalpruning",
                        help="Method for pruning", choices=["causalpruning", "magpruning"])
    parser.add_argument("--amount_mag", type=float, default=0.4, help="Amount")
    parser.add_argument("--test", action="store_true", help="Test the model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
