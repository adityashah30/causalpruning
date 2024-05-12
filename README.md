# Causal Pruning

This project introduces a novel way of pruning deep models. Current state of 
the art is limited to local techniques like L1pruning (also called magnitude
pruning).

This method relies on the change in loss between different epochs to establish
a causal link between the weights and the loss value -- hence the name: causal
pruning.
