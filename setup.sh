#!/bin/bash

mkdir -p ./data
mkdir -p ./checkpoints
mkdir -p ./tensorboard

tar -xvzf ./data/tinyimagenet200.tar.gz -C ./data/tinyimagenet200

python datasets_setup.py
