#!/bin/bash

git lfs pull

mkdir -p ./data
mkdir -p ./checkpoints
mkdir -p ./tensorboard

tar -xvzf ./data/tinyimagenet200.tar.gz -C ./data

uv run datasets_setup.py

