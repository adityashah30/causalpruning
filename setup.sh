#!/bin/bash

git lfs pull

mkdir -p ./data
mkdir -p ./checkpoints
mkdir -p ./tensorboard

if [ -z "$(ls './data/tinyimagenet200')" ]; then
  tar -xvzf ./data/tinyimagenet200.tar.gz -C ./data
fi

uv run datasets_setup.py

