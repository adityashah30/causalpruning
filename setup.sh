#!/bin/bash

git lfs pull

mkdir -p ./data
mkdir -p ./checkpoints
mkdir -p ./tensorboard

tar -xvzf ./data/tinyimagenet200.tar.gz -C ./data

conda env create -f environment.yml
conda activate cpn

python datasets_setup.py
