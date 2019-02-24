#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1

#SBATCH --gres=gpu:1
#SBATCH --qos="debug"
#SBATCH -p "ug-gpu-small"
#SBATCH -t 02:00:00

source /etc/profile
module load cuda/8.0

stdbuf -oL env/bin/python ./transfer_resnet_cnn.py
