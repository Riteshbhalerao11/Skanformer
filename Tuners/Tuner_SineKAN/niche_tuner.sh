#!/bin/bash

#SBATCH ... args

module load pytorch/X

nvidia-smi

wandb agent Org/Project/Id --count 100
