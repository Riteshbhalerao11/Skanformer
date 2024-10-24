#!/bin/bash

#SBATCH ... args


module load pytorch/X

nvidia-smi

wandb agent org/Project/id --count 100
