#!/bin/bash

#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --output="/pscratch/sd/r/ritesh11/ew_logs/tuner/slurm-%j.out"
#SBATCH --error="/pscratch/sd/r/ritesh11/ew_logs/tuner/slurm-%j.out"
#SBATCH --mail-user=ritesh.slurm@gmail.com
#SBATCH --mail-type=ALL


module load pytorch/2.1.0-cu12

nvidia-smi

wandb agent ves_ritesh/RSineKANformer_EW_2-2_niche_tuner/s0j1px4z --count 100