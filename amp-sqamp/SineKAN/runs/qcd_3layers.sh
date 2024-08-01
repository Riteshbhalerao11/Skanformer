#!/bin/bash

#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --output="/pscratch/sd/r/ritesh11/qcd_logs/slurm-%j.out"
#SBATCH --error="/pscratch/sd/r/ritesh11/qcd_logs/slurm-%j.out"
#SBATCH --mail-user=ritesh.slurm@gmail.com
#SBATCH --mail-type=ALL


module load pytorch/2.1.0-cu12

nvidia-smi

srun torchrun --standalone --nproc_per_node 4 main.py \
    --project_name "SineKANformer_QCD" \
    --run_name "run_aug_3layers_$SLURM_JOB_ID" \
    --model_name "sinekanformer" \
    --root_dir "$SCRATCH/QCD/SineKAN" \
    --data_dir "$SCRATCH/QCD/data/QCD_small_aug" \
    --device "cuda" \
    --epochs 50 \
    --training_batch_size 36 \
    --test_batch_size 36 \
    --valid_batch_size 36 \
    --num_workers 32 \
    --embedding_size 512 \
    --ff_dims 2048 \
    --nhead 8 \
    --num_layers 3 \
    --warmup_ratio 0 \
    --clip_grad_norm 5 \
    --weight_decay 0 \
    --dropout 0.1 \
    --src_max_len 896 \
    --tgt_max_len 896 \
    --curr_epoch 0 \
    --optimizer_lr 5e-6 \
    --is_constant_lr \
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --resume_best True \
    --save_freq 10 \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
