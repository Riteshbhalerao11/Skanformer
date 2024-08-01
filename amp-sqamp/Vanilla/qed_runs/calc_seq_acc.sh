#!/bin/bash

module load pytorch/2.1.0-cu12

nvidia-smi

python seq_acc.py \
    --project_name "TEST" \
    --run_name "TEST" \
    --model_name "transformer" \
    --root_dir "$SCRATCH/QED" \
    --data_dir "$SCRATCH/QED/data/QED_aug.csv" \
    --device "cuda" \
    --epochs 50 \
    --training_batch_size 64 \
    --test_batch_size 64 \
    --valid_batch_size 96 \
    --num_workers 32 \
    --embedding_size 512 \
    --hidden_dim 4096 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --warmup_ratio 0.1 \
    --dropout 0.1 \
    --src_max_len 288 \
    --tgt_max_len 288 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --save_freq 10 \
    --test_freq 5 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \

