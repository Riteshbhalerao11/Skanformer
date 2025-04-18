#!/bin/bash

#SBATCH .. args


module load pytorch

nvidia-smi

srun torchrun --standalone --nproc_per_node 4 -m modeling.vanilla.seq_acc \
    --project_name "Dummy_Transformer_Project" \
    --run_name "dummy_run" \
    --model_name "dummy_transformer" \
    --root_dir "transformer_checkpoints" \
    --data_dir "transformer_data" \
    --device "cuda" \
    --epochs 50 \
    --training_batch_size 64 \
    --test_batch_size 64 \
    --valid_batch_size 64 \
    --num_workers 32 \
    --embedding_size 512 \
    --hidden_dim 8192 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --warmup_ratio 0 \
    --dropout 0.1 \
    --weight_decay 1e-3 \
    --src_max_len 302 \
    --tgt_max_len 302 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --train_shuffle True\
    --pin_memory True \
    --world_size 2 \
    --save_freq 9 \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
