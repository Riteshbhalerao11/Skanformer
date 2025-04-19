import argparse
import random
from datetime import timedelta
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .config import TransformerConfig
from .model import Model

from Modeling.constants import BOS_IDX, EOS_IDX, PAD_IDX, SPECIAL_SYMBOLS, UNK_IDX
from Modeling.tokenizer import Tokenizer


def create_tokenizer(df, config, index_pool_size, momentum_pool_size):
    """Creates a tokenizer and builds source and target vocabularies.

    Args:
        df (pd.DataFrame): Dataset containing text samples.
        config (object): Configuration object.
        index_pool_size (int): Size of the index pool.
        momentum_pool_size (int): Size of the momentum pool.

    Returns:
        tuple: Tokenizer object, source vocab, target vocab, source index-to-string, target index-to-string.
    """
    tokenizer = Tokenizer(df, index_pool_size, momentum_pool_size, SPECIAL_SYMBOLS, UNK_IDX, config.to_replace)
    src_vocab = tokenizer.build_src_vocab(config.seed)
    src_itos = {v: k for k, v in src_vocab.get_stoi().items()}
    tgt_vocab = tokenizer.build_tgt_vocab()
    tgt_itos = {v: k for k, v in tgt_vocab.get_stoi().items()}
    return tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos


def init_distributed_mode(config):
    """Initializes PyTorch distributed training mode."""
    dist.init_process_group(backend=config.backend, timeout=timedelta(minutes=30))


def generate_eqn_mask(n: int, device: torch.device) -> torch.Tensor:
    """Generates an autoregressive mask for target equations.

    Args:
        n (int): Sequence length.
        device (torch.device): Device to place the mask on.

    Returns:
        torch.Tensor: Upper triangular causal mask.
    """
    mask = (torch.triu(torch.ones((n, n), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    return mask


def create_mask(src: torch.Tensor, tgt: torch.Tensor, device: torch.device) -> tuple:
    """Creates source/target masks and padding masks for Transformer.

    Args:
        src (torch.Tensor): Source tensor (S, B).
        tgt (torch.Tensor): Target tensor (T, B).
        device (torch.device): Computation device.

    Returns:
        tuple: (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_eqn_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def generate_unique_random_integers(x, start=0, end=3000):
    """Generates `x` unique integers from the range [start, end].

    Args:
        x (int): Number of integers.
        start (int): Range start.
        end (int): Range end.

    Returns:
        list: List of unique random integers.

    Raises:
        ValueError: If x exceeds the number of unique values in the range.
    """
    if x > (end - start + 1):
        raise ValueError("x cannot be greater than the range of unique values available")
    return random.sample(range(start, end), x)


def decode_sequence(src: List[int], itos):
    """Decodes a list of token indices into a string.

    Args:
        src (List[int]): Token indices.
        itos (dict): Index-to-token mapping.

    Returns:
        str: Decoded string.
    """
    return ''.join(itos[y] for y in src if y not in {PAD_IDX, BOS_IDX, EOS_IDX})


def collate_fn(batch: list) -> tuple:
    """Collates a batch of (src, tgt) pairs into padded tensors.

    Args:
        batch (list): List of (src, tgt) tuples.

    Returns:
        tuple: Padded src and tgt tensors.
    """
    src_batch = [src for src, _ in batch]
    tgt_batch = [tgt for _, tgt in batch]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def calculate_line_params(point1, point2):
    """Calculates slope and intercept of a line from two points.

    Args:
        point1 (tuple): (x1, y1)
        point2 (tuple): (x2, y2)

    Returns:
        tuple: (slope, intercept)

    Raises:
        ValueError: If x1 == x2 (vertical line).
    """
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        raise ValueError("The x coordinates must differ to define a valid line.")

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def get_model(config):
    """Instantiates and initializes the Transformer model.

    Args:
        config (object): Configuration containing model hyperparameters.

    Returns:
        Model: Initialized Transformer model.
    """
    model = Model(
        config.num_encoder_layers,
        config.num_decoder_layers,
        config.embedding_size,
        config.nhead,
        config.src_voc_size,
        config.tgt_voc_size,
        config.hidden_dim,
        config.dropout
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model



def parse_args():
    """Parses command-line arguments for Transformer training configuration."""

    parser = argparse.ArgumentParser(description="Transformer Training Configuration")

    # Project & model details
    parser.add_argument("--project_name", type=str, required=True, help="Project name")
    parser.add_argument("--run_name", type=str, required=True, help="Run name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    # Directory paths
    parser.add_argument("--root_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")

    # Device & training setup
    parser.add_argument("--device", type=str, default="cuda", help='Device: "cuda" or "cpu"')
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs")
    parser.add_argument("--training_batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, required=True, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of data loader workers")

    # Transformer architecture
    parser.add_argument("--embedding_size", type=int, required=True, help="Word embedding dimension")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden layer dimension")
    parser.add_argument("--nhead", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, required=True, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, required=True, help="Number of decoder layers")

    # Optimization settings
    parser.add_argument("--warmup_ratio", type=float, required=True, help="Warmup ratio for learning rate")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay (AdamW)")
    parser.add_argument("--optimizer_lr", type=float, required=True, help="Optimizer learning rate")
    parser.add_argument("--is_constant_lr", action="store_true", help="Use a constant learning rate")

    # Sequence settings
    parser.add_argument("--src_max_len", type=int, required=True, help="Max source sequence length")
    parser.add_argument("--tgt_max_len", type=int, required=True, help="Max target sequence length")

    # Training state
    parser.add_argument("--curr_epoch", type=int, required=True, help="Current epoch (for resuming)")
    parser.add_argument("--use_half_precision", action="store_true", help="Enable FP16 training")


    # Data loading
    parser.add_argument("--train_shuffle", type=bool, default=False, help="Shuffle training data")
    parser.add_argument("--valid_shuffle", type=bool, default=False, help="Shuffle validation data")
    parser.add_argument("--pin_memory", type=bool, default=False, help="Enable pinned memory for data loading")

    # Distributed training
    parser.add_argument("--world_size", type=int, default=1, help="Number of processes (distributed training)")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed training backend")
    parser.add_argument("--resume_best", type=bool, default=False, help="Resume best model")
    parser.add_argument("--run_id", type=str, default=None, help="WandB run ID to resume")

    # Vocabulary settings
    parser.add_argument("--src_voc_size", type=int, default=None, help="Source vocabulary size")
    parser.add_argument("--tgt_voc_size", type=int, default=None, help="Target vocabulary size")

    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=3, help="Checkpoint save frequency (epochs)")
    parser.add_argument("--save_last", type=bool, default=False, help="Save the last model")
    parser.add_argument("--save_limit", type=int, default=5, help="Maximum number of saved checkpoints")

    # Logging & debugging
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--update_lr", type=float, default=None, help="Updated learning rate")
    parser.add_argument("--end_lr", type=float, default=1e-8, help="Final learning rate")
    parser.add_argument("--clip_grad_norm", type=float, default=-1, help="Gradient clipping threshold (-1 to disable)")
    parser.add_argument("--log_freq", type=int, default=50, help="Logging frequency (steps)")
    parser.add_argument("--test_freq", type=int, default=10, help="Testing frequency (steps)")
    parser.add_argument("--truncate", type=bool, default=False, help="Truncate sequences")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode")

    # Experimental settings
    parser.add_argument("--to_replace", type=bool, default=False, help="Replace index and momentum terms")
    parser.add_argument("--index_pool_size", type=int, default=100, help="Index token pool size")
    parser.add_argument("--momentum_pool_size", type=int, default=100, help="Momentum token pool size")

    return parser.parse_args()


def create_config_from_args(args):
    return TransformerConfig(
        project_name=args.project_name,
        run_name=args.run_name,
        model_name=args.model_name,
        root_dir=args.root_dir,
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        training_batch_size=args.training_batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        embedding_size=args.embedding_size,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        src_max_len=args.src_max_len,
        tgt_max_len=args.tgt_max_len,
        curr_epoch=args.curr_epoch,
        optimizer_lr=args.optimizer_lr,
        is_constant_lr = args.is_constant_lr,
        use_half_precision=args.use_half_precision,
        train_shuffle=args.train_shuffle,
        valid_shuffle=args.valid_shuffle,
        pin_memory=args.pin_memory,
        world_size=args.world_size,
        resume_best=args.resume_best,
        run_id=args.run_id,
        backend=args.backend,
        src_voc_size=args.src_voc_size,
        tgt_voc_size=args.tgt_voc_size,
        save_freq=args.save_freq,
        test_freq = args.test_freq,
        save_limit=args.save_limit,
        seed=args.seed,
        update_lr=args.update_lr,
        end_lr=args.end_lr,
        clip_grad_norm=args.clip_grad_norm,
        save_last=args.save_last,
        log_freq=args.log_freq,
        debug=args.debug,
        to_replace=args.to_replace,
        index_pool_size=args.index_pool_size,
        momentum_pool_size=args.momentum_pool_size
    )