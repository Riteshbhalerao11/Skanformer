from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for Transformer training."""

    # Project & model details
    project_name: str
    run_name: str
    model_name: str

    # Directories
    root_dir: str  # Path for storing checkpoints
    data_dir: str  # Path for dataset storage

    # Device configuration
    device: str  # "cuda" or "cpu"

    # Training hyperparameters
    epochs: int
    training_batch_size: int
    valid_batch_size: int
    num_workers: int

    # Transformer model architecture
    embedding_size: int
    hidden_dim: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int

    # Optimization settings
    warmup_ratio: float
    dropout: float
    weight_decay: float
    optimizer_lr: float
    is_constant_lr: bool  # Whether to use a constant learning rate

    # Sequence settings
    src_max_len: int
    tgt_max_len: int

    # Training state
    curr_epoch: int
    use_half_precision: bool  # Enable FP16 training

    # Data loading
    train_shuffle: bool
    valid_shuffle: bool
    pin_memory: bool  # Use pinned memory for faster GPU data transfer

    # Distributed training
    world_size: int
    backend: Optional[str] = "nccl"  # Backend for distributed training
    resume_best: bool = False
    run_id: Optional[str] = None  # WandB run_id to resume

    # Vocabulary
    src_voc_size: Optional[int] = None
    tgt_voc_size: Optional[int] = None

    # Checkpointing
    save_freq: Optional[int] = 3
    save_last: Optional[bool] = True
    save_limit: Optional[int] = 5

    # Logging & debugging
    seed: Optional[int] = 42  # Random seed for reproducibility
    update_lr: Optional[float] = None  # New learning rate (if updated)
    end_lr: Optional[float] = 1e-8  # Minimum learning rate
    clip_grad_norm: Optional[float] = -1  # Gradient clipping (-1 disables)
    log_freq: Optional[int] = 50  # Steps per log entry
    test_freq: Optional[int] = 10  # Steps per test run
    truncate: Optional[bool] = False  # Whether to truncate sequences
    debug: Optional[bool] = False  # Enable debug mode

    # Additional features
    to_replace: bool = False  # Replace index/momentum terms
    index_pool_size: int = 100  # Index token pool size
    momentum_pool_size: int = 100  # Momentum token pool size

    def to_dict(self):
        """Convert configuration to a dictionary."""
        return asdict(self)



@dataclass
class TransformerTestConfig:

    # Model name
    model_name: str

    # Directory where data and model checkpoints will be stored
    root_dir: str

    data_dir: str

    # Device for training (e.g., "cuda" for GPU, "cpu")
    device: str

    # Dimensionality of word embeddings
    embedding_size: int

    # Dimensionality of hidden layers in the transformer model
    hidden_dim: int

    # Number of attention heads in the transformer model
    nhead: int

    # Number of encoder layers in the transformer model
    num_encoder_layers: int

    # Number of decoder layers in the transformer model
    num_decoder_layers: int

    # Dropout rate
    dropout: float
    
    # Maximum length of source and target sequences
    src_max_len: int
    tgt_max_len: int


    # Size of vocabulary for source and target sequences
    src_voc_size: Optional[int] = None
    tgt_voc_size: Optional[int] = None

    # Seed for reproducibility
    seed: Optional[int] = 42

    #to replace index and momentum
    to_replace: bool = False


    #token pool sizes
    index_pool_size : int = 100   
    momentum_pool_size : int = 100

    # if debug
    debug: Optional[bool] = False

    # trucate sequences
    truncate: Optional[bool]= False

    def to_dict(self):
        return asdict(self)