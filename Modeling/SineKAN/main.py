import torchtext; torchtext.disable_torchtext_deprecation_warning()
import pandas as pd
import numpy as np
import random
import torch
import os

from .fn_utils import create_config_from_args, create_tokenizer, init_distributed_mode,  parse_args
from .trainer import Trainer

def main(config, df_train,df_valid,tokenizer,src_vocab,tgt_vocab,tgt_itos):
    print(config.to_dict())
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(config,df_train,df_valid,tokenizer,src_vocab,tgt_vocab,tgt_itos)
    trainer.fit()


if __name__ == '__main__':

    args = parse_args()
    config = create_config_from_args(args)

    config.world_size = int(os.environ['WORLD_SIZE'])
    
    config.optimizer_lr *= config.world_size

    init_distributed_mode(config)

    df_train = pd.read_csv(config.data_dir+"train.csv")
    df_test = pd.read_csv(config.data_dir+"test.csv")
    df_valid = pd.read_csv(config.data_dir+"valid.csv")

    df = pd.concat([df_train,df_valid,df_test]).reset_index(drop=True)

    tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos = create_tokenizer(df,config,
                                                                           config.index_pool_size,config.momentum_pool_size)
    config.src_voc_size = len(src_vocab)
    config.tgt_voc_size = len(tgt_vocab)    
    
    if config.debug:
        config.epochs = 2
        df_train = df_train.sample(100).reset_index(drop=True)
        df_valid = df_valid.sample(100).reset_index(drop=True)
    
    print(f"TRAIN SAMPLES : {df_train.shape}")
    print("Data loading complete")

    main(config,df_train,df_valid,tokenizer,src_vocab,tgt_vocab,tgt_itos)
    torch.distributed.destroy_process_group()