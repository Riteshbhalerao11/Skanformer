import torch
import torch.nn as nn
from torch.nn import Transformer
from torch import Tensor
import math
from typing import List, Union

def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    i_n1 = ratio * i_n
    return i_n1

class SineKANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device: Union[str, int] = 'cuda', grid_size=5, is_first=False, add_bias=True, norm_freq=True):
        super(SineKANLayer, self).__init__()
        self.grid_size = grid_size
        self.device = device
        self.is_first = is_first
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A, self.K, self.C = 0.9724108095811765, 0.9884401790754128, 0.999449553483052

        self.grid_norm_factor = (torch.arange(grid_size) + 1)
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)

        if is_first:
            self.amplitudes = torch.nn.Parameter(torch.empty(
                output_dim, input_dim, 1).normal_(0, .4) / output_dim / self.grid_norm_factor)
        else:
            self.amplitudes = torch.nn.Parameter(torch.empty(
                output_dim, input_dim, 1).uniform_(-1, 1) / output_dim / self.grid_norm_factor)

        grid_phase = torch.arange(
            1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(
            0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = torch.nn.Parameter(torch.arange(
                1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = torch.nn.Parameter(torch.arange(
                1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        # self.phase = torch.nn.Parameter(phase)
        self.register_buffer('phase', phase)

        if self.add_bias:
            self.bias = torch.nn.Parameter(
                torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y
    
class KANFeedForwardBlock(nn.Module):

    def __init__(self, in_size: int, ff_dims: List[int], grid_size: int = 8, device: Union[str, int] = 'cuda') -> None:
        super().__init__()
        self.ffn = torch.nn.ModuleList()
        local_ff_dims = ff_dims.copy()
        for i, d in enumerate(local_ff_dims):
            self.ffn.append(SineKANLayer(
                in_size, d, grid_size=grid_size, device=device, is_first=(i == 0)))
            in_size = d

    def forward(self, x):
        for f in self.ffn:
            x = f(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer architectures.

    Args:
        emb_size (int): The embedding size.
        dropout (float): Dropout rate.
        maxlen (int, optional): Maximum sequence length. Defaults to 5000.
    """

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """
    Token embedding module for transformer architectures.

    Args:
        vocab_size (int): Size of the vocabulary.
        emb_size (int): The embedding size.
    """

    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Model(nn.Module):
    """
    Transformer-based model for sequence-to-sequence tasks.

    Args:
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        emb_size (int): Size of the embedding.
        nhead (int): Number of attention heads.
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 512.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 ff_dims: List[int] = [2048, 1024, 512],
                 device: Union[str, int] = 'cuda'
                 ):
        super(Model, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.KAN = KANFeedForwardBlock(emb_size,ff_dims,device=device)
        self.generator = nn.Linear(ff_dims[-1], tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        """
        Forward pass of the model.

        Args:
            src (Tensor): Source input.
            trg (Tensor): Target input.
            src_mask (Tensor): Mask for source input.
            tgt_mask (Tensor): Mask for target input.
            src_padding_mask (Tensor): Padding mask for source input.
            tgt_padding_mask (Tensor): Padding mask for target input.
            memory_key_padding_mask (Tensor): Padding mask for memory.

        Returns:
            Tensor: Output tensor.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        outs = self.KAN(outs)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        """
        Encode the source input.

        Args:
            src (Tensor): Source input.
            src_mask (Tensor): Mask for source input.

        Returns:
            Tensor: Encoded tensor.
        """
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        """
        Decode the target input.

        Args:
            tgt (Tensor): Target input.
            memory (Tensor): Memory tensor.
            tgt_mask (Tensor): Mask for target input.

        Returns:
            Tensor: Decoded tensor.
        """
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


