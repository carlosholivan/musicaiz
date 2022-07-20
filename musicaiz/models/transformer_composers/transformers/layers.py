import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
  

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device: str):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.position_emb = PositionalEncoding(
            d_model=embedding_dim
        )

    def forward(self, input_ids):
        x = self.token_emb(input_ids) * math.sqrt(self.vocab_size)
        x = self.position_emb(x.type(torch.IntTensor).to(self.device))
        return x


class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection,self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 4)
        self.w_2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(p=dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
