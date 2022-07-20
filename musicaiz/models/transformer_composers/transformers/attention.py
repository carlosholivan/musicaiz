import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        device: str,
        dropout: float = 0.1,
        causal: bool = False
    ):
        super(MultiheadAttention, self).__init__()

        self.n_head = n_heads
        self.d_model = embed_dim
        self.d_k = self.d_v = embed_dim // n_heads
        self.causal = causal
        self.device = device

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)

        self.self_attention = self_attention
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask = True):
        batch_num = query.size(0)

        query = self.w_q(query).view(batch_num, -1, self.n_head, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_num, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_num, -1, self.n_head, self.d_k).transpose(1, 2)

        attention_result, attention_score = self.self_attention(query, key, value, self.device, mask, self.causal)
        attention_result = attention_result.transpose(1,2).contiguous().view(batch_num, -1, self.n_head * self.d_k)
        attn_output = self.w_o(attention_result)

        return attn_output


def self_attention(query, key, value, device: str, mask=True, causal=False):
    key_transpose = torch.transpose(key, -2, -1)
    matmul_result = torch.matmul(query, key_transpose)
    d_k = query.size()[-1]
    attention_score = matmul_result/math.sqrt(d_k)

    if mask:
        mask = (torch.triu(torch.ones((query.size()[2], query.size()[2]))) == 1)
        mask = mask.transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, -1e20)
        attention_score = mask.masked_fill(mask == 1, float(0.0))
    
    if causal:
        query_len = query.size()[2]
        i, j = torch.triu_indices(query_len, query_len, 1)
        attention_score[i, j] = -1e4

    softmax_attention_score = F.softmax(attention_score, dim=-1).to(device)

    # When training with fp16, `softmax_attention_score` music be float16 but not when generating
    try:
        result = torch.matmul(softmax_attention_score, value)
    except:
        result = torch.matmul(softmax_attention_score.to(torch.float16), value)

    return result, softmax_attention_score
