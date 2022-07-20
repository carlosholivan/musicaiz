from torch import nn

from musicaiz.models.transformer_composers.transformers.attention import MultiheadAttention
from musicaiz.models.transformer_composers.transformers.layers import (
    ResidualConnection,
    FeedForward,
    Embedding
)


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_decoders,
        sequence_len,
        n_heads,
        device: str,
        causal=False
    ):
        super(GPT2, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_decoders = n_decoders
        self.sequence_len = sequence_len
        self.n_heads = n_heads
        self.causal = causal
        self.device = device

        # Embedding
        self.embedding = Embedding(
            vocab_size=vocab_size, embedding_dim=embedding_dim, device=device
        )

        self.decoders = nn.Sequential(
          *[Decoder(
            d_model=embedding_dim, n_head=n_heads, device=self.device, dropout=0.1
          ) for _ in range(n_decoders)]
        )

        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids.long())
        x = self.decoders(x)
        lm_logits = self.lm_head(x)

        return lm_logits


class Decoder(nn.Module):
  def __init__(self, d_model, n_head, dropout, device, causal=True):
    super(Decoder,self).__init__()

    self.masked_multi_head_attention = MultiheadAttention(
      embed_dim=d_model, n_heads=n_head, causal=causal, device=device
    )
    self.residual_1 = ResidualConnection(d_model, dropout=dropout)
    self.feed_forward= FeedForward(d_model)
    self.residual_2 = ResidualConnection(d_model, dropout=dropout)


  def forward(self, x):
    x = self.residual_1(x, lambda x: self.masked_multi_head_attention(x, x, x))
    x = self.residual_2(x, self.feed_forward)

    return x