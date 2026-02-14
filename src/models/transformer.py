import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, config["embed_dim"])
        self.pos_embedding = nn.Embedding(config["block_size"], config["embed_dim"])

        self.blocks = nn.Sequential(*[
            Block(config["embed_dim"], config["num_heads"], config["dropout"])
            for _ in range(config["num_layers"])
        ])

        self.ln = nn.LayerNorm(config["embed_dim"])
        self.head = nn.Linear(config["embed_dim"], vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)
