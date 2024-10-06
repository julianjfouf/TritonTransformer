import torch
import torch.nn as nn
import torch.nn.functional as F

# triton kernels
from kernels.embedding import embedding
from kernels.mlp import mlp
from kernels.linear_residual import linear_residual
from kernels.linear import linear
from kernels.cross_entropy import cross_entropy_loss
from kernels.layer_norm import layer_norm
from kernels.attention import attention

class MLP(nn.Module):

    def __init__(self, n_hidden):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_hidden, 4 * n_hidden, bias=False, dtype=torch.float16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * n_hidden, n_hidden, bias=False, dtype=torch.float16)

    def forward(self, residual, x):
        return mlp(residual, x, self.fc1.weight, self.fc2.weight, activation="ReLU")

class Attention(nn.Module):

    def __init__(self, n_hidden, n_heads, head_size):
        super(Attention, self).__init__()
        self.qkv = nn.Linear(n_hidden, 3 * n_heads * head_size, bias=False, dtype=torch.float16)
        self.o = nn.Linear(n_heads * head_size, n_hidden, bias=False, dtype=torch.float16)
        self.n_heads = n_heads

    def forward(self, residual, x):
        B, T, C = x.shape

        qkv = linear(x, self.qkv).reshape(B, T, 3, self.n_heads, -1).permute(0, 2, 3, 1, 4) # B, T, 3 * H * C -> B, T, 3, H, C -> B, 3, H, T, C
        
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        # x = attention(q, k, v) # B, H, T, C, noticeable innacurracy from attention
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = x.permute(0, 2, 1, 3).reshape(B, T, C) # B, H, T, C -> B, T, H, C -> B, T, H * C = B, T, C

        x = linear_residual(residual, x, self.o.weight)

        return x

class Block(nn.Module):

    def __init__(self, n_hidden, n_heads, head_size):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_hidden, dtype=torch.float16)
        self.attn = Attention(n_hidden, n_heads, head_size)
        self.ln2 = nn.LayerNorm(n_hidden, dtype=torch.float16)
        self.mlp = MLP(n_hidden)

    def forward(self, x):
        # x = self.attn(x, layer_norm(x, self.ln1.weight, self.ln1.bias))
        x = self.attn(x, self.ln1(x))
        # x = self.mlp(x, layer_norm(x, self.ln2.weight, self.ln2.bias))
        x = self.mlp(x, self.ln2(x))

        return x

class TritonTransformer(nn.Module):

    def __init__(self, vocab_size, context_length, n_layers, n_hidden, n_heads, head_size):
        super(TritonTransformer, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_hidden, dtype=torch.float16)
        self.pos_emb = nn.Embedding(context_length, n_hidden, dtype=torch.float16)

        self.blocks = nn.ModuleList([
            Block(n_hidden, n_heads, head_size) for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(n_hidden, dtype=torch.float16)
        self.lm_head = nn.Linear(n_hidden, vocab_size, bias=False, dtype=torch.float16)

        self.lm_head.weight = self.tok_emb.weight

        print(sum(p.numel() for p in self.parameters())/1e6, "million parameters")

    def forward(self, x, targets=None):
        B, T = x.shape

        x = embedding(x, self.tok_emb.weight, self.pos_emb.weight, dtype=torch.float16)

        for block in self.blocks:
            x = block(x)
        
        # x = layer_norm(x, self.ln.weight, self.ln.bias)
        x = self.ln(x)

        x = linear(x, self.lm_head)

        if targets is not None:
            loss = cross_entropy_loss(x.reshape(B * T, -1), targets.reshape(B * T))
        else:
            loss = None
        
        return x, loss