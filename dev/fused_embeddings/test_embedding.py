import torch
import torch.nn as nn

from embedding import embedding

def test_embedding():
    B = 4
    T = 1024
    vocab_size = 1024
    n_hidden = 512
    ids = torch.randint(0, vocab_size, size=(B, T), device="cuda")

    tok_embedding = nn.Embedding(vocab_size, n_hidden, device="cuda")
    pos_embedding = nn.Embedding(T, n_hidden, device="cuda")

    output_triton = embedding(ids, tok_embedding.weight, pos_embedding.weight)
    output_torch = tok_embedding(ids) + pos_embedding(torch.arange(0, T, device="cuda"))

    if torch.allclose(output_triton, output_torch):
        print("All ok!")
    else:
        print("Something is wrong!")


test_embedding()