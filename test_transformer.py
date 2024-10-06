import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtransformer import TorchTransformer
from tritontransformer import TritonTransformer

if __name__ == "__main__":
    batch_size = 4
    context_length = 1024
    vocab_size = 50257
    n_hidden = 768
    n_heads = 12
    head_size = n_hidden // n_heads
    n_layers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_input = torch.randint(0, vocab_size, size=(batch_size, context_length), device=device)
    sample_targets = torch.randint(0, vocab_size, size=(batch_size, context_length), device=device)
    model_torch = TorchTransformer(vocab_size, context_length, n_layers, n_hidden, n_heads, head_size).to(device)
    model_torch.requires_grad_ = False
    model_triton = TritonTransformer(vocab_size, context_length, n_layers, n_hidden, n_heads, head_size).to(device)
    model_triton.requires_grad_ = False

    for key in model_torch.state_dict():
        model_triton.state_dict()[key].copy_(model_torch.state_dict()[key])

    output_torch, loss_torch = model_torch(sample_input, sample_targets)
    output_triton, loss_triton = model_triton(sample_input, sample_targets)

    if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2):
        print("All ok!")
    else:
        print("Something is wrong!")

    diffs = torch.abs(output_triton - output_torch)
    print("avg diff:", torch.mean(diffs))
    print("max diff:", torch.max(diffs))
    print("min diff:", torch.min(diffs))

    max_values, indicesC = torch.max(diffs, dim=-1)
    max_values, indicesT = torch.max(max_values, dim=-1)
    for b in range(batch_size):
        index = (b, indicesT[b], indicesC[b, indicesT[b]])
        print(f"triton: {output_triton[index]} | torch: {output_torch[index]}")
    
    if torch.allclose(loss_triton, loss_torch):
        print("Loss is all ok!")
    else:
        print("Loss is wrong!")
        print("torch:", loss_torch)
        print("triton:", loss_triton)