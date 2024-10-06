import torch
import torch.nn as nn

from mlp import mlp2

def test_mlp():
    B=4
    T=1024
    C=768
    x = 100 * torch.rand(size=(B, T, C), device="cuda", dtype=torch.float16)

    ln = nn.LayerNorm(C, device="cuda", dtype=torch.float16)

    fc1 = nn.Linear(C, 4 * C, bias=False, device="cuda", dtype=torch.float16)
    relu = nn.ReLU()
    fc2 = nn.Linear(4 * C, C, bias=False, device="cuda", dtype=torch.float16)

    before = x.clone()
    output_triton = mlp2(x, ln(x), fc1.weight, fc2.weight, activation="ReLU")#mlp2(x, x, fc1.weight, fc2.weight, activation="ReLU")
    output_torch = x + fc2(relu(fc1(ln(x))))#x + fc2(relu(fc1(x)))

    if torch.allclose(x, before):
        print("x is preserved")
    else:
        print("x is not preserverd")
        
    if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2):
        print("All ok!")
    else:
        print("Something is wrong!")
        print(output_triton)
        print(output_torch)

    diffs = output_triton - output_torch
    print("avg diff:", torch.mean(diffs))
    print("max diff:", torch.max(diffs))
    print("min diff:", torch.min(diffs))

test_mlp()