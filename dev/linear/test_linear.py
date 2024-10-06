import torch
import torch.nn as nn

from linear import linear

def test_linear():
    B = 8
    T = 1024
    IN = 768
    OUT = 50257
    x = torch.randn(size=(B, T, IN), device="cuda", dtype=torch.float16)
    fc = nn.Linear(IN, OUT, bias=False, device="cuda", dtype=torch.float16)

    output_triton = linear(x, fc)
    output_torch = fc(x)

    diffs = torch.abs(output_triton - output_torch)
    print("avg diff:", torch.mean(diffs))
    print("max diff:", torch.max(diffs))
    print("min diff:", torch.min(diffs))

    # diffs = torch.abs(output_triton - output_torch)
    # max_values, indicesC = torch.max(diffs, dim=-1)
    # max_values, indicesT = torch.max(max_values, dim=-1)
    # print("avg diff:", torch.mean(diffs))
    # print("max diff:", max_values)
    # print("min diff:", torch.min(diffs))
    # print(max_values.shape)
    # print(indicesC.shape)
    # print(indicesT.shape)

    # for b in range(8):
    #     index = (b, indicesT[b], indicesC[b, indicesT[b]])
    #     print(f"triton: {output_triton[index]} | torch: {output_torch[index]}")

test_linear()