import torch
import torch.nn as nn

from layer_norm import layer_norm1, layer_norm2, layer_norm3

def test_layer_norm():
    B = 1
    T = 1024
    C = 768
    ln_fp16 = nn.LayerNorm(C, device="cuda", dtype=torch.float16)
    ln_fp32 = nn.LayerNorm(C, device="cuda", dtype=torch.float32)
    ln_fp64 = nn.LayerNorm(C, device="cuda", dtype=torch.float64)
    x = 500.0 * torch.rand(size=(B, T, C), device="cuda", dtype=torch.float16)

    before = x.clone()
    output_triton = layer_norm3(x, ln_fp16.weight, ln_fp16.bias)
    output_triton_fp16 = output_triton.to(torch.float16)
    output_torch = ln_fp32(x.to(torch.float32)).to(torch.float16)
    output_torch_fp16 = output_torch.to(torch.float16)

    triton_diffs = torch.abs(output_triton - output_triton_fp16)
    print("triton avg diff:", torch.mean(triton_diffs))
    print("triton max diff:", torch.max(triton_diffs))
    print("triton min diff:", torch.min(triton_diffs))

    torch_diffs = torch.abs(output_torch - output_torch_fp16)
    print("torch avg diff:", torch.mean(torch_diffs))
    print("torch max diff:", torch.max(torch_diffs))
    print("torch min diff:", torch.min(torch_diffs))

    if torch.allclose(x, before):
        print("x is preserved")
    else:
        print("x is not preserved")
        diffs = torch.abs(x - before)
        print("mean x diff:", torch.mean(diffs))

    if torch.allclose(output_triton, output_torch):
        print("All ok!")
    else:
        print("Something is wrong!")

    diffs = torch.abs(output_triton_fp16 - output_torch_fp16)
    print("avg diff:", torch.mean(diffs))
    print("max diff:", torch.max(diffs))
    print("min diff:", torch.min(diffs))

    max_values, indicesC = torch.max(diffs, dim=-1)
    max_values, indicesT = torch.max(max_values, dim=-1)
    for b in range(B):
        index = (b, indicesT[b], indicesC[b, indicesT[b]])
        print(f"triton_fp32: {output_triton[index]} | torch_fp32: {output_torch[index]}")
        print(f"triton_fp16: {output_triton_fp16[index]} | torch_fp16: {output_torch_fp16[index]}")

test_layer_norm()